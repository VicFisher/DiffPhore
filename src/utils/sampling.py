import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.loader import DataLoader
from utils.diffusion_utils import modify_conformer, set_time, set_time_phore
from utils.torsion import modify_conformer_torsion_angles
from utils.geometry import rigid_transform_Kabsch_3D_torch
from datasets.process_pharmacophore import calc_phore_fitting
# from utils.training import calculate_fitscore
from datasets.process_mols import write_mol_with_multi_coords
import copy
from rdkit import Chem
import os


def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, keep_update=False):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            norm = complex_graph['ligand'].norm.reshape(-1, complex_graph['ligand'].x.shape[0], 3) \
                + complex_graph['ligand'].pos.unsqueeze(0) if hasattr(complex_graph['ligand'], 'norm') else None
            complex_graph['ligand'].pos, complex_graph['ligand'].norm = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates, 
                                                norm=norm)
            if keep_update:
                complex_graph.fw_tor_update = torsion_updates

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T 
        complex_graph['ligand'].norm = (complex_graph['ligand'].norm - molecule_center) @ random_rotation.T - complex_graph['ligand'].pos \
            if complex_graph['ligand'].norm is not None else None
        if complex_graph['ligand'].norm is not None:
            complex_graph['ligand'].norm = complex_graph['ligand'].norm.reshape(complex_graph['ligand'].pos.shape[0], -1)
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update
            if keep_update:
                complex_graph.fw_tr_update = tr_update.cpu().numpy()
        if keep_update:
            complex_graph.fw_rot_update = random_rotation.cpu().numpy()


def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            # Apply noise
            new_data_list.extend([modify_conformer(
                complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence


def sampling_phore(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=20, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time_phore(complex_graph_batch, t_tr, t_rot, t_tor, b, device)
            
            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            # Apply noise
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None, 
                                          keep_update=model_args.keep_update if hasattr(model_args, 'keep_update') else False)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time_phore(confidence_complex_graph_batch, 0, 0, 0, N, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence


def sampling_phore_with_fitscore(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)
    random_samples = model_args.random_samples if hasattr(model_args, 'random_samples') else 0

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time_phore(complex_graph_batch, t_tr, t_rot, t_tor, b, device)
            
            with torch.no_grad():
                tr_score, rot_score, tor_score = model(complex_graph_batch)

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
            d_sigma_tr = tr_g * np.sqrt(dt_tr)
            d_sigma_rot = rot_g * np.sqrt(dt_rot)
            last_step = (no_final_step_noise and t_idx == inference_steps - 1)
            if not no_random and random_samples > 1 and not last_step:
                d_sigma_tr = d_sigma_tr.unsqueeze(0)
                d_sigma_rot = d_sigma_rot.unsqueeze(0)
                # print(f"tr_score.shape: {tr_score.shape}")
                # print(f"rot_score.shape: {rot_score.shape}")
                # print(f"tor_score.shape: {tor_score.shape}")
                
                tr_score = tr_score.unsqueeze(0)
                rot_score = rot_score.unsqueeze(0)
                # tor_score = tor_score.unsqueeze(0)

            if ode:
                tr_perturb = (0.5 * d_sigma_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * d_sigma_rot ** 2 * rot_score.cpu()).cpu()
            else:
                if no_random or last_step:
                    tr_z = torch.zeros((b, 3)) 
                else:
                    tr_z = torch.normal(mean=0, std=1, size=(random_samples, b, 3) if random_samples > 1 else (b, 3))
                tr_perturb = (d_sigma_tr ** 2 * tr_score.cpu() + d_sigma_tr * tr_z).cpu().float()

                if no_random or last_step:
                    rot_z = torch.zeros((b, 3)) 
                else:
                    rot_z = torch.normal(mean=0, std=1, size=(random_samples, b, 3) if random_samples > 1 else (b, 3))
                rot_perturb = (d_sigma_rot ** 2 * rot_score.cpu() + d_sigma_rot * rot_z).cpu().float()
            
            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                d_sigma_tor = tor_g * np.sqrt(dt_tor)
                if not no_random and random_samples > 1:
                    d_sigma_tor = d_sigma_tor.unsqueeze(0)
                if ode:
                    tor_perturb = (0.5 * d_sigma_tor ** 2 * tor_score.cpu()).numpy()
                else:
                    if no_random or last_step:
                        tor_z = torch.zeros(tor_score.shape)
                    else:
                        size = (random_samples,) + tor_score.shape if random_samples > 1 else tor_score.shape
                        tor_z = torch.normal(mean=0, std=1, size=size)

                tor_perturb = (d_sigma_tor ** 2 * tor_score.cpu() + d_sigma_tor * tor_z).float().numpy()
                index = 1 if not no_random and random_samples > 1 and not last_step else 0
                # print(f"tor_perturb.shape: {tor_perturb.shape}")
                # print(f"index: {index}")
                torsions_per_molecule = tor_perturb.shape[index] // b
            else:
                tor_perturb = None
            
            if not no_random and random_samples > 1:
                tmp_data_list = []
                for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list()):
                    for j in range(random_samples):
                        tmp_data_list.extend([modify_conformer(copy.deepcopy(complex_graph), tr_perturb[j, i:i + 1], rot_perturb[j, i:i + 1].squeeze(0),
                                                tor_perturb[j, i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None, 
                                                keep_update=model_args.keep_update if hasattr(model_args, 'keep_update') else False)])
                
                filterHs = torch.not_equal(complex_graph_batch[0]['ligand'].x[:, 0], 0).cpu().numpy()
                ligand_pos = np.asarray(
                    [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in tmp_data_list])

                # phore_file = complex_graph_batch.phore_file[0]
                # print(f"complex_graph_batch.mol: {complex_graph_batch.mol}")
                # print(f"complex_graph_batch.mol[0]: {complex_graph_batch.mol[0]}")
                mol = Chem.RemoveAllHs(copy.deepcopy(complex_graph_batch.mol[0][0]))

                dock_pose = ligand_pos + complex_graph_batch[0].original_center.cpu().numpy()
                name = complex_graph_batch.name[0][0]
                phore_file = complex_graph_batch[0].phore_file[0][0] if hasattr(complex_graph_batch[0], 'phore_file') else None
                scores = calculate_fitscore(model_args, dock_pose, name, mol, store_ranked_pose=False, phore_file=phore_file)
                indexs = (torch.tensor(scores).view(random_samples, -1)).argmax(dim=0) + torch.arange(b) * random_samples
                new_data_list.extend([tmp_data_list[idx] for idx in indexs])


            else:
            # Apply noise
                new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                            tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None, 
                                            keep_update=model_args.keep_update if hasattr(model_args, 'keep_update') else False)
                            for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])

        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time_phore(confidence_complex_graph_batch, 0, 0, 0, N, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence


def calculate_fitscore(args, ligand_pos, name, mol, phore_file=None, store_ranked_pose=True):
    tmp_path = os.path.join(args.run_dir, f'mapping_process/{name}')
    os.makedirs(tmp_path, exist_ok=True)
    # print(f"[I] The docked poses will be stored at `{tmp_path}`")
    docked_file = os.path.join(tmp_path, f"{name}.sdf")
    write_mol_with_multi_coords(mol, ligand_pos, docked_file, name)
    if phore_file is None or not os.path.exists(phore_file):
        if args.dataset == 'zinc':
            phore_file = os.path.join(args.zinc_path, f"sample_phores/{name}.phore")
            # phore_file = os.path.join(args.zinc_path, f"ZINC20_LogPlt5_InStock/{args.flag}/sample_phores/{name}.phore")
        elif args.dataset == 'pdbbind':
            if args.flag == "phoreDedup":
                phore_file = os.path.join(args.data_dir, f"phore_dedup/{name}/{name}_complex.phore")
            else:
                phore_file = os.path.join(args.data_dir, f"phore/{name}/{name}_complex.phore")
        else:
            raise NotImplementedError
    # print(f'phore_file in calculate_score: {phore_file}')
    score_file = os.path.join(tmp_path, f"{name}.score")
    dbphore_file = os.path.join(tmp_path, f"{name}.dbphore")
    log_file = os.path.join(tmp_path, f"{name}.log")
    scores = calc_phore_fitting(docked_file, phore_file, score_file, dbphore_file, log_file, overwrite=True, 
                                fitness=getattr(args, 'fitness', 1))
    if store_ranked_pose and scores is not None:
        ranked_pose_path = os.path.join(args.run_dir, 'ranked_poses/')
        os.makedirs(ranked_pose_path, exist_ok=True)
        ranked_pose_file = os.path.join(ranked_pose_path, f'{name}_ranked.sdf')
    if store_ranked_pose and scores is not None:
        ranked_pose_path = os.path.join(args.run_dir, 'ranked_poses/')
        os.makedirs(ranked_pose_path, exist_ok=True)
        ranked_pose_file = os.path.join(ranked_pose_path, f'{name}_ranked.sdf')
        perm = np.argsort(np.array(scores))[::-1]
        # print(f"Perm: {perm.tolist()}")
        _ranked_ligand_pos = ligand_pos[perm]
        write_mol_with_multi_coords(mol, _ranked_ligand_pos, ranked_pose_file, name, 
                                    marker='rank', properties={'fitscore': np.array(scores)[perm]})

    return scores


def sample_step(complex_graph_batch, model, model_args, tr_sigma, rot_sigma, tor_sigma, delta_t=0.05,
                no_random=False, ode=False):
        b = complex_graph_batch.num_graphs
        # complex_graph_batch = complex_graph_batch.to(device)
        dt_tr, dt_rot, dt_tor = delta_t, delta_t, delta_t

        with torch.no_grad():
            tr_score, rot_score, tor_score = model(complex_graph_batch)

        tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
        rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))

        if ode:
            tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
            rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
        else:
            tr_z = torch.zeros((b, 3)) if no_random else torch.normal(mean=0, std=1, size=(b, 3))
            tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

            rot_z = torch.zeros((b, 3)) if no_random else torch.normal(mean=0, std=1, size=(b, 3))
            rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()
            
        if not model_args.no_torsion:
            tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
            if ode:
                tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
            else:
                tor_z = torch.zeros(tor_score.shape) if no_random else torch.normal(mean=0, std=1, size=tor_score.shape)
                tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
            
            torsions_per_molecule = tor_perturb.shape[0] // b
        else:
            tor_perturb = None

        # Apply noise
        data_list = [modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                        tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None, 
                                        keep_update=model_args.keep_update if hasattr(model_args, 'keep_update') else False)
                        for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())]
        return data_list, tor_perturb, tr_perturb, rot_perturb


def t_centered_A(A, _R, t):
    return A.mean(axis=0) @ _R.T - A.mean(axis=0) + t
# def t_centered_A(A, B, _R):
#     return B - (A - A.mean(axis=0)) @ _R.T - A.mean(axis=0)


def get_updates_from_0_to_n(g, g_n, torsion_updates):
    g_0 = copy.deepcopy(g)
    if torsion_updates is not None:
        flexible_new_pos, flexible_new_norm = modify_conformer_torsion_angles(g_0['ligand'].pos,
                                                           g_0['ligand', 'ligand'].edge_index.T[g_0['ligand'].edge_mask],
                                                           g_0['ligand'].mask_rotate if isinstance(g_0['ligand'].mask_rotate, np.ndarray) else g_0['ligand'].mask_rotate[0],
                                                           torsion_updates,
                                                           norm=None)
        flexible_new_pos = flexible_new_pos.to(g_0['ligand'].pos.device)
        _R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, g_0['ligand'].pos.T)
        aligned_flexible_pos = flexible_new_pos @ _R.T + t.T
        g_0['ligand'].pos = aligned_flexible_pos

    R1, t1 = rigid_transform_Kabsch_3D_torch(g_0['ligand'].pos.T, g_n['ligand'].pos.T)
    # t2 = t_centered_A(g_0['ligand'].pos, g_n['ligand'].pos, R1)
    t2 = t_centered_A(g_0['ligand'].pos, R1, t1.T)
    R1 = R.from_matrix(R1.numpy()).as_rotvec()
    return t2, R1
