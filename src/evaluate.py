import faulthandler
import pickle
faulthandler.enable()

import warnings
import json
import os
from functools import partial
import sys
import time

# import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.training import inference_epoch
from utils.utils import  get_model

import os
import torch

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.loader import DataLoader


from datasets.pdbbind_phore import PhoreDataset, NoiseTransformPhore
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.utils import get_model
import yaml

RDLogger.DisableLog('rdApp.*')

def str2bool(inp):
    inp = inp.lower()
    if inp in ['y', 'yes', 'true', 't']:
        return True
    else:
        return False

parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path and --ligand parameters')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein .pdb file')
parser.add_argument('--ligand', type=str, default=None, help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--esm_embeddings_path', type=str, default='data/esm2_output', help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
parser.add_argument('--keep_update', type=str2bool, default=False, help='whether to save the updates during inference.')
parser.add_argument('--sample_per_complex', type=int, default=10, help='Samples of per complex.')
parser.add_argument('--fitscore', type=str2bool, default=True, help='whether to calculate the fitscore of the predicted ligand-phore pair.')

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--cache_path', type=str, default='../data/cache', help='Folder from where to load/restore cached dataset')
parser.add_argument('--no_random', action='store_true', default=False, help='Use no randomness in reverse diffusion')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--ode', action='store_true', default=False, help='Use ODE formulation for inference')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for creating the dataset')
parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
parser.add_argument('--keep_local_structures', action='store_true', default=False, help='Keeps the local structure when specifying an input with 3D coordinates instead of generating them with RDKit')
parser.add_argument('--split_file', type=str, default='../data/splits/timesplit_no_lig_overlap_val', help='Select a dataset to test.')
parser.add_argument('--store_ranked_pose', type=str2bool, default=False, help='Whether to store the ranked docking poses.')
parser.add_argument('--overwrite', type=str2bool, default=False, help='Whether to store the ranked docking poses.')
parser.add_argument('--random_samples', type=int, default=0, help='Number of random noise to sample .')
parser.add_argument('--dataset', type=str, choices=['pdbbind', 'posebusters'], default='pdbbind', help='Evaluation on the specified dataset.')
parser.add_argument('--mode', type=str, default='complex', help='The mode of calculating pharmacophores for original data.')
parser.add_argument('--use_ancphore', type=str2bool, default=False, help='Whether to use the calculated pharmacophores in ancphore running directory.')
parser.add_argument('--only_dataset', type=str2bool, default=False, help='Whether to process the dataset only.')
args = parser.parse_args()

def main():
    """
    Performing ligand-pharmacophore fitting on the PoseBusters set and time-split test set of PDBbind
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    # args.out_dir = args.out_dir.replace("-", '__')
    cache_file = os.path.join(args.out_dir, 'inference_results.pkl')
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    score_model_args.keep_update = args.keep_update
    score_model_args.fitscore = args.fitscore
    score_model_args.sample_per_complex = args.sample_per_complex
    score_model_args.run_dir = args.out_dir
    score_model_args.store_ranked_pose = args.store_ranked_pose
    score_model_args.random_samples = args.random_samples
    score_model_args.dataset = 'pdbbind'

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    test_dataset = get_input(score_model_args)

    if args.only_dataset:
        print("[I] Process dataset only, exit.")
        return 

    if not os.path.exists(cache_file) or args.overwrite:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
        print(f'loading state dict from `{args.model_dir}/{args.ckpt}`')
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu')) 
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        inf_metrics = inference_epoch(model, test_dataset.graphs, device, t_to_sigma, score_model_args)
        if not args.fitscore:
            print("Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                    .format(inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
        else:
            print("Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f} | fitscore_gt0.7 {:.3f} fitscore_gt0.4 {:.3f}"
                    .format(inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5'], 
                            inf_metrics['fitscore_gt0.7'], inf_metrics['fitscore_gt0.4']))
        # if args.keep_update:
        pickle.dump(inf_metrics, open(cache_file, 'wb'))

    if args.keep_update:
        test_graphs = test_dataset.graphs  
        print("#"*50)
        print("Evaluating Performance Metrics")
        no_overlap_file = "../data/splits/timesplit_test_no_rec_overlap" if args.dataset == 'pdbbind' else "../data/splits/posebusters_test_no_overlap"
        test_no_overlap = [line.strip() for line in open(no_overlap_file, 'r').readlines()
                    if line.strip() != ""]  
        performance_matrics = evaluate_results(args.out_dir, test_graphs, test_no_overlap)
        for k, v in performance_matrics.items():
            print(f'{k}: {v}')


def get_input(score_model_args):
    """
    Process the input dataset of the evaluation of DiffPhore
    """
    root = '../data/PDBBind' if args.dataset == 'pdbbind' else '../data/PoseBusters'
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    transform = NoiseTransformPhore(t_to_sigma=t_to_sigma, no_torsion=score_model_args.no_torsion, 
                                    tank_args={}, contrastive=True, 
                                    return_node=getattr(score_model_args, 'return_node', False),
                                    contrastive_model_dir=score_model_args.contrastive_model_dir) \
                                        if getattr(score_model_args, 'contrastive', False) else None
    if not args.use_ancphore:
        # test_dataset = PhoreDataset(transform=transform, root='../data/PDBBind/', cache_path='../data/cache_torsion', split_path='../data/splits/timesplit_no_lig_overlap_val', 
        test_dataset = PhoreDataset(transform=transform, root='../data/PDBBind/', cache_path='../data/cache_torsion', split_path=args.split_file, 
                                    remove_hs=True, max_lig_size=None, save_single=False, use_sdf=True, near_phore=False, keep_original=True,
                                    consider_ex=True, neighbor_cutoff=5.0, ex_connected=True, use_LAS_constrains=True, tank_features=False, use_phore_rule=True,
                                    matching=False, popsize=20, maxiter=20, require_ligand=True, num_workers=args.num_workers, min_phore_num=3, max_phore_num=15)

    else:
        phore_path = "/home/worker/users/YJL/DiffPhore/experiments/baselines/output/align/"
        phore_source = os.path.join(phore_path, f"{args.dataset}/{args.mode}/ancphore/process/")
        ids = os.listdir(phore_source)
        phore_files = [os.path.join(phore_source, f"{idx}/{idx}{'_pharmacophore.phore' if args.mode == 'complex' else '_random_pharmacophore.phore'}") for idx in ids]
        ligand_files = [os.path.join(root, f"all/{idx}/{idx}_ligand.sdf") for idx in ids]
        print(f'[I] Loading previously calculated pharmacophores from `{phore_source}`')
        # print(f'[I] phore_files: `{phore_files}`')
        # print(f'[I] ligand_files: `{ligand_files}`')
        
        # x['ligand_description'], x['phore']
        protein_ligand_records =[{'ligand_description': ligand_files[idx], 'phore': phore_file} \
            for idx, phore_file in enumerate(phore_files) if os.path.exists(phore_file) and os.path.exists(ligand_files[idx])]
        # print(f"[I] protein_ligand_records: {protein_ligand_records}")

        test_dataset = PhoreDataset(transform=transform, root='', cache_path=args.cache_path, split_path="", 
                                protein_ligand_records=protein_ligand_records, dataset=f'EV4{args.dataset}{args.mode[0]}',
                                num_workers=args.num_workers, keep_local_structures=args.keep_local_structures,
                                remove_hs=True, max_lig_size=None, save_single=False, use_sdf=True, near_phore=False, keep_original=True,
                                consider_ex=True, neighbor_cutoff=5.0, ex_connected=True, use_LAS_constrains=True, 
                                tank_features=False, use_phore_rule=True, matching=False, popsize=20, maxiter=20, require_ligand=True)
        score_model_args.save_single = test_dataset.save_single
    
    print("Number of test samples:", len(test_dataset))
    if len(test_dataset) < 1:
        print("[E] No valid test samples, exist")
        exit(-1)

    return test_dataset


def evaluate_results(inference_path, test_graphs, test_no_overlap, topk=[1, 5, 10]):
    """
    Evaluate the ligand-pharmacophore matching performance of DiffPhore by RMSD and fitness score (from AncPhore)
    Args: 
        inference_path: The path to store the result files.
        test_graphs: The process graphs of the samples to be tested (HeteroData).
        test_no_overlap: The name list (List or file) recording the sample ids not overlapping with the training set.
        topk: Choose top K ranked poses to evaluate the performance.
    
    Return:
        performance_metrics: The dict recording the performance results.

    """
    performance_metrics = {}
    inference_file = os.path.join(inference_path, 'inference_results.pkl')
    inference_results = pickle.load(open(inference_file, 'rb'))
    failed_indices = inference_results['failed_indices'] if 'failed_indices' in inference_results else []

    # test_no_overlap = [line.strip() for line in open("../data/splits/timesplit_test_no_rec_overlap").readlines()
    #                 if line.strip() != ""] 
    # no_overlap = np.array([inference_results['dock_process']['name'].index(name) for name in test_no_overlap \
    #                        if name in inference_results['dock_process']['name']])
    df_no_overlap = pd.DataFrame(inference_results['dock_process']['name'], columns=['name']).reset_index()
    df_no_overlap['no_overlap'] = df_no_overlap['name'].map(lambda x: any([name in x for name in test_no_overlap]))
    no_overlap = df_no_overlap[df_no_overlap['no_overlap']]['index'].values

    failed_indices = np.array(failed_indices).reshape(-1).tolist()
    test_graphs = [g for idx, g in enumerate(test_graphs) if idx not in failed_indices]
    centroid_distances, min_ex_cross_distances, \
        min_self_distances, min_base_cross_distances = analyze_pose_validity(inference_results, test_graphs)
    run_times = np.array(inference_results['run_time']) if 'run_time' in inference_results else None
    N = centroid_distances.shape[1]

    fitscore = np.array(inference_results['fitscore'] if [] not in inference_results['fitscore'] else \
        [m for m in inference_results['fitscore'] if m]).reshape(-1, N)
    rmsds = np.array(inference_results['rmsd']).reshape(-1, N)
    np.save(os.path.join(inference_path, 'rmsds.npy'), rmsds)
    np.save(os.path.join(inference_path, 'fitscore.npy'), fitscore)
    np.save(os.path.join(inference_path, 'centroid_distances.npy'), centroid_distances)
    np.save(os.path.join(inference_path, 'min_ex_cross_distances.npy'), min_ex_cross_distances)
    np.save(os.path.join(inference_path, 'min_self_distances.npy'), min_self_distances)
    np.save(os.path.join(inference_path, 'min_base_cross_distances.npy'), min_base_cross_distances)
    if run_times is not None:
        np.save(os.path.join(inference_path, 'run_times.npy'), run_times)

    topk = [k for k in topk if k <= N]
    perm_by_rmsd = np.argsort(rmsds, axis=1)
    perm_by_fitscore = np.argsort(fitscore, axis=1)[:, ::-1]
    
    
    for overlap in ['', 'no_overlap_']:
        index = np.arange(fitscore.shape[0]) if overlap == '' else no_overlap
        _rmsds = rmsds[index]
        _fitscore = fitscore[index]
        _centroid_distances = centroid_distances[index]
        _min_ex_cross_distances = min_ex_cross_distances[index]
        _min_self_distances = min_self_distances[index]
        # _min_base_cross_distances = min_base_cross_distances[index]
        _run_times = run_times[index] if run_times is not None else None
        if _run_times is not None:
            performance_metrics.update({
                f'{overlap}run_times_std': _run_times.std().__round__(2),
                f'{overlap}run_times_mean': _run_times.mean().__round__(2)}
            )
        perm = {
            'rmsd': perm_by_rmsd[index],
            'fitscore': perm_by_fitscore[index]
        }
        performance_metrics.update({
            # f'{overlap}run_times_std': run_times.std().__round__(2),
            # f'{overlap}run_times_mean': run_times.mean().__round__(2),
            f'{overlap}exclusion_clash_fraction': (
                        100 * (_min_ex_cross_distances < 1.0).sum() / len(_min_ex_cross_distances) / N).__round__(2),
            f'{overlap}self_intersect_fraction': (
                        100 * (_min_self_distances < 0.4).sum() / len(_min_self_distances) / N).__round__(2),
            f'{overlap}mean_rmsd': _rmsds.mean(),
            f'{overlap}rmsds_below_1': (100 * (_rmsds < 1).sum() / len(_rmsds) / N),
            f'{overlap}rmsds_below_2': (100 * (_rmsds < 2).sum() / len(_rmsds) / N),
            f'{overlap}rmsds_below_5': (100 * (_rmsds < 5).sum() / len(_rmsds) / N),
            f'{overlap}rmsds_percentile_25': np.percentile(_rmsds, 25).round(2),
            f'{overlap}rmsds_percentile_50': np.percentile(_rmsds, 50).round(2),
            f'{overlap}rmsds_percentile_75': np.percentile(_rmsds, 75).round(2),

            f'{overlap}mean_centroid': _centroid_distances.mean().__round__(2),
            f'{overlap}centroid_below_2': (100 * (_centroid_distances < 2).sum() / len(_centroid_distances) / N).__round__(2),
            f'{overlap}centroid_below_5': (100 * (_centroid_distances < 5).sum() / len(_centroid_distances) / N).__round__(2),
            f'{overlap}centroid_percentile_25': np.percentile(_centroid_distances, 25).round(2),
            f'{overlap}centroid_percentile_50': np.percentile(_centroid_distances, 50).round(2),
            f'{overlap}centroid_percentile_75': np.percentile(_centroid_distances, 75).round(2),

            f'{overlap}mean_fitscore': _fitscore.mean().__round__(2),
            f'{overlap}fitscore_above_0.7': (100 * (_fitscore > 0.7).sum() / len(_fitscore) / N).__round__(2),
            f'{overlap}fitscore_above_0.4': (100 * (_fitscore > 0.4).sum() / len(_fitscore) / N).__round__(2),
            f'{overlap}fitscore_percentile_25': np.percentile(_fitscore, 25).round(2),
            f'{overlap}fitscore_percentile_50': np.percentile(_fitscore, 50).round(2),
            f'{overlap}fitscore_percentile_75': np.percentile(_fitscore, 75).round(2),
        })


        for rankby in ['rmsd', 'fitscore']:
        # for rankby in ['rmsd', 'fitscore']:
            p = perm[rankby]
            _ranked_rmsd = np.take_along_axis(_rmsds, p, axis=1)
            _ranked_fitscore = np.take_along_axis(_fitscore, p, axis=1)
            _ranked_centroid_distances = np.take_along_axis(_centroid_distances, p, axis=1)
            _ranked_min_self_distances = np.take_along_axis(_min_self_distances, p, axis=1)
            _ranked_min_ex_cross_distances = np.take_along_axis(_min_ex_cross_distances, p, axis=1)
            _topk  = [1] if rankby == 'rmsd' else topk
            rankby = f"rankby{rankby.capitalize()}_" if rankby == 'fitscore' else ''
            for k in _topk:
                _ranked_rmsd_k = _ranked_rmsd[:, :k].min(axis=1)
                _ranked_fitscore_k = _ranked_fitscore[:, :k].mean(axis=1)
                # _ranked_fitscore_k = _ranked_fitscore[:, :k].min(axis=1)
                _ranked_centroid_distances_k = _ranked_centroid_distances[:, :k].min(axis=1)
                _ranked_min_self_distances_k = _ranked_min_self_distances[:, :k].min(axis=1)
                _ranked_min_ex_cross_distances_k = _ranked_min_ex_cross_distances[:, :k].min(axis=1)
                performance_metrics.update({
                    f'{overlap}{rankby}top{k}_exclusion_clash_fraction': (
                                100 * (_ranked_min_ex_cross_distances_k < 1.0).sum() / len(_ranked_min_ex_cross_distances_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_self_intersect_fraction': (
                                100 * (_ranked_min_self_distances_k < 0.4).sum() / len(_ranked_min_self_distances_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_rmsds_below_1': (100 * (_ranked_rmsd_k < 1).sum() / len(_ranked_rmsd_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_rmsds_below_2': (100 * (_ranked_rmsd_k < 2).sum() / len(_ranked_rmsd_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_rmsds_below_5': (100 * (_ranked_rmsd_k < 5).sum() / len(_ranked_rmsd_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_rmsds_percentile_25': np.percentile(_ranked_rmsd_k, 25).round(2),
                    f'{overlap}{rankby}top{k}_rmsds_percentile_50': np.percentile(_ranked_rmsd_k, 50).round(2),
                    f'{overlap}{rankby}top{k}_rmsds_percentile_75': np.percentile(_ranked_rmsd_k, 75).round(2), 

                    f'{overlap}{rankby}top{k}_centroid_below_2': (
                                100 * (_ranked_centroid_distances_k < 2).sum() / len(_ranked_centroid_distances_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_centroid_below_5': (
                                100 * (_ranked_centroid_distances_k < 5).sum() / len(_ranked_centroid_distances)).__round__(2),
                    f'{overlap}{rankby}top{k}_centroid_percentile_25': np.percentile(_ranked_centroid_distances_k, 25).round(2),
                    f'{overlap}{rankby}top{k}_centroid_percentile_50': np.percentile(_ranked_centroid_distances_k, 50).round(2),
                    f'{overlap}{rankby}top{k}_centroid_percentile_75': np.percentile(_ranked_centroid_distances_k, 75).round(2),
                    
                    f'{overlap}{rankby}top{k}_fitscore_above_0.7': (
                                100 * (_ranked_fitscore_k > 0.7).sum() / len(_ranked_fitscore_k)).__round__(2),
                    f'{overlap}{rankby}top{k}_fitscore_above_0.4': (
                                100 * (_ranked_fitscore_k > 0.4).sum() / len(_ranked_centroid_distances)).__round__(2),
                    f'{overlap}{rankby}top{k}_fitscore_percentile_25': np.percentile(_ranked_fitscore_k, 25).round(2),
                    f'{overlap}{rankby}top{k}_fitscore_percentile_50': np.percentile(_ranked_fitscore_k, 50).round(2),
                    f'{overlap}{rankby}top{k}_fitscore_percentile_75': np.percentile(_ranked_fitscore_k, 75).round(2)
                    }
                )
    
    json.dump(performance_metrics, open(os.path.join(inference_path, 'performance_metrics.json'), 'w'), indent=4)

    return performance_metrics


def analyze_spatial_info(docked_poses, origin_poses, graph):
    """
    Analyze spatial information of docked poses
    Args:
        docked_poses: np.ndarray of shape [B, N, 3]
        origin_poses: np.ndarray of shape [B, N, 3]
        phore_graph: 'phore' part of the HeteroData
    """
    ex_mask = graph['phore'].x[:, 0] == 10.0
    ex_pos = graph['phore'].pos[ex_mask].numpy().reshape(1, -1, 3) + graph.original_center.numpy()[None, :, :]
    centroid_distance = np.linalg.norm(docked_poses.mean(axis=1) - origin_poses.mean(axis=1), axis=1)
    ex_cross_distances = np.linalg.norm(ex_pos[:, :, None, :] - docked_poses[:, None, :, :], axis=-1)
    min_ex_cross_distances = np.min(ex_cross_distances, axis=(1, 2))
    self_distances = np.linalg.norm(docked_poses[:, :, None, :] - docked_poses[:, None, :, :], axis=-1)
    self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
    min_self_distances = np.min(self_distances, axis=(1, 2))
    base_cross_distances = np.linalg.norm(ex_pos[:, :, None, :] - origin_poses[:, None, :, :], axis=-1)
    min_base_cross_distances = np.min(base_cross_distances, axis=(1, 2))
    return centroid_distance, min_ex_cross_distances, min_self_distances, min_base_cross_distances


def analyze_pose_validity(inference_results, test_graphs):
    """
    Analyze the pose validity of the generated conformations
    Args:
        inference_results: The dict storing the docking process
        test_graphs: The processed input samples in Graph format
    
    Return:
        pose validity metrics including:
            centroid distances [arr_centroid_distances], 
            minimium distance to exclusion spheres of the generated conformations [arr_min_ex_cross_distances], 
            minimium intra-molecular distance [arr_min_self_distances],
            minimium distance to exclusion spheres of the groud truth conformations [arr_min_base_cross_distances]
    """
    centroid_distances = []
    min_ex_cross_distances = []
    min_self_distances = []
    min_base_cross_distances = []
    for idx, graph in enumerate(test_graphs):
        docked_poses = np.array(inference_results['dock_process']['docked_poses'][idx])[:, -1, :, :] +\
                       inference_results['dock_process']['original_centers'][idx]
        origin_poses = inference_results['dock_process']['origin_pos'][idx].reshape(1, -1, 3)
        centroid_distance, min_ex_cross_distance, min_self_distance, \
            min_base_cross_distance = analyze_spatial_info(docked_poses, origin_poses, graph)
        centroid_distances.append(centroid_distance)
        min_ex_cross_distances.append(min_ex_cross_distance)
        min_self_distances.append(min_self_distance)
        min_base_cross_distances.append(min_base_cross_distance)
    arr_centroid_distances = np.array(centroid_distances)
    arr_min_ex_cross_distances = np.array(min_ex_cross_distances)
    arr_min_self_distances = np.array(min_self_distances)
    arr_min_base_cross_distances = np.array(min_base_cross_distances)
    return arr_centroid_distances, arr_min_ex_cross_distances, arr_min_self_distances, arr_min_base_cross_distances


if __name__ == '__main__':
    st = time.time()
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current Working Dir: {os.getcwd()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.system("echo Current Hostname: $(hostname)")
    print(f'Current PID: {os.getpid()}')
    print(f"Current Command: {' '.join(sys.argv)}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Current GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    main()
    end = time.time()
    print(f"Job Finished! {end-st:.3f} seconds cost.")
