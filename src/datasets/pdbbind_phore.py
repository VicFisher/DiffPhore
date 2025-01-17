import binascii
import copy
import os
import pickle
import random
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Subset
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from torch_geometric.data.batch import Batch

from rdkit import Chem
import yaml
from datasets.process_mols import (extract_pair_distribution, generate_conformer,
                                   generate_ligand_phore_feat,
                                   get_lig_graph_with_matching,
                                   lipinski_rule_analysis, read_molecule,
                                   write_mol_with_coords)
from datasets.process_pharmacophore import get_phore_graph, parse_phore, calc_phore_fitting, parse_score_file
from utils import so3, torus
from utils.diffusion_utils import modify_conformer, set_time_phore
from utils.utils import read_strings_from_txt
from utils.sampling import sample_step, get_updates_from_0_to_n


class NoiseTransformPhore(BaseTransform):
    """
    A class to apply noise transformations to molecular data for the purpose of generating random poses.
    Attributes:
        t_to_sigma (callable): Function to convert time to sigma values.
        no_torsion (bool): Flag to indicate whether torsion should be applied.
        epochs (int, optional): Number of epochs for training.
        reject (bool): Flag to indicate whether to reject samples based on certain criteria.
        cofactor (float): Cofactor used in rejection criteria.
        calc_fitscore (bool): Flag to indicate whether to calculate fit score.
        fitscore_tmp (str): Temporary directory for storing fit score files.
        delta_t (float): Initial delta time value.
        rate_from_infer (float): Rate of sampling from inference results.
        epoch_from_infer (int): Epoch to start sampling from inference results.
        dynamic_coeff (float): Coefficient for dynamic scheduling.
        model (torch.nn.Module, optional): Model used for inference.
        args (Namespace, optional): Arguments for the model.
        p (float, optional): Probability for dynamic scheduling.
    Methods:
        __call__(data): Applies noise transformation to the given data.
        apply_noise(data, t_tr, t_rot, t_tor, tr_update=None, rot_update=None, torsion_updates=None, debug=False): 
            Applies noise to the data based on the given parameters.
        sample_modification(data, tr_update, rot_update, torsion_updates, tr_sigma, rot_sigma, tor_sigma): 
            Samples modifications for translation, rotation, and torsion.
        step(): Increments the current epoch.
        reset_step(): Resets the current epoch to zero.
        set_step(epoch): Sets the current epoch to the given value.
        get_fitscore(data): Calculates the fit score for the given data.
        sample_from_infer(data_0, data, t_n_1, tr_sigma, rot_sigma, tor_sigma, torsion_updates=None, debug=False): 
            Samples data from inference results.
        from_infer(t): Determines whether to sample from inference results based on the current epoch and rate.
        update_model(state_dict): Updates the model with the given state dictionary.
        dynamic_schedule(epoch, max_rate=0.4, u=400, c=10): Calculates the dynamic schedule probability.
        time2delta(xt, steps=(20, 2000), eps=0.01, debug=False): Converts time to delta time.
    """
    def __init__(self, t_to_sigma, no_torsion, epochs=None, reject=False, cofactor=0.3, 
                 calc_fitscore=False,  fitscore_tmp='/tmp/diffphore/fitscore_tmp/', 
                 delta_t=0.05, rate_from_infer=0.1, epoch_from_infer=300, dynamic_coeff=0,
                 model=None, args=None, **kwargs):
        self.t_to_sigma = t_to_sigma
        self.no_torsion = no_torsion
        self.epochs = epochs
        self.reject = reject
        self.current_epoch = 0
        self.cofactor = cofactor
        # self.contrastive = contrastive
        # self.contrastive_model_dir = contrastive_model_dir
        # self.contrastive_model = None
        # self.return_node = return_node
        self.calc_fitscore = calc_fitscore
        self.fitscore_tmp = fitscore_tmp
        self.delta_t = delta_t
        self.delta_t0 = delta_t
        self.rate_from_infer = rate_from_infer
        self.epoch_from_infer = epoch_from_infer
        self.dynamic_coeff = dynamic_coeff
        self.p = None

        if self.calc_fitscore and not os.path.exists(fitscore_tmp):
            os.makedirs(fitscore_tmp, exist_ok=True)
        if self.rate_from_infer > 0:
            _model = copy.deepcopy(model)
            self.model = _model if not hasattr(_model, 'module') else _model.module
            self.model = self.model.cpu()
            self.model.device = torch.device('cpu')
        else:
            self.model = None
        self.args = args
        
    def __call__(self, data):
        count = 0
        data.skip = False
        while count < 5:
            count += 1
            try:
                t = np.random.uniform()
                if self.delta_t0 == 0:
                    self.delta_t = self.time2delta(t)
                t_tr, t_rot, t_tor = t, t, t
                data = self.apply_noise(data, t_tr, t_rot, t_tor)
                return data
            except Exception as e:
                if count == 5:
                    print(f"[W] Failed to generate random pose for `{data.name}` 5 times, skip the sample. {e}")
                    data.skip = True
                    # raise e
                else:
                    print(f"[W] Failed to generate random pose for `{data.name}`, try again. {e}")
                # self.add_contrastive_feat(data)
        return data

    def apply_noise(self, data, t_tr, t_rot, t_tor, 
                    tr_update=None, rot_update=None, 
                    torsion_updates=None, debug=False):
        from_infer = self.from_infer(t_tr)
        if not torch.is_tensor(data['ligand'].pos):
            data['ligand'].pos = random.choice(data['ligand'].pos)
        data_0 = copy.deepcopy(data) if from_infer else None
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        set_time_phore(data, t_tr, t_rot, t_tor, 1, device=None)

        tr_update, rot_update, torsion_updates = self.sample_modification(data, tr_update, rot_update, torsion_updates, tr_sigma, rot_sigma, tor_sigma)
        modify_conformer(data, tr_update, torch.from_numpy(rot_update).float(), torsion_updates)

        if not from_infer:
            self.get_fitscore(data)
            data.tr_score = -tr_update / tr_sigma ** 2
            data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
            data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
            
            data.tor_sigma_edge = None if self.no_torsion else np.ones(data['ligand'].edge_mask.sum()) * tor_sigma
            return data
        else:
            # print("[I] Sampling perturbed pose from inference result.")
            data_0, info = self.sample_from_infer(data_0, data, t_tr, tr_sigma, rot_sigma, tor_sigma, 
                                                  torsion_updates, debug=debug)
            if debug:
                info['tr_update'] = tr_update
                info['rot_update'] = rot_update
                info['torsion_updates'] = torsion_updates
                data_0.info = info
            return data_0

    # def add_contrastive_feat(self, data):
    #     if self.contrastive_model is not None:
    #         with torch.no_grad():
    #             single_batch = Batch.from_data_list([data])
    #             # lig_rep = self.contrastive_model.lig_encoder(single_batch, return_node=self.return_node)
    #             # phore_rep = self.contrastive_model.phore_encoder(single_batch, return_node=self.return_node)
    #             lig_rep, phore_rep = self.contrastive_model.encoder(single_batch, return_scalar=True) # type: ignore
    #             lig_rep, phore_rep = lig_rep, phore_rep
    #             data['ligand'].contrastive = lig_rep
    #             data['phore'].contrastive = phore_rep

    def sample_modification(self, data, tr_update, rot_update, torsion_updates, tr_sigma, rot_sigma, tor_sigma):
        to_reject = True
        while to_reject:
            to_reject = False
            tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3)) if tr_update is None else tr_update
            rot_update = so3.sample_vec(eps=rot_sigma) if rot_update is None else rot_update
            torsion_updates = np.random.normal(loc=0.0, scale=tor_sigma, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
            torsion_updates = None if self.no_torsion else torsion_updates
            if self.reject:
                assert self.epochs is not None
                x1 = np.random.uniform()
                x2 = np.random.uniform()
                y = self.cofactor * self.current_epoch / self.epochs
                if x1 <= y or x2 <= y:
                    T_ = tr_update.norm().item() / tr_sigma
                    R_ = float((rot_update ** 2).sum() ** 0.5) / rot_sigma
                    Theta_ = float(abs(torsion_updates).mean()) / tor_sigma if torsion_updates is not None and len(torsion_updates) > 0 else None
                    if (x1 <= y and (T_ > R_ or (Theta_ is not None and T_ > Theta_))) or (x2 <= y and (Theta_ is not None and R_ > Theta_)):
                        to_reject = True
                        # if self.debug:
                        #     print(f"[I] Epoch {self.current_epoch} `{data.name}` ({x:.3f} <= {y:.3f}): Sample Rejected T_({T_:.3f}) R_({R_:.3f}) Theta_({Theta_:.3f})")
                    # if self.debug:
                    #     print(f"[I] Epoch {self.current_epoch} `{data.name}`: T_({T_:.3f}) R_({R_:.3f}) Theta_({Theta_:.3f})")

        return tr_update, rot_update, torsion_updates

    def step(self):
        self.current_epoch += 1

    def reset_step(self):
        self.current_epoch = 0

    def set_step(self, epoch):
        self.current_epoch = epoch
        
        if self.current_epoch == self.epoch_from_infer and self.rate_from_infer > 0 and self.dynamic_coeff == 0:
            print(f"[I] Reach the set epoch `{self.epoch_from_infer}` to start sample training data from inference result with possibility of `{self.rate_from_infer}`")
        
        if self.dynamic_coeff > 0:
            self.p = self.dynamic_schedule(self.current_epoch, max_rate=self.rate_from_infer, 
                                                u=self.epoch_from_infer, c=self.dynamic_coeff) 

    def get_fitscore(self, data):
        if self.calc_fitscore:
            name = data.name
            lig_pos = data['ligand'].pos.numpy() + data.original_center.numpy()
            filterHs = torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy()
            mol = Chem.RemoveAllHs(copy.deepcopy(data.mol))
            mol.SetProp("_Name", name)
            tmp_out = os.path.join(self.fitscore_tmp, name)
            if not os.path.exists(tmp_out):
                os.makedirs(tmp_out, exist_ok=True)
            tmp_lig_file = os.path.join(tmp_out, "ligand.sdf")
            write_mol_with_coords(mol, lig_pos[filterHs], tmp_lig_file)
            phore_file = data.phore_file
            score_file = os.path.join(tmp_out, f"{name}.score")
            dbphore_file = os.path.join(tmp_out, f"{name}.dbphore")
            log_file = os.path.join(tmp_out, f"{name}.log")
            scores = calc_phore_fitting(tmp_lig_file, phore_file, score_file, 
                                        dbphore_file, log_file, overwrite=True, return_all=True)
            if len(scores) == 0:
                print(data.name, scores)
            fitscore, ph_overlap, ex_overlap = scores[0]
            data.fitscore = fitscore
            data.ph_overlap = ph_overlap
            data.ex_overlap = ex_overlap

    def sample_from_infer(self, data_0, data, t_n_1, tr_sigma, rot_sigma, tor_sigma, 
                          torsion_updates=None, debug=False):
        result, info = data, {}

        if self.model is not None and self.args is not None and data_0 is not None:
            # Predict tr_score, tor_score, rot_score with model(x_n_1, y, t_n_1)
            # Compute \deltar_t_(n+1)->t_n, \deltaR_t_(n+1)->t_n, \delta\theta_t_(n+1)->t_n
           
            _data_batch = Batch.from_data_list([copy.deepcopy(data)])
            # _data_batch = Batch.from_data_list([copy.deepcopy(data)])
            _data, tor_perturb_n_1_n, tr_perturb_n_1_n, rot_perturb_n_1_n = sample_step(_data_batch, self.model, self.args, 
                                                                                        tr_sigma, rot_sigma, tor_sigma, 
                                                                                        delta_t=self.delta_t)
            _data = _data[0]

            # Compute \deltar_0->t_n, \deltaR_0->t_n, \delta\theta_0->t_n
            tor_up = np.zeros(_data['ligand'].edge_mask.sum().item(), dtype=np.float64)
            if torsion_updates is not None:
                tor_up += torsion_updates
            if tor_perturb_n_1_n is not None:
                tor_up += tor_perturb_n_1_n

            tr_up, rot_up = get_updates_from_0_to_n(data_0, _data, tor_up)

            # Get x_n w.r.t \deltar_0->t_n, \deltaR_0->t_n, \delta\theta_0->t_n
            t_tr, t_rot, t_tor = t_n_1 - self.delta_t, t_n_1 - self.delta_t, t_n_1 - self.delta_t
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
            set_time_phore(data_0, t_tr, t_rot, t_tor, 1, device=None)
            modify_conformer(data_0, tr_up, torch.from_numpy(rot_up).float(), tor_up)

            self.get_fitscore(data_0)
            data_0.tr_score = - tr_up / tr_sigma ** 2
            data_0.rot_score = torch.from_numpy(so3.score_vec(vec=rot_up, eps=rot_sigma)).float().unsqueeze(0)
            data_0.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(tor_up, tor_sigma)).float()
            data_0.tor_sigma_edge = None if self.no_torsion else np.ones(data_0['ligand'].edge_mask.sum()) * tor_sigma            
            result = data_0

            if debug:
                info['tor_perturb_n_1_n'] = tor_perturb_n_1_n
                info['tr_perturb_n_1_n'] = tr_perturb_n_1_n
                info['rot_perturb_n_1_n'] = rot_perturb_n_1_n
                info['tor_perturb_0_n'] = tor_up
                info['tr_perturb_0_n'] = tr_up
                info['rot_perturb_0_n'] = rot_up
                info['rmsd'] = ((data_0['ligand'].pos - _data['ligand'].pos) ** 2).sum(dim=1).mean().sqrt()

        return result, info

    def from_infer(self, t):
        if self.model is not None and t > self.delta_t:
            if self.dynamic_coeff == 0:
                return self.current_epoch >= self.epoch_from_infer and np.random.uniform() < self.rate_from_infer
            else:
                return np.random.uniform() < self.p
        return False

    def update_model(self, state_dict):
        if self.model is not None:
            self.model.load_state_dict(state_dict)

    def dynamic_schedule(self, epoch, max_rate=0.4, u=400, c=10):
        return max_rate * (1 - u / (u + np.exp(c * epoch / u)))

    def time2delta(self, xt, steps=(20, 2000), eps=0.01, debug=False):
        range_step = np.arange(steps[0], steps[1]+1)
        delta_step = xt * range_step % 1
        delta_step_abs = np.abs(delta_step - delta_step.round())
        succ = True
        try:
            step = np.random.choice(range_step[delta_step_abs<eps])
        except Exception as e:
            print(f"[W] Failed to sample delta t from in the range of {list(steps)}, choosing the best one. {e}")
            step = range_step[delta_step_abs.argsort()[0]]
            succ = False

        if debug:
            return 1 / step, succ

        return 1 / step


class PhoreDataset(Dataset):
    """
    A dataset class for handling pharmacophore datasets.
    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        cache_path (str, optional): Path to cache the processed data.
        split_path (str, optional): Path to the split file.
        limit_complexes (int, optional): Limit the number of complexes to load.
        num_workers (int, optional): Number of workers for parallel processing.
        popsize (int, optional): Population size for optimization.
        maxiter (int, optional): Maximum number of iterations for optimization.
        matching (bool, optional): Whether to perform matching.
        keep_original (bool, optional): Whether to keep the original structures.
        max_lig_size (int, optional): Maximum ligand size.
        consider_ex (bool, optional): Whether to consider excluded volumes.
        neighbor_cutoff (int, optional): Neighbor cutoff distance.
        ex_connected (bool, optional): Whether excluded volumes are connected.
        remove_hs (bool, optional): Whether to remove hydrogen atoms.
        num_conformers (int, optional): Number of conformers to generate.
        require_ligand (bool, optional): Whether a ligand is required.
        move_to_center (bool, optional): Whether to move structures to the center.
        protein_ligand_records (list, optional): List of protein-ligand records.
        keep_local_structures (bool, optional): Whether to keep local structures.
        tank_features (bool, optional): Whether to use tank features.
        use_LAS_constrains (bool, optional): Whether to use LAS constraints.
        require_affinity (bool, optional): Whether affinity is required.
        interaction_threshold (int, optional): Interaction threshold distance.
        use_phore_rule (bool, optional): Whether to use pharmacophore rules.
        save_single (bool, optional): Whether to save single files.
        dataset (str, optional): Name of the dataset.
        chembl_path (str, optional): Path to ChEMBL dataset.
        zinc_path (str, optional): Path to ZINC dataset.
        prepared_dataset_path (str, optional): Path to prepared dataset.
        ro5_filter (bool, optional): Whether to apply Rule of Five filter.
        use_sdf (bool, optional): Whether to use SDF format.
        near_phore (bool, optional): Whether to consider near pharmacophores.
        min_phore_num (int, optional): Minimum number of pharmacophores.
        max_phore_num (int, optional): Maximum number of pharmacophores.
        flag (str, optional): Additional flag for dataset processing.
        phore_path (str, optional): Path to pharmacophore files.
        confidence_mode (bool, optional): Whether to use confidence mode.
        fitscore_cutoff (float, optional): Cutoff for fit score.
        **kwargs: Additional arguments.
    """
    def __init__(self, root, transform=None, cache_path='../data/cache', split_path='../data/split/', limit_complexes=0,
                 num_workers=1, popsize=15, maxiter=15, matching=True, keep_original=True, max_lig_size=None, consider_ex = True, 
                 neighbor_cutoff=5, ex_connected=False, remove_hs=False, num_conformers=1, require_ligand=True, move_to_center=True, 
                 protein_ligand_records=None, keep_local_structures=False, tank_features=False, use_LAS_constrains=True, 
                 require_affinity=True, interaction_threshold=5, use_phore_rule=False, save_single=False, dataset='pdbbind',
                 chembl_path='../data/ChEMBL/', zinc_path='../data/ZINC/', prepared_dataset_path=None, ro5_filter=False, 
                 use_sdf=True, near_phore=False, min_phore_num=0, max_phore_num=999, flag="", phore_path=None, confidence_mode=False, 
                 fitscore_cutoff=0.0,
                 **kwargs):
        super(PhoreDataset, self).__init__(root, transform)
        self.root = root
        self.complex_lig_dir = os.path.join(self.root, 'all')
        if phore_path is None:
            self.phore_path = os.path.join(self.root, ('phore_dedup' if flag=='phoreDedup' else 'phore') if use_sdf else 'phore_mol2')
        else:
            self.phore_path = os.path.join(self.root, phore_path) if not os.path.exists(phore_path) else phore_path

        self.ligand_path = os.path.join(self.root, 'mol2')
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.num_workers = num_workers
        self.remove_hs = remove_hs
        self.max_lig_size = max_lig_size
        self.require_ligand = require_ligand
        self.protein_ligand_records = protein_ligand_records

        self.keep_local_structures = keep_local_structures
        self.move_to_center = move_to_center
        self.consider_ex = consider_ex
        self.neighbor_cutoff = neighbor_cutoff
        self.ex_connected = ex_connected
        self.tank_features = tank_features
        self.use_LAS_constrains = use_LAS_constrains
        self.require_affinity = require_affinity
        self.interaction_threshold = interaction_threshold
        self.use_phore_rule = use_phore_rule
        self.chembl_path = chembl_path
        self.zinc_path = zinc_path
        self.ro5_filter = ro5_filter
        self.min_phore_num = min_phore_num
        self.confidence_mode = confidence_mode

        self.prepared_dataset_path = prepared_dataset_path
        self.dataset = dataset
        self.dataset_id = 'ligand_id'

        if self.dataset == 'chembl':
            self.dataset_id = 'chembl_id'
            self.prepared_dataset_path = self.chembl_path if self.prepared_dataset_path is None \
                else self.prepared_dataset_path
        elif self.dataset == 'zinc':
            self.dataset_id = 'zinc_id'
            self.prepared_dataset_path = self.zinc_path if self.prepared_dataset_path is None \
                else self.prepared_dataset_path

        if matching or protein_ligand_records is not None:
            cache_path += '_torsion'
            if protein_ligand_records is not None and len(protein_ligand_records) > 100000:
                save_single = True
                print(f'[I] Too many samples (>100000). Saving samples as single files')

        self.full_cache_path = os.path.join(cache_path, 
                                            (f'{self.dataset.upper()}' if dataset != "pdbbind" else f'PDBBind_{os.path.splitext(os.path.basename(self.split_path))[0].upper()}')
                                            + f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                            + ('' if not consider_ex else "_EX")
                                            + ('' if neighbor_cutoff is None else f'NeibCut{neighbor_cutoff}')
                                            + ('' if not ex_connected else '_EX2Phore')
                                            + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            # + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if not use_phore_rule else f'_phoreRule')
                                            + ('' if not use_sdf or dataset != 'pdbbind' else f'_sdf')
                                            + ('' if not near_phore else f'_nearPhore')
                                            + ('' if protein_ligand_records is None else \
                                               str(binascii.crc32(json.dumps(protein_ligand_records).encode())))\
                                            # + ('' if protein_path_list is None or ligand_descriptions is None else \
                                            #    str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))\
                                            + ('' if flag.strip() == "" else f"_{flag}"))
        print(f"[I] Saving dataset info to `{self.full_cache_path}`")
        
        # self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
        #                                     + (f'_{self.dataset}' if save_single else f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}')
        #                                     + f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
        #                                     + ('' if not consider_ex else "_EX")
        #                                     + ('' if neighbor_cutoff is None else f'NeibCut{neighbor_cutoff}')
        #                                     + ('' if not ex_connected else '_EX2Phore')
        #                                     + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
        #                                     # + ('' if not keep_local_structures else f'_keptLocalStruct')
        #                                     + ('' if not use_phore_rule else f'_phoreRule')
        #                                     + ('' if not use_sdf or dataset != 'pdbbind' else f'_sdf')
        #                                     + ('' if not near_phore else f'_nearPhore')
        #                                     + ('' if protein_ligand_records is None else \
        #                                        str(binascii.crc32(json.dumps(protein_ligand_records).encode())))\
        #                                     # + ('' if protein_path_list is None or ligand_descriptions is None else \
        #                                     #    str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode())))\
        #                                     + ('' if flag.strip() == "" else f"_{flag}"))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.save_single = save_single
        self.graph_pkls_back = None

        confidence_failed_names = ['3skk', '4hze', '4ie2', '4kii', '4q3q', '4q3s', '3sjt', '4q3r', '4hxq']
        exclude_pdb = []
        scores = {}
        if dataset == 'pdbbind':
            anal_path = os.path.join(self.root, "analysis")
            fitscore = os.path.join(anal_path, 'pdbbind_complex_fitscore.tsv')
            if not os.path.exists(fitscore):
                df_pdbbind = pd.read_csv('/home/worker/users/YJL/DiffPhore/data/splits/pdbbind_total', names=['pdb_id'])
                df_pdbbind['score'] = df_pdbbind['pdb_id'].map(lambda x: calc_pdbbind_fitscore(x, anal_path, pdbbind_path=self.root, dedup=flag=="phoreDedup"))
                # df_pdbbind[df_pdbbind['score']!= -2]['score'].plot(kind='hist', bins=20)
                df_pdbbind.to_csv(fitscore, sep='\t', index=False)
            else:
                df_pdbbind = pd.read_csv(fitscore, sep='\t')
            scores = dict(zip(df_pdbbind['pdb_id'], df_pdbbind['score']))
            if fitscore_cutoff > 0:
                exclude_pdb = df_pdbbind[df_pdbbind['score'] < fitscore_cutoff]['pdb_id'].values.tolist()

        # self.propotion = propotion
        if not self.save_single:
            if not os.path.exists(os.path.join(self.full_cache_path, "heterographs.pkl"))\
                    or (require_ligand and not os.path.exists(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"))):
                os.makedirs(self.full_cache_path, exist_ok=True)
                if protein_ligand_records is None:
                    self.preprocessing()
                else:
                    self.inference_preprocessing()

            graphs, rdkit_ligands, affinity = [], [], []
            print('loading data from memory: ', os.path.join(self.full_cache_path, "heterographs.pkl"))
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'rb') as f:
                graphs = pickle.load(f)
            if require_ligand:
                print("loading ligands from cache.")
                with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'rb') as f:
                    rdkit_ligands = pickle.load(f)
            if self.tank_features and self.require_affinity:
                affinity = self.get_affinity()

            self.graphs, self.rdkit_ligands, self.affinity = [], [], []
            for idx in range(len(graphs)):
                skip_flag = False
                if graphs[idx].name in exclude_pdb and not self.confidence_mode:
                    skip_flag = True

                if graphs[idx].name in confidence_failed_names and self.confidence_mode:
                    skip_flag = True
                
                if min_phore_num > 0 and max_phore_num < 999:
                    # print("graphs[idx]['phore'].phoretype[:,:-1].sum().item() =", graphs[idx]['phore'].phoretype[:,:-1].sum().item())
                    ph_num = graphs[idx]['phore'].phoretype[:,:-1].sum().item()
                    if ph_num < min_phore_num or ph_num > max_phore_num:
                        skip_flag = True
                
                if graphs[idx]['ligand'].pos.shape[0] == 0:
                    print(f"[W] Sample `{graphs[idx]['name']}` with 0 atom, skipped.")
                    skip_flag = True
            
                if graphs[idx]['phore'].pos.shape[0] == 0:
                    print(f"[W] Sample `{graphs[idx]['name']}` with 0 phore, skipped.")
                    skip_flag = True

                if skip_flag:
                    continue

                if graphs[idx].name in scores:
                    graphs[idx].complex_fitscore = scores[graphs[idx].name]

                if require_ligand: 
                    graphs[idx].mol = copy.deepcopy(rdkit_ligands[idx])

                self.graphs.append(graphs[idx])
                self.rdkit_ligands.append(rdkit_ligands[idx])
                if len(affinity):
                    self.affinity.append(affinity[idx]) # type: ignore
        else:
            self.graphs = None
            pkllist_file = os.path.join(self.full_cache_path, 'graphlist.pkl')
            self.graph_pkls = []
            if not os.path.exists(pkllist_file):
                if self.protein_ligand_records is None:
                    ligand_processed = os.path.join(self.prepared_dataset_path, 'filtered_phore_df_')
                    self.ligandOnly_preprocessing(ligand_processed)
                else:
                    self.inference_preprocessing()
                pickle.dump(self.graph_pkls, open(pkllist_file, 'wb'))

            else:
                self.graph_pkls = pickle.load(open(pkllist_file, 'rb'))
                print(f"Loading total graph list from `{pkllist_file}`")
                # if os.path.exists(self.split_path) and os.path.isfile(self.split_path):
                #     pass
            self.split()

        if self.ro5_filter:
            self.filter_by_ro5()
        # if self.min_phore_num > 0:
        #     self.filter_by_phore()


    def preprocessing(self):
        """
        Preprocesses the complexes from the specified split path and saves the results to the cache path.

        This method reads complex names from a text file, processes them in parallel if multiple workers are specified,
        and saves the processed data into pickle files. The data includes complex graphs and RDKit ligands.

        If the number of workers is greater than 1, the preprocessing is done in parallel batches of 1000 complexes,
        and the progress is saved periodically. The final combined results are also saved.

        Attributes:
            split_path (str): Path to the text file containing the list of complex names.
            full_cache_path (str): Path to the directory where the processed data will be saved.
            limit_complexes (int, optional): Limit on the number of complexes to process. If None or 0, all complexes are processed.
            num_workers (int): Number of worker processes to use for parallel processing.

        Raises:
            Exception: If there is an error during the preprocessing or saving of data.
        """
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')
        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        if self.num_workers > 1:
            # running preprocessing in parallel on multiple workers and saving the progress every 1000 complexes
            for i in range(len(complex_names_all)//1000+1):
                if os.path.exists(os.path.join(self.full_cache_path, f"heterographs{i}.pkl")):
                    continue
                complex_names = complex_names_all[1000*i:1000*(i+1)]
                complex_graphs, rdkit_ligands = [], []
                if self.num_workers > 1:
                    p = Pool(self.num_workers, maxtasksperchild=1)
                    p.__enter__()
                else:
                    p = None
                with tqdm(total=len(complex_names), desc=f'loading complexes {i}/{len(complex_names_all)//1000+1}') as pbar:
                    map_fn = p.imap_unordered if self.num_workers > 1 else map
                    for t in map_fn(self.get_graph, zip(complex_names, [None] * len(complex_names), [None] * len(complex_names), [None] * len(complex_names))):
                        complex_graphs.extend(t[0])
                        rdkit_ligands.extend(t[1])
                        pbar.update()
                if self.num_workers > 1: p.__exit__(None, None, None)

                with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'wb') as f:
                    pickle.dump((complex_graphs), f)
                with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'wb') as f:
                    pickle.dump((rdkit_ligands), f)

            complex_graphs_all = []
            for i in range(len(complex_names_all)//1000+1):
                with open(os.path.join(self.full_cache_path, f"heterographs{i}.pkl"), 'rb') as f:
                    l = pickle.load(f)
                    complex_graphs_all.extend(l)
            with open(os.path.join(self.full_cache_path, f"heterographs.pkl"), 'wb') as f:
                pickle.dump((complex_graphs_all), f)

            rdkit_ligands_all = []
            for i in range(len(complex_names_all) // 1000 + 1):
                with open(os.path.join(self.full_cache_path, f"rdkit_ligands{i}.pkl"), 'rb') as f:
                    l = pickle.load(f)
                    rdkit_ligands_all.extend(l)
            with open(os.path.join(self.full_cache_path, f"rdkit_ligands.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands_all), f)
        else:
            complex_graphs, rdkit_ligands = [], []
            # with tqdm(total=len(complex_names_all), desc='loading complexes') as pbar:
            for t in map(self.get_graph, zip(complex_names_all, [None]*len(complex_names_all), [None] * len(complex_names_all), [None] * len(complex_names_all))):
                complex_graphs.extend(t[0])
                rdkit_ligands.extend(t[1])
                    # pbar.update()
            with open(os.path.join(self.full_cache_path, "heterographs.pkl"), 'wb') as f:
                pickle.dump((complex_graphs), f)
            with open(os.path.join(self.full_cache_path, "rdkit_ligands.pkl"), 'wb') as f:
                pickle.dump((rdkit_ligands), f)


    def inference_preprocessing(self):
        """
        Preprocesses protein-ligand records for inference by generating graphs from descriptions.
        This method processes the protein-ligand records in chunks, generates graphs for each record,
        and saves the processed data to disk. If `save_single` is True, the graphs are saved individually;
        otherwise, they are saved as a single file.
        Attributes:
            protein_ligand_records (list): List of protein-ligand records to be processed.
            num_workers (int): Number of workers to use for parallel processing.
            save_single (bool): Flag indicating whether to save each graph individually.
            full_cache_path (str): Path to the cache directory.
        Methods:
            generate_graph_from_description: Function to generate a graph from a ligand description and phore.
        Raises:
            OSError: If there is an issue creating directories or saving files.
        Prints:
            Information about the number of batches and total samples processed.
        """
        records = self.protein_ligand_records
        if self.num_workers > 1:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.num_workers)
        
        gen_func = self.generate_graph_from_description
        graphs = []
        ligands = []
        dump_path = ""
        if self.save_single:
            dump_path = os.path.join(self.full_cache_path, 'records')
            gen_func = partial(self.generate_graph_from_description, dump_path=dump_path)
            os.makedirs(dump_path, exist_ok=True)

        processed_path = os.path.join(self.full_cache_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)

        chunk_size = 10000
        chunks = len(records) // chunk_size
        if len(records) % chunk_size != 0:
            chunks += 1
        print(f"[I] {chunks} batchs of data to process with {chunk_size} samples each.")

        for chunk_id in range(chunks):
            processed_file = os.path.join(processed_path, f'chunk_{chunk_id}.pkl')
            if not os.path.exists(processed_file):
                print(f"[I] Processing {chunk_id+1}/{chunks} samples")
                df_chunk = pd.DataFrame(records[chunk_id*chunk_size:(chunk_id+1)*chunk_size])
                if self.num_workers > 1:
                    df_chunk['graphs'] = df_chunk.parallel_apply(lambda x: gen_func(x['ligand_description'], x['phore']), axis=1)
                else:
                    df_chunk['graphs'] = df_chunk.apply(lambda x: gen_func(x['ligand_description'], x['phore']), axis=1)
                pickle.dump(df_chunk, open(processed_file, 'wb'))
            else:
                df_chunk = pickle.load(open(processed_file, 'rb'))
                
            for _graphs in df_chunk['graphs'].values.tolist():
                graphs.extend(_graphs[0])
                ligands.extend(_graphs[1])

        print(f"[I] Total samples: {len(graphs)}")
        if self.save_single:
            self.graph_pkls = copy.deepcopy(graphs)
        else:
            pickle.dump(graphs, open(os.path.join(self.full_cache_path, 'heterographs.pkl'), 'wb'))
            pickle.dump(ligands, open(os.path.join(self.full_cache_path, 'rdkit_ligands.pkl'), 'wb'))


    def generate_graph_from_description(self, ligand_description, phore, dump_path=""):
        """
        Generates molecule-pharmacophore graphs from a ligand description and a pharmacophore file.
        Args:
            ligand_description (str): The description of the ligand, which can be a SMILES string or a file path.
            phore (str): The path to the pharmacophore file.
            dump_path (str, optional): The path to dump the generated graph files. Defaults to "".
        Returns:
            tuple: A tuple containing two lists:
                - graphs (list): A list of generated graph objects or file paths to the dumped graph files.
                - ligands (list): A list of RDKit molecule objects corresponding to the ligands.
        Raises:
            Exception: If RDKit fails to read the molecule from the provided ligand description.
        Notes:
            - If the ligand description is a SMILES string, a conformer will be generated for the molecule.
            - If the ligand description is a file path, the molecule will be read from the file.
            - If the pharmacophore file contains multiple pharmacophores, a graph will be generated for each pharmacophore.
            - If `dump_path` is provided and exists, the generated graphs will be dumped to files in the specified path.
            - If no valid graphs are generated, a warning message will be printed.
        """
        graphs, ligands = [], []
        try: 
            # Read Molecule
            # print(f'ligand_description: {ligand_description}')
            if not os.path.exists(ligand_description):
                mol_name = ""
                if ' ' in ligand_description:
                    parts = ligand_description.split()
                    ligand_description, mol_name = parts[0], parts[-1]

                mol = Chem.MolFromSmiles(ligand_description)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    generate_conformer(mol)
                    mol_name = ligand_description if mol_name.strip() == "" else mol_name
            else:
                # print(f"The mol file existence: {os.path.exists(ligand_description)}. {ligand_description}")
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    if ligand_description.endswith('.sdf'):
                        ligand_description_1 = ligand_description.replace('.sdf', '.mol2')
                        if os.path.exists(ligand_description_1):
                            print(f"[W] Failed to read the sdf file `{ligand_description}`, try to read the mol2 file.")
                            ligand_description = ligand_description_1
                            mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                    if mol is None:
                        raise Exception('RDKit could not read the molecule ', ligand_description)
                if not self.keep_local_structures:
                    mol.RemoveAllConformers()
                    mol = Chem.AddHs(mol)
                    generate_conformer(mol)
                mol_name = '.'.join(os.path.basename(ligand_description).split(".")[:-1])

            # Read Pharmacophore
            phores = parse_phore(phore_file=phore)
            for _phore in phores:
                name = f"{_phore.id}__{mol_name}"
                graph = self.generate_graph(name, _phore, mol)
                graph.original_id = _phore.id
                graph.mol = mol
                graph.phore_file = os.path.abspath(phore)
                if dump_path != "" and os.path.exists(dump_path):
                    graph_file = os.path.join(dump_path, f"{name}_graph.pkl")
                    pickle.dump(graph, open(graph_file, "wb"))
                    graphs.append(graph_file)
                else:
                    graphs.append(graph)
                    ligands.append(mol)

        except Exception as e:
            print(f'Failed to generate graph. We are skipping it. The reason is the exception: {e}')
        
        if len(graphs) == 0:
            print(f"[W] No valid graph is generated from the ligand description: `{ligand_description}` and the pharmacophore file: `{phore}`")

        return graphs, ligands


    def ligandOnly_preprocessing(self, ligand_dump_path, overwrite=False):
        """
        Preprocess ligand data by loading, processing, and caching ligand information.
        Args:
            ligand_dump_path (str): Path to the directory containing ligand dump files.
            overwrite (bool): Whether to overwrite existing processed files. Default is False.
        Raises:
            RuntimeError: If no docking samples are found after preprocessing.
        This method performs the following steps:
        1. Creates necessary directories for records and processed data if they do not exist.
        2. Loads ligand dump files from the specified path.
        3. Processes each ligand dump file:
            - Loads the ligand data from the file.
            - Applies the `get_graph_from_df` function to each ligand sample to generate graph files.
            - Caches the processed ligand data.
        4. Extends the `graph_pkls` attribute with the generated graph files.
        5. Raises an error if no docking samples are found after preprocessing.
        """
        record_path = os.path.join(self.full_cache_path, 'records')
        processed_path = os.path.join(self.full_cache_path, 'processed')
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        ligand_dump_files = [os.path.join(ligand_dump_path, filename) for filename in os.listdir(ligand_dump_path)]
        self.graph_pkls = []
        
        if self.num_workers > 1:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.num_workers)
        for idx, ligand_dump_file in enumerate(ligand_dump_files):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ({idx+1}/{len(ligand_dump_files)}) Start to load `{ligand_dump_file}`")
            processed_file = os.path.join(processed_path, os.path.basename(ligand_dump_file))
            if not os.path.exists(processed_file) or overwrite:
                df_ligand = pickle.load(open(ligand_dump_file, 'rb'))
                print(f"`{ligand_dump_file}` loaded, {len(df_ligand)} samples to process...")
                func = partial(self.get_graph_from_df, record_path=record_path, overwrite=False)
                if self.num_workers > 1:
                    df_ligand['graph_files'] = df_ligand.parallel_apply(lambda x: func(x['mol'], x['random_phores'], x[self.dataset_id]), axis=1)
                else:
                    df_ligand['graph_files'] = df_ligand.apply(lambda x: func(x['mol'], x['random_phores'], x[self.dataset_id]), axis=1)
                pickle.dump(df_ligand, open(processed_file, 'wb'))
            else:
                print("Processed before, loading cache...")
                df_ligand = pickle.load(open(processed_file, 'rb'))
            
            for graph_files in df_ligand['graph_files'].values.tolist():
                self.graph_pkls.extend(graph_files)
        if len(self.graph_pkls) == 0 and len(self.graphs) == 0:
            raise RuntimeError("No docking sample after preprocess, check your input please.")


    def filter_by_phore(self):
        """
        Filters the dataset by the number of pharmacophores (phores) in each graph.

        This method filters the graphs and ligands in the dataset based on the number of pharmacophores
        present in each graph. If the dataset is 'pdbbind', it iterates through the graphs and counts
        the number of pharmacophores according to the specified rule. If the count of pharmacophores
        exceeds the minimum threshold (`min_phore_num`), the graph and corresponding ligand are added
        to the filtered lists.

        Attributes:
            dataset (str): The name of the dataset, should be 'pdbbind' for this method to execute.
            graphs (list): A list of graph objects, each containing pharmacophore information.
            rdkit_ligands (list): A list of RDKit ligand objects corresponding to the graphs.
            use_phore_rule (bool): A flag indicating which rule to use for counting pharmacophores.
            min_phore_num (int): The minimum number of pharmacophores required for a graph to be included
                                 in the filtered dataset.

        Modifies:
            self.graphs (list): Updates with the filtered list of graphs.
            self.rdkit_ligands (list): Updates with the filtered list of RDKit ligands.
        """
        if self.dataset == 'pdbbind':
            graphs, ligands = [], []
            for i in range(len(self.graphs)):
                num_ph = sum(1 - self.graphs[i]['phore'].phoretype[:, -1]) if self.use_phore_rule else sum(self.graphs[i]['phore'].x[:, 0] != 10)
                if int(num_ph) > self.min_phore_num:
                    graphs.append(copy.deepcopy(self.graphs[i]))
                    ligands.append(copy.deepcopy(self.rdkit_ligands[i]))
            self.graphs = graphs
            self.rdkit_ligands = ligands


    def filter_by_ro5(self):
        """
        Filters the dataset by the Rule of Five (Ro5).
        """
        if self.dataset == 'pdbbind':
            print("[I] The dataset is to be filtered by Rule of Five.")
            sample_num = len(self.graphs)
            df_graphs = pd.DataFrame({'graphs': self.graphs, 'ligands': self.rdkit_ligands})
            df_graphs['violations'] = df_graphs['ligands'].map(lambda x: lipinski_rule_analysis(x))
            df_graphs = df_graphs[df_graphs['violations'] < 1]
            self.graphs = df_graphs['graphs'].tolist()
            self.rdkit_ligands = df_graphs['ligands'].tolist()
            print(f'[I] Filtered successfully, samples changes ({sample_num} -> {len(self.graphs)})')


    def split(self):
        """
        Splits the dataset based on a provided split file.

        This method attempts to split the dataset by loading indices from a specified split file.
        If the split file is valid and contains indices, the dataset is filtered to include only
        the specified indices. If the split file is invalid or no overlap is found, the original
        dataset is retained.
        """
        self.origin_graph_pkls = copy.deepcopy(self.graph_pkls)
        if isinstance(self.split_path, str) and os.path.exists(self.split_path) and os.path.isfile(self.split_path):
            split_index = []
            print(f'Loading current index from cache `{self.split_path}`.')
            if self.split_path.endswith(".pkl"):
                split_index = pickle.load(open(self.split_path, 'rb'))
            else:
                split_index = read_strings_from_txt(self.split_path)
            if len(split_index):
                df_curr = pd.DataFrame(self.graph_pkls, columns=['filename'])
                df_curr['index'] = df_curr['filename'].map(lambda x: os.path.basename(x).replace("_graph.pkl", ""))
                df_split_index = pd.DataFrame(split_index, columns=['index'])
                df_curr = pd.merge(df_curr, df_split_index, how='inner', on='index')
                self.graph_pkls = df_curr['filename'].tolist()
                if len(self.graph_pkls) == 0:
                    print(f"[W] No overlap between split file and current index. Keep original full dataset.")
                    self.graph_pkls = copy.deepcopy(self.origin_graph_pkls)
        else:
            print(f"[W] Invalid split list file specified. Keep original full dataset. `{self.split_path}`")


    def random_select(self, propotion=None, number=None, load_graphs=False, debug=False):
        """
        Randomly selects a subset of graphs from the dataset.

        Args:
        propotion (float, optional): The proportion of the dataset to select. If specified, 
                                     it should be a value between 0 and 1. Defaults to None.
        number (int, optional): The number of graphs to select. If specified, it overrides 
                                the propotion parameter. Defaults to None.
        load_graphs (bool, optional): If True, loads the selected graphs into memory. Defaults to False.
        debug (bool, optional): If True, prints debug information. Defaults to False.

        Returns:
        None
        """
        if self.graph_pkls_back is None:
            self.graph_pkls_back = copy.deepcopy(self.graph_pkls)
        if number is None or number == 0:
            number = len(self.graph_pkls_back)
            if propotion is not None and propotion < 1:
                number = int(number * propotion)
        number = min(len(self.graph_pkls_back), number)
        if debug:
            print(f"[I] Selecting `{number}` random samples.")
        self.graph_pkls = random.sample(self.graph_pkls_back, k=number)

        if load_graphs and not (self.graphs is not None and number == len(self.graph_pkls_back) and number == len(self.graphs)):
            self.graphs = None
            graphs = [self.get(idx) for idx in range(number)]
            self.graphs = copy.deepcopy(graphs)
            # print("graph loaded ...")


    def len(self):
        return len(self.graphs) if not self.save_single else len(self.graph_pkls)


    def get(self, idx):
        if self.save_single and self.graphs is None:
            graph = pickle.load(open(self.graph_pkls[idx], 'rb'))
            if not hasattr(graph, 'phore_file') and self.dataset == 'zinc':
                graph.phore_file = os.path.abspath(os.path.join(self.zinc_path, f"sample_phores/{graph.name}.phore"))

        else:
            graph = copy.deepcopy(self.graphs[idx])
            if self.dataset == 'pdbbind' and not hasattr(graph, 'phore_file'):
                graph.phore_file = os.path.abspath(os.path.join(self.phore_path, f"{graph.name}/{graph.name}_complex.phore"))
            if self.require_ligand:
                if not hasattr(graph, 'mol'):
                    graph.mol = copy.deepcopy(self.rdkit_ligands[idx])
            if self.tank_features:
                self.get_tank_features(graph)

        # if not hasattr(graph['ligand'], 'orig_pos'):
        #     # graph['ligand'].orig_pos = copy.deepcopy(graph['ligand'].pos)
        #     graph['ligand'].orig_pos = graph['ligand'].pos.numpy() + graph.original_center.numpy()
        
        if self.use_phore_rule:
            if not hasattr(graph['ligand'], 'phorefp'):
            # if not hasattr(graph['ligand'], 'phorefp') or not hasattr(graph['ligand'], 'ph'):
                generate_ligand_phore_feat(graph.mol, graph, remove_hs=self.remove_hs)
            if not hasattr(graph['phore'], 'phoretype'):
                graph['phore'].phoretype = torch.tensor([[int(x) == y for y in range(11)] for x in graph['phore'].x[:, 0]]).float()

        return graph


    def get_tank_features(self, graph):
        graph['compound_pair'].x = extract_pair_distribution(graph.mol, self.use_LAS_constrains, 
                                                                     remove_hs=self.remove_hs, sanitize=True, 
                                                                     coords=graph['ligand'].pos.numpy())
        affinity = self.affinity[graph.name] if graph.name in self.affinity else 0
        graph.affinity = torch.tensor([affinity], dtype=torch.float)
        lig_pos = graph['ligand'].orig_pos if not isinstance(graph['ligand'].orig_pos, list) else graph['ligand'].orig_pos[0]
        lig_pos = torch.tensor(lig_pos, dtype=torch.float())
        dis_map = torch.cdist(graph['phore'].pos, lig_pos, compute_mode='donot_use_mm_for_euclid_dist')
        graph.y = torch.tensor(dis_map < self.interaction_threshold, dtype=torch.float).flatten()
        dis_map[dis_map > self.interaction_threshold] = self.interaction_threshold
        graph.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()


    def get_graph_from_df(self, ligand, phores, original_id, record_path, overwrite=True):
        """
        Generates graph representations from a list of phores and a ligand, and saves them to disk.
        Args:
            ligand (object): The ligand object to be used in graph generation.
            phores (list): A list of phore objects or file paths to phore files.
            original_id (str): The original identifier for the graph.
            record_path (str): The directory path where the graph files will be saved.
            overwrite (bool, optional): If True, existing graph files will be overwritten. Defaults to True.
        Returns:
            list: A list of file paths to the generated graph files.
        Raises:
            Exception: If there is an error in generating or saving a graph, an error message is printed.
        """
        graph_files = []
        for phore in phores:
            try:
                phore_file = ""
                if isinstance(phore, str) and os.path.exists(phore):
                    phore_file = phore
                    phore = parse_phore(phore)[0]
                
                graph_file = os.path.join(record_path, f"{phore.id}_graph.pkl")
                # graph_file = os.path.abspath(os.path.join(record_path, f"{phore.id}_graph.pkl"))
                flag = not os.path.exists(graph_file) or overwrite
                if not flag:
                    try:
                        pickle.load(open(graph_file, 'rb'))
                    except Exception as e:
                        flag = True 
                if flag:
                    graph = self.generate_graph(phore.id, phore, ligand)
                    graph.original_id = original_id
                    graph.mol = ligand
                    if phore_file != "":
                        graph.phore_file = phore_file
                    pickle.dump(graph, open(graph_file, 'wb'))
                    print(f"[I] {phore.id} is generated ...")
                    
                if os.path.exists(graph_file):
                    graph_files.append(graph_file)
            except Exception as e:
                if isinstance(phore, str):
                    print(f"[E] Failed to generate graph for `{phore}`. {e}")
                else:
                    print(f"[E] {phore.id} failed. {e}")
        return graph_files


    def get_graph(self, par):
        """
        Generates graphs and ligands from the given parameters.

        Args:
            par (tuple): A tuple containing the following elements:
                - name (str): The name of the complex.
                - phore (str or None): The pharmacophore file or None.
                - ligand (object or None): The ligand object or None.
                - ligand_description (str): Description of the ligand.

        Returns:
            tuple: A tuple containing two lists:
                - graphs (list): A list of generated graphs.
                - ligands (list): A list of corresponding ligands.

        Raises:
            FileNotFoundError: If the pharmacophore file is not found.
            Exception: If there is an error in generating the graph.
        """
        graphs, ligands = [], []
        name, phore, ligand, ligand_description = par
        try:
            if not os.path.exists(os.path.join(self.complex_lig_dir, name)) and ligand is None:
                print("Folder not found", name)
                return [], []

            phores = []
            try:
                phores = parse_phore(name=name, data_path=self.phore_path) if phore is None else parse_phore(phore_file=phore)
            except FileNotFoundError as e:
                print(e)
                return [], []
            if ligand is not None:
                name = f'{name}____{ligand_description}'
                ligs = [ligand]
            else:
                ligs = read_mols(self.complex_lig_dir, name, remove_hs=False)
            failed_indices = []
            for phore in phores:
                for i, lig in enumerate(ligs):
                    if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                        print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                        continue
                    # try:
                    if not self.consider_ex and len(phore.features) <= 1:
                        print(f"{name} skipped. Less than 2 pharmacophores detected.")
                        continue
                    try:
                        graph = self.generate_graph(name, phore, lig)
                    except Exception as e:
                        print(f"[W] Failed to generate the graph `{name}`, {e}")
                        continue
                    graphs.append(graph)
                    ligands.append(lig)

            # for idx_to_delete in sorted(failed_indices, reverse=True):
            #     del ligs[idx_to_delete]
        except Exception as e:
            print(f"[W] {e}")
        return graphs, ligands


    def generate_graph(self, name, phore, lig):
        """
        Generates a graph representation of a ligand and pharmacophore.

        Args:
            name (str): The name of the graph.
            phore (object): The pharmacophore object.
            lig (object): The ligand object.

        Returns:
            HeteroData: A heterogeneous graph containing the ligand and pharmacophore data.

        The function performs the following steps:
            1. Initializes a HeteroData graph.
            2. Adds ligand graph with matching information.
            3. Adds pharmacophore graph with specified parameters.
            4. If `use_phore_rule` is True, generates ligand pharmacophore features and updates the graph.
            5. Computes the center of the pharmacophore and adjusts positions if `move_to_center` is True.
        """
        graph = HeteroData()
        graph['name'] = name
                    # try:
        get_lig_graph_with_matching(lig, graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                    self.num_conformers, remove_hs=self.remove_hs)
                    # get_lig_graph_with_matching(lig, graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                    #                             self.num_conformers, remove_hs=self.remove_hs, use_LAS_constrains=self.use_LAS_constrains)
        get_phore_graph(phore, graph, consider_ex=self.consider_ex, 
                        neighbor_cutoff=self.neighbor_cutoff,  ex_connected=self.ex_connected)
        if self.use_phore_rule:
            generate_ligand_phore_feat(lig, graph, remove_hs=self.remove_hs)
            graph['phore'].phoretype = torch.tensor([[int(x) == y for y in range(11)] for x in graph['phore'].x[:, 0]]).float()
                    # except Exception as e:
                    #     print(f'Skipping {name} because of the error:')
                    #     print(e)
                    #     failed_indices.append(i)
                    #     continue
        phore_center = torch.mean(graph['phore'].pos, dim=0, keepdim=True)
        graph.original_center = phore_center
        if self.move_to_center:
            graph['phore'].pos -= phore_center
            if (not self.matching) or self.num_conformers == 1:
                graph['ligand'].pos -= phore_center
            else:
                for p in graph['ligand'].pos:
                    p -= phore_center
        return graph


    def get_affinity(self):
        index_file = os.path.join(self.complex_lig_dir, "index/INDEX_general_PL_data.2020")
        assert os.path.exists(index_file)
        with open(index_file) as f:
            records = {line.split()[0]: float(line.split()[3]) for line in f.readlines() if line[0] in '0123456789'}
            return records


def print_statistics(complex_graphs):
    statistics = ([], [], [], [])

    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)

    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(4):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def construct_loader(args, t_to_sigma, model=None, **kwargs):
    """
    Constructs and returns training and validation data loaders for the Phore dataset.
    Args:
        args (Namespace): A namespace object containing various arguments and configurations.
        t_to_sigma (callable): A function or callable object that maps time to sigma values.
        model (optional): A model object to be used in the NoiseTransformPhore. Default is None.
        **kwargs: Additional keyword arguments.
    Returns:
        tuple: A tuple containing the training data loader and validation data loader.
    """
    transform = NoiseTransformPhore(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                                    epochs=args.n_epochs, reject=getattr(args, 'reject', False), 
                                    cofactor=getattr(args, 'reject_rate', 0.3),
                                    # contrastive=args.contrastive if hasattr(args, 'contrastive') else False,
                                    # contrastive_model_dir=args.contrastive_model_dir if hasattr(args, 'contrastive_model_dir') else "",
                                    # return_node=args.return_node if hasattr(args, 'return_node') else False,
                                    calc_fitscore=args.confidence_mode, 
                                    fitscore_tmp=os.path.join(args.run_dir, 'fitscore_tmp'),
                                    model=model, args=args, delta_t=getattr(args, 'delta_t', 0.05),
                                    rate_from_infer=getattr(args, 'rate_from_infer', 0),
                                    epoch_from_infer=getattr(args, 'epoch_from_infer', 300),
                                    dynamic_coeff=getattr(args, 'dynamic_coeff', 0)
                                    )
    # args.contrastive_ns = transform.contrastive_ns if hasattr(transform, 'contrastive_ns') else 0
    matching = False
    if args.dataset in ['chembl', 'zinc']:
        matching = False
    elif args.dataset == 'pdbbind' and not args.no_torsion:
        matching = True
    args.matching = matching
    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes, 'save_single': args.save_single,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size, 'matching': args.matching,
                   'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter, 'consider_ex': args.consider_ex,
                   'neighbor_cutoff': args.neighbor_cutoff, 'ex_connected': args.ex_connected, 'num_workers': args.num_workers,
                   'use_LAS_constrains':args.use_las_constrains, 'tank_features': args.model_type == 'tank', 'dataset': args.dataset,
                   'use_phore_rule': args.use_phore_rule, 'ro5_filter': args.ro5_filter, 'use_sdf': args.use_sdf,
                   'near_phore': args.near_phore, 'zinc_path': args.zinc_path, 'chembl_path': args.chembl_path, 
                   'min_phore_num': getattr(args, 'min_phore_num', 0), 'max_phore_num': getattr(args, 'max_phore_num', 999), 
                   'fitscore_cutoff': getattr(args, 'fitscore_cutoff', 0.0),
                   'flag': args.flag, 'phore_path': args.phore_path, 
                   'confidence_mode': args.confidence_mode
                   }

    train_dataset = PhoreDataset(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                                 num_conformers=args.num_conformers, **common_args)
    val_dataset = PhoreDataset(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)

    if args.debug:
        train_dataset = _Subset(train_dataset, list(range(50)))
        val_dataset = _Subset(val_dataset, list(range(30)))
    
    print("Number of train samples:", len(train_dataset))
    print("Number of validate samples:", len(val_dataset))

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_dataloader_workers, shuffle=True, 
                                pin_memory=args.pin_memory, prefetch_factor=2)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, 
                              num_workers=args.num_dataloader_workers, shuffle=True, 
                              pin_memory=args.pin_memory, prefetch_factor=2)
    return train_loader, val_loader


def construct_contrastive_loader(args):
    """
    Constructs and returns the training and validation data loaders for a contrastive learning task.
    Args:
        args (argparse.Namespace): The arguments required for constructing the data loaders. 
    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    matching = True
    if args.dataset in ['chembl', 'zinc']:
        matching = False
    elif args.dataset == 'pdbbind' and not args.no_torsion:
        matching = True
    args.matching = matching
    common_args = {
        'transform': None, 'root': args.data_dir, 'limit_complexes': args.limit_complexes, 'save_single': args.save_single,
        'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size, 'matching': args.matching,
        'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter, 'consider_ex': args.consider_ex,
        'neighbor_cutoff': args.neighbor_cutoff, 'ex_connected': args.ex_connected, 'num_workers': args.num_workers,
        'use_LAS_constrains':args.use_las_constrains, 'tank_features': args.model_type == 'tank', 'dataset': args.dataset,
        'use_phore_rule': args.use_phore_rule, 'ro5_filter': args.ro5_filter, 'use_sdf': args.use_sdf,
        'near_phore': args.near_phore, 'zinc_path': args.zinc_path, 'chembl_path': args.chembl_path, 
        'min_phore_num': args.min_phore_num, 'flag': args.flag, 'phore_path': args.phore_path
    }

    train_dataset = PhoreDataset(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                                 num_conformers=args.num_conformers, **common_args)
    val_dataset = PhoreDataset(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)

    if args.debug:
        train_dataset = _Subset(train_dataset, list(range(50)))
        val_dataset = _Subset(val_dataset, list(range(30)))
    
    print("Number of train samples:", len(train_dataset))
    print("Number of validate samples:", len(val_dataset))

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, 
                                num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    
    cache_label(train_loader)
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Train dataloader label cached.")
    cache_label(val_loader)
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Validate dataloader label cached.")
    
    return train_loader, val_loader


class _Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices
        self.graphs = [self.dataset.graphs[idx] for idx in self.indices]
        if hasattr(self.dataset, 'transform'):
            self.transform = self.dataset.transform
        

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def read_mol(complex_lig_dir, name, remove_hs=False):
    """
    Reads a molecule from a specified directory and file name. Attempts to read 
    from an SDF file first, and if that fails, tries to read from a MOL2 file.

    Args:
        complex_lig_dir (str): The directory containing the ligand files.
        name (str): The base name of the ligand files (without extension).
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecule. Defaults to False.

    Returns:
        Molecule: The molecule object read from the file, or None if reading fails.
    """
    lig = read_molecule(os.path.join(complex_lig_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
        lig = read_molecule(os.path.join(complex_lig_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(complex_lig_dir, name, remove_hs=False):
    """
    Reads molecular files from a specified directory and returns a list of molecules.

    This function searches for files with the ".sdf" extension in the specified directory.
    If a file cannot be sanitized, it attempts to read a corresponding ".mol2" file instead.

    Args:
        complex_lig_dir (str): The directory containing the molecular files.
        name (str): The subdirectory name within `complex_lig_dir` to search for molecular files.
        remove_hs (bool, optional): Whether to remove hydrogen atoms from the molecules. Defaults to False.

    Returns:
        list: A list of molecules read from the files.
    """
    ligs = []
    for file in os.listdir(os.path.join(complex_lig_dir, name)):
        if file.endswith(".sdf") and 'rdkit' not in file:
            lig = read_molecule(os.path.join(complex_lig_dir, name, file), remove_hs=remove_hs, sanitize=True)
            if lig is None and os.path.exists(os.path.join(complex_lig_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
                print('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
                lig = read_molecule(os.path.join(complex_lig_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs


def score_func(a_id, b_name, cutoff=0.5, strict=False, 
               source_path="/home/worker/users/YJL/DiffPhore/data/ZINC/ZINC20_LogPlt5_InStock/noStack", 
               ancphore_path="/home/worker/users/YJL/DiffPhore/programs/AncPhore"):

    """
    Calculate the score for fitting a ligand to a pharmacophore.

    Args:
    a_id (str): Identifier for the ligand.
    b_name (str): Name of the pharmacophore.
    cutoff (float, optional): Threshold for determining a successful fit. Default is 0.5.
    strict (bool, optional): If True, raises an exception on failure. If False, returns a score of 1.0 if a_id matches the prefix of b_name. Default is False.
    source_path (str, optional): Path to the source directory containing ligand and pharmacophore files. Default is "/home/worker/users/YJL/DiffPhore/data/ZINC/ZINC20_LogPlt5_InStock/noStack".
    ancphore_path (str, optional): Path to the AncPhore program. Default is "/home/worker/users/YJL/DiffPhore/programs/AncPhore".

    Returns:
    float: The calculated score for the ligand-pharmacophore fit. Returns 0 if no score is obtained and strict is False.
    """
    lig_file_path = os.path.join(source_path, 'ligand_files')
    sample_phore_path = os.path.join(source_path, 'sample_phores')
    score = 0
    scores = []
    try:
        a_lig_file  = os.path.join(lig_file_path,  f'{a_id}.mol')
        b_phore_file = os.path.join(sample_phore_path, f'{b_name}.phore')
        tmp_path = os.path.join(source_path, 'fitScore_tmp')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path, exist_ok=True)
        score_file = os.path.join(tmp_path,  f"{a_id}__{b_name}.score")
        dbphore_file = os.path.join(tmp_path, f"{a_id}__{b_name}.dbphore")
        log_file = os.path.join(tmp_path, f"{a_id}__{b_name}.log")
        if os.path.exists(score_file):
            scores = parse_score_file(score_file)

        if len(scores) == 0:
            scores = calc_phore_fitting(a_lig_file, b_phore_file, score_file, dbphore_file, log_file, 
                                        ancphore_path=ancphore_path, overwrite=True)
        score = scores[0]
        # score = int(scores[0] > cutoff)
    except Exception as e:
        print(f"No score obtained for {a_id} and {b_name}, scores:{scores}, {e}")
        if not strict and a_id == b_name.split("_")[0]:
            score = 1.0
        # raise e
    return score


def cache_label(dataloader, data=None, batch=None, nworkers=60, overwrite=False, debug=False):
    """
    Cache labels for a given dataloader.
    Args:
    dataloader (DataLoader): The dataloader containing the dataset.
    data (optional): The data to be processed. Default is None.
    batch (int, optional): The batch number to process. Default is None.
    nworkers (int, optional): Number of workers to use for parallel processing. Default is 60.
    overwrite (bool, optional): Whether to overwrite existing cache files. Default is False.
    debug (bool, optional): Whether to print debug information. Default is False.
    Returns:
    scores (optional): The calculated scores if data and batch are provided. Otherwise, returns None.
    """
    batch_size = dataloader.batch_size
    full_cache_path = dataloader.dataset.full_cache_path
    split_name = os.path.basename(dataloader.dataset.split_path).split(".")[0]
    out_path = os.path.join(full_cache_path, f'{split_name}_{batch_size}')
    dataloader.label_path = os.path.abspath(out_path)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    
    if nworkers > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=nworkers)
    
    if data is not None and isinstance(batch, int):
        dump_file = os.path.join(out_path, f'label_{batch}.pt')
        scores = calc_label(data, dump_file, overwrite=overwrite, nworkers=nworkers)
        return scores
    else:
        for i, data in enumerate(dataloader):
            dump_file = os.path.join(out_path, f'label_{i}.pt')
            calc_label(data, dump_file, overwrite=overwrite, nworkers=nworkers)
            if debug:
                print(f"Batch {i} calculated and dumped.")
        return None


def calc_label(data, dump_file, overwrite=False, nworkers=60):
    """
    Calculate and save or load label scores for a given dataset.
    This function calculates label scores for pairs of data entries and saves the results to a file.
    If the file already exists and `overwrite` is False, it loads the scores from the file instead.
    Args:
        data (list): A list of data entries, where each entry has a 'name' attribute.
        dump_file (str): The file path to save or load the scores.
        overwrite (bool, optional): Whether to overwrite the existing file. Defaults to False.
        nworkers (int, optional): The number of workers to use for parallel processing. Defaults to 60.
    Returns:
        torch.Tensor: A tensor containing the calculated scores, with shape (num, num), where num is the length of the data list.
    """
    if not os.path.exists(dump_file) or overwrite:
        num = len(data)
        total_list = []
        for x in range(num):
            for y in range(num):
                total_list.append({'id': data[x].name.split("_")[0], 'name': data[y].name})
        df = pd.DataFrame(total_list)
        _df = pd.DataFrame(total_list).drop_duplicates()
        if nworkers > 1:
            _df['score'] = _df.parallel_apply(lambda x: score_func(x['id'], x['name']), axis=1)
        
        else:
            _df['score'] = _df.apply(lambda x: score_func(x['id'], x['name']), axis=1)
        df = pd.merge(df, _df, on=['id', 'name'], how='left')
        # df.to_csv(os.path.join(out_path, f'label_{i}.csv'), sep='\t', index=False)
        scores = torch.tensor(df['score'].values).float().reshape(num, num)
        torch.save(scores, dump_file)
    else:
        scores = torch.load(dump_file)
    return scores


def calc_pdbbind_fitscore(pdbid, out_path, 
                          pdbbind_path="/home/worker/users/YJL/DiffPhore/data/PDBBind", dedup=False):
    """
    Calculate the fitting score for a given PDBBind entry.

    Args:
        pdbid (str): The PDB ID of the entry.
        out_path (str): The output directory path where results will be saved.
        pdbbind_path (str, optional): The base path to the PDBBind dataset. Defaults to "/home/worker/users/YJL/DiffPhore/data/PDBBind".
        dedup (bool, optional): Whether to use deduplicated phore files. Defaults to False.

    Returns:
        float: The fitting score. If the score calculation fails, returns -2.
    """
    ligand_file = os.path.join(pdbbind_path, f"sdf/{pdbid}_ligand.sdf")
    phore_file = os.path.join(pdbbind_path, f"{'phore_dedup' if dedup else 'phore'}/{pdbid}/{pdbid}_complex.phore")
    tmp_dir = os.path.join(out_path, f"{pdbid}")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    score_file = os.path.join(tmp_dir, f"{pdbid}{'_dedup' if dedup else ''}.score")
    dbphore_file = os.path.join(tmp_dir, f"{pdbid}{'_dedup' if dedup else ''}.dbphore")
    log_file = os.path.join(tmp_dir, f"{pdbid}{'_dedup' if dedup else ''}.log")
    scores = calc_phore_fitting(ligand_file, phore_file, score_file, dbphore_file, log_file)
    if scores is None:
        scores = [-2]
    scores = scores[0]
    return scores


if __name__ == '__main__':
    # print(f"PID: {os.getpid()}")
    # print(f"Current Command: {' '.join(sys.argv)}")
    # args = parse_train_args()
    # construct_dataset(args)
    train_path = '/home/kg108/YJL/DiffPhore/data/cache_torsion/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_EXNeibCut5.0_EX2Phore_phoreRule/'
    val_path = '/home/kg108/YJL/DiffPhore/data/cache_torsion/limit0_INDEXtimesplit_no_lig_overlap_val_maxLigSizeNone_H0_EXNeibCut5.0_EX2Phore_phoreRule/'
    
    pickle.dump(torch.load(os.path.join(train_path, 'heterographs.pkl')), 
                open(os.path.join(train_path, 'heterographs_1.pkl'), 'wb'))
    pickle.dump(torch.load(os.path.join(train_path, 'rdkit_ligands.pkl')), 
                open(os.path.join(train_path, 'rdkit_ligands_1.pkl'), 'wb'))
    pickle.dump(torch.load(os.path.join(val_path, 'heterographs.pkl')), 
                open(os.path.join(val_path, 'heterographs_1.pkl'), 'wb'))
    pickle.dump(torch.load(os.path.join(val_path, 'rdkit_ligands.pkl')), 
                open(os.path.join(val_path, 'rdkit_ligands_1.pkl'), 'wb'))
