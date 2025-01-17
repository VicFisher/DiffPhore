from argparse import Namespace
import os
import signal
import subprocess
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import torch
import yaml

from models.score_model_phore import TensorProductScoreModel as PhoreModel
from rdkit import Chem
from rdkit.Chem import MolToPDBFile, RemoveHs
from spyrmsd import molecule, rmsd
from torch_geometric.nn.data_parallel import DataParallel
from utils.diffusion_utils import get_timestep_embedding



def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    """
    Calculate the Root Mean Square Deviation (RMSD) between two molecular structures using Open Babel.

    Args:
    mol1_path (str or object): Path to the first molecule file or a molecule object.
    mol2_path (str or object): Path to the second molecule file or a molecule object.
    cache_name (str, optional): Name for the cache file. If None, a name based on the current date and time will be generated.

    Returns:
    np.ndarray: An array of RMSD values.

    Notes:
    - If the input paths are not strings, they are assumed to be molecule objects and will be converted to PDB files.
    - The function uses Open Babel's `obrms` command to calculate the RMSD.
    - Temporary files are stored in the `.openbabel_cache` directory.
    """
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    """
    Remove all hydrogen atoms from a molecule, with various customizable parameters.

    Parameters:
    mol (rdkit.Chem.Mol): The molecule from which hydrogen atoms will be removed.

    Returns:
    rdkit.Chem.Mol: The molecule with all specified hydrogen atoms removed.
    """
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=args.lr_decay_factor,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 1000)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, contrastive_model=None):
    # Initialize the model
    if args.model_type == 'diff': 
        timestep_emb_func = get_timestep_embedding(
            embedding_type=args.embedding_type,
            embedding_dim=args.sigma_embed_dim,
            embedding_scale=args.embedding_scale)

        model = PhoreModel(t_to_sigma=t_to_sigma,
                           device=device,
                           no_torsion=args.no_torsion,
                           timestep_emb_func=timestep_emb_func,
                           num_conv_layers=args.num_conv_layers,
                           lig_max_radius=args.max_radius,
                           scale_by_sigma=args.scale_by_sigma,
                           sigma_embed_dim=args.sigma_embed_dim,
                           ns=args.ns, nv=args.nv,
                           distance_embed_dim=args.distance_embed_dim,
                           cross_distance_embed_dim=args.cross_distance_embed_dim,
                           batch_norm=not args.no_batch_norm,
                           dropout=args.dropout,
                           use_second_order_repr=args.use_second_order_repr,
                           cross_max_distance=args.cross_max_distance,
                           dynamic_max_cross=args.dynamic_max_cross,
                           confidence_mode=confidence_mode,
                           consider_norm=args.consider_norm,
                           use_phore_rule=args.phore_rule,
                           auto_phorefp=args.auto_phorefp,
                           angle_match=args.angle_match,
                           cross_distance_transition=args.cross_distance_transition,
                           phore_direction_transition=args.phore_direction_transition,
                           phoretype_match_transition=args.phoretype_match_transition,
                           new=args.new,
                           ex_factor=args.ex_factor,
                           boarder=getattr(args, 'boarder', False), 
                           by_radius=getattr(args, 'by_radius', False),
                           clash_tolerance=getattr(args, 'clash_tolerance', 0.4),
                           clash_cutoff=getattr(args, 'clash_cutoff', [1.0, 2.0, 3.0, 4.0, 5.0]),
                           use_att=getattr(args, 'use_att', False),
                           use_phore_match_feat=getattr(args, 'use_phore_match_feat', False),
                           num_confidence_outputs=len(
                               args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                args.rmsd_classification_cutoff, list) else 1,
                           atom_weight=getattr(args, 'atom_weight', 'softmax'),
                           trioformer_layer=getattr(args, 'trioformer_layer', 1),
                           contrastive_model=contrastive_model,
                           contrastive_node=getattr(args, 'return_node', False),
                           norm_by_ph=getattr(args, 'norm_by_ph', False),
                           dist_for_fitscore=getattr(args, 'dist_for_fitscore', False),
                           angle_for_fitscore=getattr(args, 'angle_for_fitscore', False),
                           type_for_fitscore=getattr(args, 'type_for_fitscore', False),
                           sigmoid_for_fitscore=getattr(args, 'sigmoid_for_fitscore', False),
                           readout=getattr(args, 'readout', 'mean'),
                           as_exp=getattr(args, 'as_exp', False),
                           scaler=getattr(args, 'scaler', 1.0)
                        )
    

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    if args.debug:
        print(model)
    return model


def load_contrastive_model(args, t_to_sigma, device):
    contrastive_model = None
    if args.contrastive and os.path.exists(args.contrastive_model_dir):
        try:
            contrastive_model_cfg = yaml.load(open(os.path.join(args.contrastive_model_dir, 'model_parameters.yml'), 'r'), Loader=yaml.FullLoader)
            # self.contrastive_model_cfg = pickle.load(open(os.path.join(contrastive_model_dir, 'model_configs.pkl'), 'rb'))
            contrastive_ns = contrastive_model_cfg['ns']
            if contrastive_model_cfg['num_conv_layers'] >= 3:
                contrastive_ns *= 2
            checkpoint = os.path.join(args.contrastive_model_dir, 'best_model.pt')
            contrastive_args = Namespace(**contrastive_model_cfg)
            # device = torch.device('cpu')
            contrastive_model_ckp = torch.load(checkpoint, map_location=torch.device('cpu'))
            print(f"[I] Contrastive model loaded from {checkpoint}")
            # t_to_sigma_contrastive = partial(t_to_sigma_compl, args=contrastive_args)
            contrastive_model = get_model(contrastive_args, device, t_to_sigma, no_parallel=True, confidence_mode=True)
            contrastive_model.load_state_dict(contrastive_model_ckp['model'])
            contrastive_model.eval()
            contrastive_model.requires_grad_(False)
        except Exception as e:
            print(f"[E] Failed to load contrastive model from `{args.contrastive_model_dir}`. \n{e}")
            raise(e)
    return contrastive_model


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    """
    Calculate the symmetry-corrected Root Mean Square Deviation (RMSD) between two sets of coordinates.

    Args:
    mol (rdkit.Chem.Mol): The reference molecule.
    coords1 (numpy.ndarray): The first set of coordinates.
    coords2 (numpy.ndarray): The second set of coordinates.
    mol2 (rdkit.Chem.Mol, optional): An optional second molecule for comparison. Defaults to None.

    Returns:
    float: The symmetry-corrected RMSD value.
    """
    with time_limit(100):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        
        ## To guarantee correct behavior when fine-tuning a model (requires_grad of model input part parameters is forced to set as False)
        self.shadow_params = [p.clone().detach() for p in parameters]
        # self.shadow_params = [p.clone().detach() for p in parameters]
                            #   for p in parameters if p.requires_grad]
        self.collected_params = []

    def requires_grad_(self):
        ## To guarantee correct behavior when fine-tuning a model (requires_grad of model input part parameters is forced to set as False)
        for p in self.shadow_params:
            p.requires_grad = True

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters]
            # parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                if s_param.requires_grad and param.requires_grad:
                    s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        ## To guarantee correct behavior when fine-tuning a model (requires_grad of model input part parameters is forced to set as False)
        parameters = [p for p in parameters]
        # parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if s_param.requires_grad and param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        s = model.state_dict().keys()
        mapping = {k: 'encoder.'+k for k in state_dict.keys() if k not in s}
        state_dict = {mapping[k] if k in mapping else k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    return model

