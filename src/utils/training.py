import copy
import pickle
import time
from rdkit import Chem
import os
import numpy as np
import torch
from torch import nn
import torch_geometric
from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm
# from confidence.dataset import ListDataset
from utils import so3, torus
from utils.diffusion_utils import get_t_schedule
from utils.generation_utils import get_predict_results
from utils.sampling import randomize_position, sampling_phore, \
    sampling_phore_with_fitscore, calculate_fitscore

class ListDataset(Dataset):
    """
    A custom dataset class that wraps a list of data items.

    Args:
        list (List[Data]): A list containing the data items.

    Attributes:
        data_list (List[Data]): The list of data items.

    Methods:
        len() -> int:
            Returns the number of data items in the dataset.
        
        get(idx: int) -> Data:
            Returns the data item at the specified index.
    """
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class ListFilenameDataset(Dataset):
    """
    A custom dataset class that loads data from a list of filenames.

    Attributes:
        filenameList (list): A list of file paths to the data files.

    Methods:
        len() -> int:
            Returns the number of files in the dataset.
        
        get(idx: int) -> Data:
            Loads and returns the data from the file at the specified index.
            If the 'ligand' attribute of the graph does not have 'orig_pos',
            it calculates 'orig_pos' by adding 'pos' and 'original_center'.
    """
    def __init__(self, filenameList):
        super().__init__()
        self.filenameList = filenameList

    def len(self) -> int:
        return len(self.filenameList)

    def get(self, idx: int) -> Data:
        graph = pickle.load(open(self.filenameList[idx], 'rb'))
        if not hasattr(graph['ligand'], 'orig_pos'):
            # graph['ligand'].orig_pos = copy.deepcopy(graph['ligand'].pos)
            graph['ligand'].orig_pos = graph['ligand'].pos.numpy() + graph.original_center.numpy()
        return graph


def loss_function(tr_pred, rot_pred, tor_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, apply_mean=True, no_torsion=False):
    """
    Computes the loss for translation, rotation, and torsion components.

    Args:
        tr_pred (torch.Tensor): Predicted translation scores.
        rot_pred (torch.Tensor): Predicted rotation scores.
        tor_pred (torch.Tensor): Predicted torsion scores.
        data (list): List of data objects containing ground truth scores and other relevant information.
        t_to_sigma (function): Function to compute sigma values for translation, rotation, and torsion.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        tr_weight (float, optional): Weight for the translation loss component. Default is 1.
        rot_weight (float, optional): Weight for the rotation loss component. Default is 1.
        tor_weight (float, optional): Weight for the torsion loss component. Default is 1.
        apply_mean (bool, optional): Whether to apply mean reduction to the loss components. Default is True.
        no_torsion (bool, optional): Whether to exclude the torsion component from the loss calculation. Default is False.

    Returns:
        tuple: A tuple containing the total loss, translation loss, rotation loss, torsion loss, 
                base translation loss, base rotation loss, and base torsion loss.
    """
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * tr_sigma ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), tr_base_loss, rot_base_loss, tor_base_loss


def loss_function_tank(y_pred, affinity_pred, data, consider_affinity=True, 
                       contact_weight=1, affinity_weight=0.01, pred_dis=True, pose_weight=5):
    contact_loss, affinity_loss = 0, 0
    if consider_affinity:
        affinity_criterion = nn.MSELoss()
        affinity_loss =  affinity_criterion(affinity_pred, data.affinity) * affinity_weight

    if pred_dis:
        criterion = nn.MSELoss()
        contact_loss = criterion(y_pred, data.dis_map) * contact_weight
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pose_weight))
        contact_loss = criterion(y_pred, data.y) * contact_weight
        y_pred = y_pred.sigmoid()
    loss = contact_loss  + affinity_loss 
    return loss, contact_loss.detach() if contact_loss != 0 else 0,  affinity_loss.detach() if affinity_loss != 0 else 0


class AverageMeter():
    """
    A class to compute and store the average and current values of various metrics over specified intervals.

    Attributes:
        types (list): A list of metric types to be tracked.
        intervals (int): The number of intervals over which to track metrics.
        count (int or torch.Tensor): A counter for the number of samples processed.
        acc (dict): A dictionary to accumulate metric values.
        unpooled_metrics (bool): A flag indicating whether to sum unpooled metrics.

    Methods:
        add(vals, interval_idx=None):
            Adds new values to the metrics being tracked.
        
        summary():
            Returns a summary of the average values of the metrics.
    """
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths, confidence_mode=False):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader or DataListLoader): Data loader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
        t_to_sigma (callable): Function to convert time step to sigma value.
        loss_fn (callable): Loss function to compute the training loss.
        ema_weigths (ExponentialMovingAverage): Object to update the exponential moving average of model parameters.
        confidence_mode (bool, optional): Flag to indicate if confidence mode is enabled. Defaults to False.

    Returns:
        dict: Summary of the training metrics for the epoch.
    """
    model.train()
    metrics = ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'] \
        if confidence_mode == False else ['loss', 'loss_ph', 'loss_ex']
    meter = AverageMeter(metrics)

    for data in loader:
    # for data in tqdm(loader, total=len(loader)):
        # print(f"type(data): {type(data)}")
        # print(f"data.device: {data[0].device}")
        skip = any([d.skip if hasattr(d, 'skip') else False for d in data]) \
            if isinstance(loader, DataListLoader) else \
                any(data.skip if hasattr (data, 'skip') else [False])
        if skip:
            continue
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue
        optimizer.zero_grad()
        try:
            if not confidence_mode:
                tr_pred, rot_pred, tor_pred = model(data)
                loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                    loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, device=device)
                meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])
            else:
                fitscore, ph_overlap, ex_overlap = model(data)
                loss, loss_record = loss_fn(data, fitscore, ph_overlap, ex_overlap)
                meter.add([loss_record[k] for k in metrics])
            loss.backward()
            optimizer.step()
            ema_weigths.update(model.parameters())
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch', e)
                # print(e)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch', e)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(f'[E] Failed to calculate the batch {[d.name for d in data]}')
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False, confidence_mode=False):
    """
    Evaluates the model for one epoch on the provided data loader.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        t_to_sigma (callable): Function to convert time to sigma.
        loss_fn (callable): Loss function to compute the loss.
        test_sigma_intervals (bool, optional): If True, compute metrics for different sigma intervals. Default is False.
        confidence_mode (bool, optional): If True, use confidence mode metrics. Default is False.
    Returns:
        dict: A dictionary containing the summary of metrics for the epoch.
    """
    model.eval()
    metrics = ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'] \
        if not confidence_mode else ['loss', 'loss_ph', 'loss_ex']
    meter = AverageMeter(metrics, unpooled_metrics=True)

    if test_sigma_intervals and not confidence_mode:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in loader:
    # for data in tqdm(loader, total=len(loader)):
        # print(data)
        try:
            with torch.no_grad():
                if not confidence_mode:
                    tr_pred, rot_pred, tor_pred = model(data)

                    loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
                        loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
                    if torch.any(torch.isnan(loss)):
                        names = [d.name if hasattr(d, 'name') else '' for d in data] \
                            if isinstance(loader, DataListLoader) else \
                                data.name if hasattr(data, 'name') else ['']
                        print(f"[W] Loss is nan for current batch, ignoring: {names}")
                    else:
                        meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])
                    
                    if test_sigma_intervals > 0:
                        complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([d.complex_t[noise_type] for d in data]) for
                                                                    noise_type in ['tr', 'rot', 'tor']]
                        sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                        sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                        sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                        meter_all.add(
                            [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss],
                            [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                            sigma_index_tor, sigma_index_tr])

                else:
                    fitscore, ph_overlap, ex_overlap = model(data)
                    loss, loss_record = loss_fn(data, fitscore, ph_overlap, ex_overlap)
                    meter.add([loss_record[k] for k in metrics])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                print(f'When dealing with {[d.name for d in data]}, unexpected error occured: {e}.')
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0 and not confidence_mode: out.update(meter_all.summary())
    return out


def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    """
    Perform inference for one epoch on a given model and dataset.
    Args:
        model (torch.nn.Module): The model to be used for inference.
        complex_graphs (list): List of complex graphs to be used as input data.
        device (torch.device): The device (CPU or GPU) to run the inference on.
        t_to_sigma (function): Function to convert time step to sigma value.
        args (argparse.Namespace): Arguments containing various settings and hyperparameters.
    Returns:
        dict: A dictionary containing various metrics and results from the inference process, including:
            - 'rmsds_lt2': Percentage of RMSDs less than 2.
            - 'rmsds_lt5': Percentage of RMSDs less than 5.
            - 'rmsd': List of RMSD values.
            - 'fitscore': List of fit scores.
            - 'run_time': List of run times for each complex.
            - 'failed_indices': List of indices of complexes that failed to converge.
            - 'fitscore_gt0.7': Percentage of fit scores greater than 0.7 (if applicable).
            - 'fitscore_gt0.4': Percentage of fit scores greater than 0.4 (if applicable).
            - 'dock_process': Dictionary containing detailed docking process information (if args.keep_update is True), including:
                - 'name': List of complex names.
                - 'rmsd': List of RMSD values.
                - 'fw_tr_update': List of forward translation updates.
                - 'fw_tor_update': List of forward torsion updates.
                - 'fw_rot_update': List of forward rotation updates.
                - 'rvs_tr_update': List of reverse translation updates.
                - 'rvs_tor_update': List of reverse torsion updates.
                - 'rvs_rot_update': List of reverse rotation updates.
                - 'origin_pos': List of original positions.
                - 'initial_pos': List of initial positions.
                - 'perturb_pos': List of perturbed positions.
                - 'docked_poses': List of docked poses.
                - 'original_centers': List of original centers.
    """
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs) if args.dataset not in ['chembl', 'zinc'] else ListFilenameDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []
    fw_tr_update = []
    fw_tor_update = []
    fw_rot_update = []
    rvs_tr_update = []
    rvs_tor_update = []
    rvs_rot_update = []
    initial_pos = []
    origin_pos = []
    
    perturb_pos = []
    dock_poses = []
    original_centers = []
    fitscore = []
    run_times = []
    names = []
    failed_indices = []

    N = args.sample_per_complex if hasattr(args, 'sample_per_complex') else 1
    idx = -1
    sample_func = sampling_phore_with_fitscore if hasattr(args, 'random_samples') and args.random_samples > 1 else sampling_phore
    for orig_complex_graph in loader:
    # for orig_complex_graph in tqdm(loader):
        idx += 1
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max, keep_update=args.keep_update)
        if args.keep_update:
            perturb_pos.append([g['ligand'].pos.cpu().numpy() for g in data_list])
            original_centers.append([g.original_center.cpu().numpy() for g in data_list])

        if len(orig_complex_graph['ligand'].batch) == 0:
            print(f"[W] Graph {[orig_complex_graph['name']]} with 0 atoms, skipped")
            continue
        predictions_list = None
        failed_convergence_counter = 0
        start_time = time.time()
        while predictions_list == None:
            try:
                # print(type(model))
                # predictions_list, confidences = sampling_phore(data_list=data_list, model=model,
                predictions_list, confidences = sample_func(data_list=data_list, 
                                                            model=model.module if device.type=='cuda' and hasattr(model, 'module') else model,
                                                            inference_steps=args.inference_steps,
                                                            tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                            tor_schedule=tor_schedule,
                                                            device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e

        if failed_convergence_counter > 5: 
            failed_indices.append(idx)
            continue

        name = orig_complex_graph.name[0]
        
        run_times.append(time.time() - start_time)
        if args.no_torsion:
            if not hasattr(orig_complex_graph['ligand'], "orig_pos"):
                orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                        orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(orig_complex_graph['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(rmsd)
        if args.keep_update:
            fw_tr_update.append([g.fw_tr_update for g in predictions_list])
            fw_tor_update.append([g.fw_tor_update for g in predictions_list])
            fw_rot_update.append([g.fw_rot_update for g in predictions_list])
            rvs_tr_update.append([g.rvs_tr_update for g in predictions_list])
            rvs_tor_update.append([g.rvs_tor_update for g in predictions_list])
            rvs_rot_update.append([g.rvs_rot_update for g in predictions_list])
            dock_poses.append([g.docked_poses for g in predictions_list])
            initial_pos.append(orig_complex_graph['ligand'].pos.cpu().numpy())
            origin_pos.append(orig_complex_graph['ligand'].orig_pos)
            names.append(name)

        
        if getattr(args, 'fitscore', False):
            # print(orig_complex_graph)
            mol = Chem.RemoveAllHs(copy.deepcopy(orig_complex_graph.mol[0]))
            dock_pose = ligand_pos + orig_complex_graph.original_center.cpu().numpy()
            store_ranked_pose = args.store_ranked_pose if hasattr(args, 'store_ranked_pose') else False
            phore_file = orig_complex_graph.phore_file[0] if hasattr(orig_complex_graph, 'phore_file') else None
            scores = calculate_fitscore(args, dock_pose, name, mol, phore_file=phore_file, store_ranked_pose=store_ranked_pose)
            if scores is None or len(scores) == 0:
                fitscore.append([-2.0]*N)
                print(f"[W] Warning fitscore calculated with error and set as -2.0 for `{name}`")
            else:
                fitscore.append(scores)

    rmsds = np.array(rmsds).reshape(-1, N) if N != 1 else np.array(rmsds).reshape(-1)
    print(f"rmsd: {', '.join([str(x) for x in rmsds])}")
    fitscore_failed = False 
    try:
        fitscore = np.array(fitscore).reshape(-1, N) if N != 1 else np.array(fitscore).reshape(-1)
    except Exception as e:
        print(f"[W] Error occured when calculating fitscore. {e}")
        fitscore_failed = True
    metrics = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
               'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds) / N),
               'rmsd': rmsds.tolist(), 
               'fitscore': fitscore.tolist() if not fitscore_failed else fitscore,
               'run_time': run_times,
               'failed_indices': failed_indices}

    if len(fitscore) > 0:
        print(f"fitscore: {', '.join([str(x) for x in fitscore])}")
        if not fitscore_failed:
            metrics['fitscore_gt0.7'] = 100 * (fitscore > 0.7).sum() / len(fitscore)
            metrics['fitscore_gt0.4'] = 100 * (fitscore > 0.4).sum() / len(fitscore)

    if args.keep_update:
        metrics['dock_process'] = {
                                  'name': names, 
                                  'rmsd': rmsds, 
                                  'fw_tr_update': fw_tr_update, 
                                  'fw_tor_update': fw_tor_update,
                                  'fw_rot_update': fw_rot_update,
                                  'rvs_tr_update': rvs_tr_update,
                                  'rvs_tor_update': rvs_tor_update,
                                  'rvs_rot_update': rvs_rot_update,
                                  'origin_pos': origin_pos,
                                  'initial_pos': initial_pos,
                                  'perturb_pos': perturb_pos,
                                  'docked_poses': dock_poses,
                                  'original_centers': original_centers}

    # if args.keep_update:
    #     results = pd.DataFrame({'name': [g.name for g in predictions_list], 
    #                             'rmsd': rmsds, 'fw_tr_update': [g for g in predictions_list]})
    return metrics


def train_tank_epoch(model, loader, optimizer, device, loss_fn, consider_affinity=True):
    model.train()
    meter = AverageMeter(['loss', 'contact_loss', 'affinity_loss'], unpooled_metrics=True)

    for data in loader:
    # for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            y_pred, affinity_pred = model(data)
            loss, contact_loss, affinity_loss = \
                loss_fn(y_pred, affinity_pred, data=data, device=device, consider_affinity=consider_affinity)
            loss.backward()
            optimizer.step()
            meter.add([loss.cpu().detach(), contact_loss, affinity_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    return meter.summary()


def test_tank_epoch(model, loader, device, loss_fn, consider_affinity=True):
    model.eval()
    meter = AverageMeter(['loss', 'contact_loss', 'affinity_loss'], unpooled_metrics=True)

    for data in loader:
    # for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                y_pred, affinity_pred = model(data)
            loss, contact_loss, affinity_loss = \
                loss_fn(y_pred, affinity_pred, data=data, device=device, consider_affinity=consider_affinity)
            meter.add([loss.cpu().detach(), contact_loss, affinity_loss])


        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    out = meter.summary()
    return out


def inference_tank_epoch(model, graph_dataset, device, args):
    # dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=graph_dataset, batch_size=1, shuffle=False)
    rmsds = []

    for orig_complex_graph in loader:
    # for orig_complex_graph in tqdm(loader):
        # data_list = [copy.deepcopy(orig_complex_graph)]

        # randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)
        _orig_complex_graph = copy.deepcopy(orig_complex_graph)

        predictions_list = get_predict_results(model, _orig_complex_graph, args.remove_hs, 
                                               show_progress=False, device=device)

        # filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()
        # if isinstance(_orig_complex_graph['ligand'].orig_pos, list):
        #     _orig_complex_graph['ligand'].orig_pos = _orig_complex_graph['ligand'].orig_pos[0]

        # ligand_pos = np.asarray(
        #     [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        # orig_ligand_pos = np.expand_dims(
        #     _orig_complex_graph['ligand'].orig_pos[filterHs] - _orig_complex_graph.original_center.cpu().numpy(), axis=0)
        # rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(predictions_list.sort_values(['loss'], ascending=True).iloc[0, :]['rmsd'])
    rmsds = np.array(rmsds)
    print(f"rmsd: {rmsds}")
    _rmsds = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}
    return _rmsds
