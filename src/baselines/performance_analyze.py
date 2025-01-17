import os
import json
import time
import pandas as pd
import copy
import numpy as np
import pickle

def convert_to_pdb(inpfile, outdir):
    """
    Converts a given molecular file to PDB format using Open Babel.

    Parameters:
    inpfile (str): The path to the input file that needs to be converted.
    outdir (str): The directory where the output PDB file will be saved.

    Returns:
    str: The path to the converted PDB file if conversion is successful, otherwise None.

    Notes:
    - The function first tries to convert the input file using its original format.
    - If the conversion fails and the input file is in SDF format, it attempts to convert a corresponding MOL2 file.
    - The function uses the Open Babel command-line tool for conversion.
    - If the output file already exists, the function will not attempt conversion again.
    """
    suffix = inpfile.split('.')[-1]
    outfile = os.path.join(outdir, os.path.basename(inpfile).replace(suffix, 'pdb'))
    if not os.path.exists(outfile):
        command = f"obabel -i{suffix} {inpfile} -opdb -O {outfile} > /dev/null"
        status = os.system(command)
        if status != 0 and os.path.exists(inpfile.replace('sdf', 'mol2')):
            print(f"[W] Failed to convert {inpfile} to pdb, trying mol2 format...")
            outfile = convert_to_pdb(inpfile.replace("sdf", 'mol2'), outdir)

    if os.path.exists(outfile):
        return outfile
    else:
        return None

def get_obrmsd(refpdb, predpdb, log_file):
    """
    Calculate the RMSD (Root Mean Square Deviation) between a reference PDB file and a predicted PDB file using the 'obrms' command.

    Args:
        refpdb (str): Path to the reference PDB file.
        predpdb (str): Path to the predicted PDB file.
        log_file (str): Path to the log file where the 'obrms' command output will be stored.

    Returns:
        list of float: A list of RMSD values. If the calculation fails, a list of ten 999.0 values is returned.
    """
    command = f"obrms  {predpdb} {refpdb}> {log_file}"
    # print(command)
    status = os.system(command)
    try:
        rmsds = [float(l.split()[-1]) for l in open(log_file, 'r').readlines()][:10]
    except Exception as e:
        print(f"[E] Failed to calculate the rmsd between {os.path.basename(refpdb)} and {os.path.basename(predpdb)}, setting to 999.")
        rmsds = [999.0] * 10
    return rmsds

def collect_all_records():
    """
    Collects and processes docking records for PDBBind and PoseBusters datasets.
    This function reads lists of PDB IDs for the PDBBind and PoseBusters datasets, converts ligand files to PDB format,
    performs docking using various baselines, calculates RMSD values, and collects status and time cost information.
    The results are stored in a list of dictionaries.
    Returns:
        list: A list of dictionaries, each containing the following keys:
            - 'dataset' (str): The dataset name ('pdbbind' or 'posebusters').
            - 'baseline' (str): The docking baseline used ('vina', 'gnina', 'smina', or 'unidock').
            - 'pdbid' (str): The PDB ID of the processed structure.
            - 'rmsd' (float): The calculated RMSD value between the original and docked structures.
            - 'status' (str): The status of the docking process.
            - 'time_cost' (str): The time cost of the docking process.
    """
    records = []
    pdbbind_list = '/home/worker/users/YJL/DiffPhore/data/splits/timesplit_test'
    pdbbind_dir = "/home/worker/users/YJL/DiffPhore/data/PDBBind/all"
    posebusters_dir = "/home/worker/users/YJL/DiffPhore/data/PoseBusters/all"
    pdbbind_docking_dir = "/home/worker/users/YJL/DiffPhore/experiments/baselines/output/align/pdbbind/complex/"
    posebusters_docking_dir = "/home/worker/users/YJL/DiffPhore/experiments/baselines/output/align/posebusters/complex"
    posebusters_list = '/home/worker/users/YJL/DiffPhore/data/splits/posebusters_test_all'
    pdbbind_pdb_dir = "/home/worker/users/YJL/DiffPhore/data/PDBBind/pdb"
    posebusters_pdb_dir = "/home/worker/users/YJL/DiffPhore/data/PoseBusters/pdb"

    pdbbind_l = [l.strip() for l in open(pdbbind_list, 'r').readlines()]
    posebusters_l = [l.strip() for l in open(posebusters_list, 'r').readlines()]

    for dataset in ['pdbbind', 'posebusters']:
        l = pdbbind_l if dataset == 'pdbbind' else posebusters_l
        dock_path = pdbbind_docking_dir if dataset == 'pdbbind' else posebusters_docking_dir
        inp_path = pdbbind_dir if dataset == 'pdbbind' else posebusters_dir
        out_path = pdbbind_pdb_dir if dataset == 'pdbbind' else posebusters_pdb_dir
        os.makedirs(out_path, exist_ok=True)

        for pdbid in l:
            input_file = os.path.join(inp_path, f"{pdbid}/{pdbid}_ligand.sdf")
            outpdb = convert_to_pdb(input_file, out_path)

            for baseline in ['vina', 'gnina', 'smina', 'unidock']:
                base_dir = os.path.join(dock_path, f"{baseline}/{pdbid}")
                if baseline == 'vina':
                    docked_file = f"{pdbid}.pdbqt"
                elif baseline in ['gnina', 'smina']:
                    docked_file = f"{pdbid}.sdf"
                else:
                    docked_file = f"{pdbid}_ligand_uni.sdf"
                docked_file = os.path.join(base_dir, docked_file)
                docked_pdb_file = convert_to_pdb(docked_file, base_dir)
                if docked_pdb_file is None:
                    print(f"[E] Failed to convert docked file to pdb format ({dataset}-{baseline}: {pdbid})")
                
                rmsd_file = os.path.join(base_dir, f"{pdbid}.rmsd")
                rmsd = get_obrmsd(outpdb, docked_pdb_file, rmsd_file)

                status_file = os.path.join(base_dir, f"{pdbid}.status")
                status, cost = [line.strip() for line in open(status_file, 'r').readlines() if line.strip() != ""]
                records.append({'dataset': dataset, 'baseline': baseline, 'pdbid': pdbid, 'rmsd': rmsd, 'status': status, 'time_cost': cost})
    return records

def performance_analyze(records):
    """
    Analyzes the performance of various baselines on different datasets and saves the results.
    Args:
    records (list of dict): A list of dictionaries where each dictionary contains the performance data for a single record.
   
    The function performs the following steps:
    1. Converts the input records into a pandas DataFrame.
    2. Iterates over two datasets: 'pdbbind' and 'posebusters'.
    3. For each dataset, reads a list of non-overlapping test cases.
    4. Iterates over four baselines: 'vina', 'gnina', 'smina', and 'unidock'.
    5. For each baseline and dataset combination, calculates performance metrics including:
       - Number of valid records.
       - Mean time cost.
       - Top-k RMSD values and their statistics (percentage below 1 and 2, median).
    6. Separately calculates these metrics for the non-overlapping test cases.
    7. Appends the calculated metrics to a list.
    8. Converts the list of performance metrics into a DataFrame and sorts it.
    9. Saves the performance metrics to a CSV file.
    10. Saves a subset of the performance metrics to a pickle file.
    The results are saved to:
    - '/home/worker/users/YJL/DiffPhore/experiments/baselines/output/docking_performance.csv'
    - '/home/worker/users/YJL/DiffPhore/experiments/baselines/output/docking_rmsd_topk_cache.pkl'
    """
    df = pd.DataFrame(records)
    performance_metrics = []

    for dataset in ['pdbbind', 'posebusters']:
        no_overlap_list = "timesplit_test_no_rec_overlap" if dataset == 'pdbbind' else "posebusters_test_no_overlap"
        no_overlap_list = os.path.join("/home/worker/users/YJL/DiffPhore/data/splits", no_overlap_list)
        no_overlap_list = [line.strip() for line in open(no_overlap_list, 'r').readlines() if line.strip() != ""]
        for baseline in ['vina', 'gnina', 'smina', 'unidock']:
            performance = {}
            performance_no_overlap = {}

            performance['dataset'] = dataset
            performance['baseline'] = baseline
            performance_no_overlap.update(performance)

            performance['no_overlap'] = False
            performance_no_overlap['no_overlap'] = True


            df_valid = df[(df['dataset'] == dataset) & (df['baseline']==baseline) & (df['status'] == '0')]
            performance['num_valid'] = len(df_valid)
            
            df_valid_no_overlap = df_valid[df_valid['pdbid'].isin(no_overlap_list)]
            performance_no_overlap['num_valid'] = len(df_valid_no_overlap)
            performance['mean_time'] = df_valid['time_cost'].astype(float).mean()
            performance_no_overlap['mean_time'] = df_valid_no_overlap['time_cost'].astype(float).mean()
            for topk in [1, 5]:
                topk_best = df_valid['rmsd'].map(lambda x: min(x[:topk]) if len(x)>0 else 999.).values
                topk_best_no_overlap = df_valid_no_overlap['rmsd'].map(lambda x: min(x[:topk]) if len(x)>0 else 999.).values
                performance[f'top{topk}_rmsds'] = topk_best
                performance_no_overlap[f'top{topk}_rmsds'] = topk_best_no_overlap

                performance[f'top{topk}_rmsd_lt_1'] = (topk_best < 1).mean() * 100
                performance[f'top{topk}_rmsd_lt_2'] = (topk_best < 2).mean() * 100
                # performance[f'top{topk}_rmsd_lt_5'] = (topk_best < 5).mean()
                performance[f'top{topk}_rmsd_med'] = np.median(topk_best)
                performance_no_overlap[f'top{topk}_rmsd_lt_1'] = (topk_best_no_overlap < 1).mean() * 100
                performance_no_overlap[f'top{topk}_rmsd_lt_2'] = (topk_best_no_overlap < 2).mean() * 100
                # performance_no_overlap[f'top{topk}_rmsd_lt_5'] = (topk_best_no_overlap < 5).mean()
                performance_no_overlap[f'top{topk}_rmsd_med'] = np.median(topk_best_no_overlap)
            performance_metrics.append(copy.deepcopy(performance))
            performance_metrics.append(copy.deepcopy(performance_no_overlap))
    

    df_performance = pd.DataFrame(performance_metrics).sort_values(['no_overlap', 'dataset', 'baseline'])
    sele = [c for c in df_performance.columns if 'rmsds' not in c]
    _sele = ['no_overlap', 'dataset', 'baseline'] + [c for c in df_performance.columns if 'rmsds' in c]
    df_performance[sele].to_csv('/home/worker/users/YJL/DiffPhore/experiments/baselines/output/docking_performance.csv', index=False)
    pickle.dump(pd.DataFrame(df_performance[_sele]), open('/home/worker/users/YJL/DiffPhore/experiments/baselines/output/docking_rmsd_topk_cache.pkl', 'wb'))


if __name__ == "__main__":
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current PID: {os.getpid()}")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Hostname: $(hostname)")
    record_file = "/home/worker/users/YJL/DiffPhore/experiments/baselines/output/all_docking_records.json"
    if not os.path.exists(record_file):
        records = collect_all_records()
        json.dump(records, open(record_file, 'w'), indent=4)
    else:
        print("[I] RMSD calcuated before, loading cache ...")
        records = json.load(open(record_file, 'r'))

    performance_analyze(records)

    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Finished.")

