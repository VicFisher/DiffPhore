import os
import json
import time
import pandas as pd
import copy
import numpy as np
import pickle

def convert_to_pdb(inpfile, outdir):
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

def target_fishing_results():
    targets = [
        {'drug': '4OH_Tamoxifen', 'targets': 'Estrogen-related receptor gamma', 'Entry': 'ERR3_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Estrogen receptor alpha', 'Entry': 'ESR1_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Estrogen receptor beta', 'Entry': 'ESR2_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': '17beta-Hydroxysteroid dehydrogenase', 'Entry': 'O93874_COCLU'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Cyclooxygenase-2', 'Entry': 'PGH2_MOUSE'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Glutathione S-transferase', 'Entry': 'GSTA1_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': '3alpha-Hydroxysteroid dehydrogenase', 'Entry': 'AK1C2_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_STAAU'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_MYCTU'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_ECOLI'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'Q81R22_BACAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_PNECA'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Dihydrofolate reductase', 'Entry': 'DYR_LACCA'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Calmodulin', 'Entry': 'CALM_BOVIN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Calmodulin', 'Entry': 'CALM_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Protein kinase c-theta type', 'Entry': 'KPCB_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Human fibroblast collagenase', 'Entry': 'MMP1_HUMAN'},
        {'drug': '4OH_Tamoxifen', 'targets': 'Alcohol dehydrogenase', 'Entry': 'ADH1S_HORSE'},

    ]


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

