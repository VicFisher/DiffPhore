import copy
import shutil
import time
import json
import os, sys
import argparse
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
from functools import partial
from datasets.process_pharmacophore import write_phore_to_file,\
    generate_random_exclusion_volume, parse_phore, extract_random_phore_from_origin
import pandas as pd
from pandarallel import pandarallel
import copy

ANCPHORE_PATH = os.path.join(os.path.dirname(__file__), '../programs/AncPhore_11')
PHARAO_PATH = os.path.join(os.path.dirname(__file__), '../programs/baselines/pharao')
PHARMER_PATH = os.path.join(os.path.dirname(__file__), '../programs/baselines/pharmer')
CMD_TEMP = {
    'pharmer': {
        'phor_gen': "{}/pharmer pharma -in {}{} -out {} > {} 2>&1",
        "dbcreate": "{}/pharmer dbcreate -dbdir {} -in {} > {} 2>&1",
        "dbsearch": "{}/pharmer dbsearch -dbdir {} -in {} -out {} > {} 2>&1"
    },
    'pharao': {
        "phor_gen": "{}/pharao -d {} -p {} > {} 2>&1",
        "align": "{}/pharao --reference {} -d {} -o {} -s {} > {} 2>&1"
    },
    'ancphore': {
        'phor_gen': "{} --refphore {} -l {}{} > {} 2>&1",
        "align": "{} --refphore {} -d {} --mol {} --scores {} usedMultiConformerFile > {} 2>&1"
    },
    'conf_gen': "obabel {} -osdf -O {} --conformer --nconf {} --writeconformers > {} 2>&1"
}

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', default='align', type=str, help='The task to conduct, `align`, `screen` or `fishing`')
    parser.add_argument('--mode', default='ligand', type=str, help='The way of pharmacophore generation, `ligand` or `complex`')
    parser.add_argument('--dataset', default='pdbbind', type=str, help='The dataset, `pdbbind` or `posebusters`')
    parser.add_argument('--baseline', default='ancphore', type=str, help='The baseline method, `ancphore`, `pharmer`, `pharao`')
    parser.add_argument('--out_dir', default='../experiments/baselines/output/', type=str, help='The output directory')
    parser.add_argument('--num_conformers', default=40, type=int, help='The number of conformations to generate')
    parser.add_argument('--nworkers', default=20, type=int, help='The number of conformations to generate')
    parser.add_argument('--drug', default=None, type=str, help='The drug name')
    args = parser.parse_args()
    return args


def evaluate(args):
    """
    High level evaluation task manager
    """
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.baseline == 'ancphore':
        align = ancphore_align
    elif args.baseline == 'pharmer':
        align = pharmer_align
    elif args.baseline == 'pharao':
        align = pharao_align
    else:
        raise ValueError(f'Unknown baseline: {args.baseline}')

    if args.task == 'align':
        _dataset = get_dataset(args.dataset)
    elif args.task == 'screen':
        _dataset = get_dude(conformation=True, n_conf=args.num_conformers, overwrite=False, conf_per_file=6000)
        if args.baseline == 'ancphore':
            if args.mode == 'complex':
                align = partial(ancphore_align, anchor=True)
            elif args.mode == 'ligand':
                align = partial(ancphore_align, use_ex=False)
    elif args.task == 'fishing':
        _dataset = get_ifptarget(drug=args.drug)
        align = partial(ancphore_align, anchor=True, split=False)
        args.mode = 'complex'

    else:
        raise ValueError(f'Unknown task: {args.task}')

    # Status Code
    # 1     Conformation Generation Failure
    # 1.1   Database Creation Failure (pharmer only)
    # 1.2   Pharmacophore Generation Failure
    # 1.3   Random Pharmacophore Sampling Failure (ancphore only)
    # 2     Pharmacophore Alignemnt Failure
    # 3     Too many pharmacophore (pharao only)
    # print(_dataset)
    
    results = align(args.mode, _dataset, args.out_dir, num_conformers=args.num_conformers, nworkers=args.nworkers)
    suffix = "" if args.num_conformers == 40 else f"_{args.num_conformers}"
    json.dump(results, open(os.path.join(args.out_dir, f'results{suffix}.json'), 'w'), indent=4)

    # analyze_results(results)
    return results


def pharmer_align(mode, dataset, out_dir, num_conformers=40, 
                  pharmer_path=PHARMER_PATH, **kwargs):
    """
    Pharmacophore matchingg with Pharmer tool.
    Args:
    mode (str): Mode of alignment, either 'ligand' or 'receptor'.
    dataset (list): List of dictionaries containing dataset information. Each dictionary should have keys 'id', 'dm', 'l', and 'p'.
    out_dir (str): Output directory where results and intermediate files will be stored.
    num_conformers (int, optional): Number of conformers to generate. Default is 40.
    pharmer_path (str, optional): Path to the Pharmer executable. Default is PHARMER_PATH.
    **kwargs: Additional keyword arguments.
    
    Returns:
    list: A list of dictionaries containing the results of the alignment process. Each dictionary includes the status and other relevant information.
    Notes:
    - The function performs the following steps for each entry in the dataset:
        1. Generates conformations.
        2. Creates a database using Pharmer.
        3. Generates pharmacophores.
        4. Searches the database for alignments.
    - Intermediate and result files are stored in the specified output directory.
    - If any step fails for a dataset entry, the status is updated and the process continues with the next entry.
    """
    results = []
    process_dir = os.path.join(out_dir, 'process')
    if not os.path.exists(process_dir):
        os.makedirs(process_dir, exist_ok=True)
    for data in dataset:
        result = {'status': 0}
        result.update(data)
        cache_path = os.path.abspath(os.path.join(process_dir, result['id']))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
        
        # 1.1 Conformation Generation
        generate_conformation(cache_path, result, num_conformers, False)
        if result['status'] != 0:
            copyTo(result, results)
            continue
        
        # 1.2 Create Database
        db_path = os.path.join(cache_path, f"{result['id']}_database")
        db_log = os.path.join(cache_path, f"{result['id']}_database.log")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        create_cmd = CMD_TEMP['pharmer']['dbcreate'].format(pharmer_path, db_path, result['dm'], db_log)
        status = os.system(create_cmd)
        if status != 0 or not os.path.exists(db_path):
            result['status'] = 1.1
            print(f"[W] Failed to create database for {result['id']}")
            copyTo(result, results)
            continue

        # 2.Pharmacophore Generation
        rec_flag = "" if mode == 'ligand' else f" -receptor {result['p']}"
        phar_file = os.path.join(cache_path, f"{result['id']}_pharmacophore.json")
        phar_log = os.path.join(cache_path, f"{result['id']}_pharmacophore.log")
        phar_cmd = CMD_TEMP['pharmer']['phor_gen'].format(pharmer_path, 
                                                    result['l'], rec_flag, phar_file, phar_log)
        status = os.system(phar_cmd)
        if status != 0 and not is_valid_file(phar_file):
            result['status'] = 1.2
            print(f"[W] Failed to generate pharmacophore for {result['id']}")
            copyTo(result, results)
            continue

        # 3.Search Database
        search_std = time.time()
        aligned_file = os.path.join(cache_path, f"{result['id']}_aligned.sdf")
        out_log = os.path.join(cache_path, f"{result['id']}_aligned.log")
        search_cmd = CMD_TEMP['pharmer']['dbsearch'].format(pharmer_path, db_path,
                                                        phar_file, aligned_file, out_log)
        status = os.system(search_cmd)
        if status != 0 or not os.path.exists(aligned_file):
            result['status'] = 2
            print(f"[W] Failed to search database for {result['id']}")
            copyTo(result, results)
            continue
        
        result['align_time'] = time.time() - search_std
        result['aligned_file'] = aligned_file
        copyTo(result, results)

    return results


def pharao_align(mode, dataset, out_dir, num_conformers=40, 
                 pharao_path=PHARAO_PATH, max_num=25, **kwargs):
    """
    Pharmacophore matchingg with Pharao tool.

    Args:
        mode (str): The mode of operation, must be 'ligand'.
        dataset (list): A list of dictionaries containing ligand data.
        out_dir (str): The output directory where results will be stored.
        num_conformers (int, optional): The number of conformers to generate. Defaults to 40.
        pharao_path (str, optional): The path to the Pharao executable. Defaults to PHARAO_PATH.
        max_num (int, optional): The maximum number of pharmacophores allowed. Defaults to 25.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of dictionaries containing the results of the alignment process.
    """
    assert mode == 'ligand', 'Pharao only supports ligand mode'
    results = []
    process_dir = os.path.join(out_dir, 'process')
    if not os.path.exists(process_dir):
        os.makedirs(process_dir, exist_ok=True)

    for data in dataset:
        result = {'status': 0}
        result.update(data)
        cache_path = os.path.abspath(os.path.join(process_dir, result['id']))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        # 1.Conformation Generation
        generate_conformation(cache_path, result, num_conformers, split=False)
        if result['status'] != 0:
            copyTo(result, results)
            continue

        # 1.2 Generation Pharmacophore
        ref_phore = os.path.join(cache_path, f"{result['id']}_pharmacophore.phore")
        ref_log = os.path.join(cache_path, f"{result['id']}_pharmacophore.log")
        phor_cmd = CMD_TEMP['pharao']['phor_gen'].format(pharao_path, result['l'], ref_phore, ref_log)
        os.system(phor_cmd)
        if is_valid_file(ref_phore):
            num = check_phore_num(ref_phore)
            if num > max_num:
                print(f"[W] To many pharmacophores ({num}>{max_num}) for {result['id']}. Pharmacophore alignment will take extremely long time. Skipped.")
                result['status'] = 3
                copyTo(result, results)
                continue

        # 2.Align
        phar_std = time.time()
        ref_file = result['l']
        db_file = result['dm']
        aligned_file = os.path.join(cache_path, f"{result['id']}_aligned.sdf")
        phar_log = os.path.join(cache_path, f"{result['id']}_aligned.log")
        score_file = os.path.join(cache_path, f"{result['id']}_aligned.score")
        phar_cmd = CMD_TEMP['pharao']['align'].format(pharao_path, ref_file, db_file, aligned_file, score_file, phar_log)
        status = os.system(phar_cmd)
        if status != 0 or not is_valid_file(score_file):
            result['status'] = 2
            print(f"[W] Failed to align pose for {result['id']}")
            copyTo(result, results)
            continue

        result['align_time'] = time.time() - phar_std
        result['aligned_file'] = aligned_file
        result['aligand_score'] = score_file

        copyTo(result, results)

    return results


def ancphore_align(mode, dataset, out_dir, num_conformers=40, split=True, conf_per_file=6000,
                   ancphore_path=ANCPHORE_PATH, anchor=False, random=True, use_ex=True, nworkers=1, 
                   **kwargs):
    """
    Pharmacophore matchingg with AncPhore tool.

    Args:
    mode (str): The mode in which to run the alignment.
    dataset (list): A list of datasets to be processed.
    out_dir (str): The output directory where results will be stored.
    num_conformers (int, optional): Number of conformers to generate. Default is 40.
    split (bool, optional): Whether to split the dataset. Default is True.
    conf_per_file (int, optional): Number of conformers per file. Default is 6000.
    ancphore_path (str, optional): Path to the AncPhore executable. Default is ANCPHORE_PATH.
    anchor (bool, optional): Whether to use anchor. Default is False.
    random (bool, optional): Whether to use random seed. Default is True.
    use_ex (bool, optional): Whether to use external resources. Default is True.
    nworkers (int, optional): Number of parallel workers. Default is 1.
    **kwargs: Additional keyword arguments.

    Returns:
    list: A list of results from the alignment process.
    """
    #./AncPhore --refphore test/1dg9_ancphore.phore -d test/1dg9_multi.sdf --mol test/1dg9_ancphore_aligned.sdf --score test/1dg9_ancphore_score.txt usedMultiConformerFile
    results = []
    suffix = '' if num_conformers == 40 else f"_{num_conformers}"
    process_dir = os.path.join(out_dir, f'process{suffix}')
    if not os.path.exists(process_dir):
        os.makedirs(process_dir, exist_ok=True)
    print("process_dir:", process_dir)
    if nworkers > 1:
        pandarallel.initialize(nb_workers=nworkers)
        df_dataset = pd.DataFrame()
        df_dataset['records'] = dataset
        df_dataset['results'] = df_dataset.parallel_apply(lambda x: ancphore_align_once(mode, x['records'], process_dir, 
                                             num_conformers=num_conformers, split=split, conf_per_file=conf_per_file,
                                             ancphore_path=ancphore_path, anchor=anchor, random=random, use_ex=use_ex, 
                                             **kwargs), axis=1)
        results.extend(df_dataset['results'].tolist())
    else:
        for data in dataset:
            result = ancphore_align_once(mode, data, process_dir, num_conformers=num_conformers, split=split, conf_per_file=conf_per_file,
                                ancphore_path=ancphore_path, anchor=anchor, random=random, use_ex=use_ex, **kwargs)
            copyTo(result, results)
    return results


def ancphore_align_once(mode, data, process_dir, num_conformers=40, split=True, conf_per_file=6000,
                        ancphore_path=ANCPHORE_PATH, anchor=False, random=True, use_ex=True, 
                        **kwargs):
    """
    Pharmacophores matchingg with AncPhore.
    Args:
    mode (str): The mode of operation, either 'ligand' or 'complex'.
    data (dict): A dictionary containing data related to the drug and its ID.
    process_dir (str): The directory where the process files will be stored.
    num_conformers (int, optional): The number of conformers to generate. Default is 40.
    split (bool, optional): Whether to split the conformers into multiple files. Default is True.
    conf_per_file (int, optional): The number of conformers per file. Default is 6000.
    ancphore_path (str, optional): The path to the AncPhore executable. Default is ANCPHORE_PATH.
    anchor (bool, optional): Whether to use an anchor for pharmacophore generation. Default is False.
    random (bool, optional): Whether to randomly select features for ligand-based pharmacophore. Default is True.
    use_ex (bool, optional): Whether to use exclusion spheres. Default is True.
    **kwargs: Additional keyword arguments.
    Returns:
    dict: A dictionary containing the result of the alignment process, including status, aligned files, scores, and timing information.
    Raises:
    Exception: If there is an error in generating random pharmacophore.
    """
    result = {'status': 0}
    result.update(data)
    
    cache_path = os.path.abspath(os.path.join(process_dir, result.get('drug', ''), result['id']))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)

    # print("cache_path:", cache_path)


    # 1.Conformation Generation
    generate_conformation(cache_path, result, num_conformers, split=split, conf_per_file=conf_per_file)
    if result['status'] != 0:
        return result

    # 2. Generate Pharmacophore
    phor_file = ""
    if anchor and mode == 'complex':
        phor_file = result['a']
        if not is_valid_file(phor_file):
            result['status'] = 1.2
            print(f"[W] Invalid pharmacophore file with anchor: `{phor_file}`.")
            return result

    else:
        prot_flag = "" if mode == 'ligand' else f" -p {result['p']} "
        phor_file = os.path.join(cache_path, f"{result['id']}_pharmacophore.phore")
        phor_log = os.path.join(cache_path, f"{result['id']}_pharmacophore.log")
        phar_cmd = CMD_TEMP['ancphore']['phor_gen'].format(ancphore_path, phor_file, result['l'], prot_flag, phor_log)
        status = os.system(phar_cmd)
        if status != 0 and not is_valid_file(phor_file):
            result['status'] = 1.2
            print(f"[W] Failed to generate pharmacophore for {result['id']}.")
            return result
        # random select features for ligand-based pharmacophore and generate exclusion spheres.
        if mode == 'ligand' and random:
            try:
                random_conf = {'up_num':11, 'low_num':10, 'sample_num':1, 'max_rounds':50}
                ex_conf = {'remove_hs':True, 'ex_dis':2.0, 'num_ex':2, 'mode':'shell', 'rounds':100}
                mol = read_molecule(result['l'])
                if mol is None:
                    mol = read_molecule(result['l'].replace('.sdf', '.mol2'))
                phore = parse_phore(phor_file)[0]
                random_phore = generate_random_phore(mol, phore, random_conf, ex_conf, use_ex=use_ex, mol_sdf=result['l'])[0]
                random_phor_file = os.path.join(cache_path, f"{result['id']}_random_pharmacophore.phore")
                phor_file = write_phore_to_file(random_phore, random_phor_file)

            except Exception as e:
                result['status'] = 1.3
                print(f"[W] Failed to sample random pharmacophore for {result['id']}. {e}")

            if result['status'] != 0:
                return result

    ## Start to align
    phar_std = time.time()
    dbfiles = result[result['toSearch']]
    result['aligned_file'] = []
    result['aligand_score'] = []
    result['batch_time'] = []
    align_cache_path = os.path.join(cache_path, "align_cache")
    if not os.path.exists(align_cache_path):
        os.makedirs(align_cache_path, exist_ok=True)
        # print("align_cache_path:", align_cache_path)

    for idx, dbfile in enumerate(dbfiles):
        batch_std = time.time()
        aligned_file = os.path.join(align_cache_path, f"{result['id']}_aligned_{idx}.sdf")
        score_file = os.path.join(align_cache_path, f"{result['id']}_aligned_{idx}.score")
        aligned_log = os.path.join(align_cache_path, f"{result['id']}_aligned_{idx}.log")
        # aligned_phor = os.path.join(cache_path, f'{_id}_{mode}_aligned.phore'))
        align_cmd = CMD_TEMP['ancphore']['align'].format(ancphore_path, phor_file, dbfile, 
                                                         aligned_file, score_file, aligned_log)
        # print("align_cmd:", align_cmd)
        status = os.system(align_cmd)
        if status != 0 or not is_valid_file(score_file):
            result['status'] = 2
            print(f"[W] Failed to align pose for {result['id']} in the `{idx}th` batch")
        else:
            result['aligned_file'].append(aligned_file)
            result['aligand_score'].append(score_file)
            result['batch_time'].append(time.time() - batch_std)
            print(f"[I] AncPhore: {result['id']} -> {idx+1}/{len(dbfiles)} batch processed.")

    result['align_time'] = time.time() - phar_std
    return result


def get_dataset(dataset):
    """
    Obtain the dataset for evaluations of pharmacophore tools
    """
    data = []
    data_path, test_list, ids = "", [], []
    if dataset == 'pdbbind':
        data_path = os.path.join(os.path.dirname(__file__), '../data/PDBBind/all/')
        test_list = os.path.join(os.path.dirname(__file__), '../data/splits/timesplit_test')
        ids = [pdbid.strip() for pdbid in open(test_list, 'r').readlines() if pdbid.strip() != '']
    elif dataset == 'posebusters':
        data_path = os.path.join(os.path.dirname(__file__), '../data/PoseBusters/posebusters_benchmark_set/')
        test_list = os.path.join(os.path.dirname(__file__), '../data/PoseBusters/posebusters_benchmark_set_ids.txt')
        ids = [pdbid.strip() for pdbid in open(test_list, 'r').readlines() if pdbid.strip() != '']
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    random_path = os.path.join(os.path.dirname(__file__), f'../experiments/baselines/prepared_datasets/{dataset}')

    for pdbid in ids:
        ligand_file = os.path.join(data_path, f'{pdbid}/{pdbid}_ligand.sdf')
        receptor_file = os.path.join(data_path, f'{pdbid}/{pdbid}_protein.pdb')
        rand_file = os.path.join(random_path, f'{pdbid}/{pdbid}_ligand.sdf')
        if is_valid_file(ligand_file) and is_valid_file(receptor_file):
            data.append({'l': ligand_file, 'p': receptor_file, 'rand': rand_file, 'id': pdbid})

    if len(data) == 0:
        raise ValueError(f'No data found for {dataset}')
    print("[I] Data preprocessing finished.")
    return data


def get_ifptarget(drug=None, conformation=True, num_conformers=40, overwrite=False):
    """
    Generate a list of target information for given drugs.
    Args:
    drug (str or list, optional): The name of the drug or a list of drug names. If None, all drugs in the directory are processed. Default is None.
    conformation (bool, optional): Whether to generate conformations for the drugs. Default is True.
    num_conformers (int, optional): The number of conformers to generate if conformation is True. Default is 40.
    overwrite (bool, optional): Whether to overwrite existing conformation files. Default is False.
    Returns:
    list: A list of dictionaries containing target information for each drug.
    Raises:
    AssertionError: If the drug file does not exist.
    """
    data = []
    exclude = ['Lumateperone', 'Oliceridine']
    drug_path = "/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/drugs"
    # ifp_file = "/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/IFPTarget.csv"
    ifp_file = "/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/IFPTarget_refine.csv"
    df_ifp = pd.read_csv(ifp_file)

    drug_list = os.listdir(drug_path) if drug is None else drug if isinstance(drug, list) else [drug]
    drug_list = [d for d in drug_list if d not in exclude]

    for d in drug_list:
        drug_file = os.path.join(drug_path, d, f"{d}.sdf")

        assert os.path.exists(drug_file), f"[E] Drug file does not exist: `{drug_file}`"
        drug_conf_file = ""
        conf_time = 0
        if conformation:
            drug_conf_file = os.path.join(drug_path, d, f"{d}_conf.sdf")
            drug_conf_log = os.path.join(drug_path, d, f"{d}_conf.log")
            if overwrite or not is_valid_file(drug_conf_file):
                conf_std = time.time()
                c_cmd = "time " + CMD_TEMP['conf_gen'].format(drug_file, drug_conf_file, num_conformers, drug_conf_log)
                status = os.system(c_cmd)
                if status != 0 or not is_valid_file(drug_conf_file):
                    print(f'[W] Conformation generation failed for {d}')
                else:
                    conf_time = time.time() - conf_std


        for rec in df_ifp[['phore_file', 'targetShortName', 'pdbid', 'protein_file']].to_dict('records'):
            if is_valid_file(rec['phore_file']):
                
                _data =  {
                        'id': rec['pdbid'].lower(),
                        'l': drug_file, 'p': rec['protein_file'],
                        'target': rec['targetShortName'],
                        'a': rec['phore_file'],  'drug': d,
                    }
                if drug_conf_file != "" and is_valid_file(drug_conf_file):
                    _data['dm'] = drug_conf_file
                    _data['conf_time'] = conf_time
                data.append(copy.deepcopy(_data))

    print(f"[I] Data preprocessing finished. {len(data)} samples to process for: {drug_list}")
    return data


def get_dude(conformation=True, n_conf=40, overwrite=False, conf_per_file=6000):
    """
    Processes the DUD-E dataset and generates conformations for targets.
    Args:
        conformation (bool): Whether to generate conformations for the targets. Default is True.
        n_conf (int): Number of conformations to generate. Default is 40.
        overwrite (bool): Whether to overwrite existing files. Default is False.
        conf_per_file (int): Number of conformations per file. Default is 6000.
    Returns:
        list: A list of dictionaries containing processed data for each target.
    The function performs the following steps:
    1. Reads the list of targets from the specified file.
    2. For each target, checks if the necessary files exist.
    3. If the files exist, processes the target data and generates conformations if required.
    4. Saves the processed data to a JSON file.
    5. Returns a list of dictionaries containing the processed data for each target.
    Note:
        - The function assumes the existence of specific directory structures and files.
        - The function uses external commands and tools for conformation generation.
        - The function may take a significant amount of time to run depending on the number of targets and conformations.
    """
    data_path = os.path.join(os.path.dirname(__file__), '../data/DUD_E/')
    target_path = os.path.join(os.path.dirname(__file__), '../data/DUD_E/targets/')
    crystal_path = os.path.join(os.path.dirname(__file__), '../data/DUD_E/crystal_selection/')
    target_list = os.path.join(os.path.dirname(__file__), '../data/DUD_E/process/selected_targets.list')
    target_list = [line.strip() for line in open(target_list, 'r').readlines() if line.strip() != '']
    dataset = []
    for t in target_list:
        print(f"[I] Processing the target `{t}`")
        t_path = os.path.abspath(os.path.join(target_path, t.lower()))
        c_path = os.path.abspath(os.path.join(crystal_path, t.lower()))
        if os.path.exists(t_path):
            data = {}
            data_json = os.path.join(t_path, 'data.json')
            if not is_valid_file(data_json) or overwrite:
                t_all = os.path.join(t_path, 'all_final_single.sdf.gz')
                c_protein = os.path.join(c_path, 'protein.pdb')
                c_ligand = os.path.join(c_path, 'ligand.sdf')
                c_anchor = os.path.join(c_path, 'anchor.phore')
                
                if is_valid_file(t_all) and is_valid_file(c_protein) and is_valid_file(c_ligand) and is_valid_file(c_anchor):
                    data.update({'id': t, 'p': c_protein, 'l': c_ligand, 'a': c_anchor, 'd': t_all})
                    if conformation:
                        conf_std = time.time()
                        c_all = os.path.join(t_path, 'all_final_conformation.sdf')
                        c_log = os.path.join(t_path, 'all_final_conformation.log')
                        if overwrite or not is_valid_file(c_all):
                            c_cmd = "time " + CMD_TEMP['conf_gen'].format(t_all, c_all, n_conf, c_log)
                            status = os.system(c_cmd)
                            if status != 0 or not is_valid_file(c_all):
                                print(f'Conformation generation failed for {t}')
                            else:
                                data['dm'] = c_all
                                split_path = os.path.join(t_path, 'splits')
                                if os.path.exists(split_path) and os.listdir(split_path) != 0:
                                    shutil.rmtree(split_path)
                                results = split_sdf_file(c_all, split_path, conf_per_file=conf_per_file)
                                data['conf_time'] = time.time() - conf_std
                                data['db'] = results

                json.dump(data, open(data_json, 'w'), indent=4)
            else:
                data = json.load(open(data_json, 'r'))
            dataset.append(copy.deepcopy(data))
    return dataset


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """
    Reads a molecule from a file and optionally sanitizes, calculates charges, and removes hydrogens.

    Args:
    molecule_file (str): Path to the molecule file. Supported formats are .mol2, .mol, .sdf, .pdbqt, and .pdb.
    sanitize (bool): If True, sanitizes the molecule. Default is False.
    calc_charges (bool): If True, calculates Gasteiger charges for the molecule. Default is False.
    remove_hs (bool): If True, removes hydrogens from the molecule. Default is False.

    Returns:
    mol (rdkit.Chem.rdchem.Mol or None): The RDKit molecule object if successful, None otherwise.

    Raises:
    ValueError: If the file format is not supported.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False) # type: ignore
    elif molecule_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False) # type: ignore
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False) # type: ignore
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False) # type: ignore
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False) # type: ignore
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol) # type: ignore

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol) # type: ignore
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize) # type: ignore
    except Exception as e:
        print(e)
        print("RDKit was unable to read the molecule.")
        return None

    return mol


def generate_random_phore(ligand, phore, 
                          random_conf={'up_num':10, 'low_num':4, 'sample_num':5, 'max_rounds':50}, 
                          ex_conf={'remove_hs': True, 'ex_dis': 2.0, 'num_ex': 2, 'mode': 'shell', 'rounds': 100}, 
                          use_ex=True, mol_sdf=""):
    """
    Generate random pharmacophores based on the given ligand and pharmacophore.

    Parameters:
    ligand (object): The ligand molecule.
    phore (object): The original pharmacophore.
    random_conf (dict, optional): Configuration for generating random pharmacophores. 
        Defaults to {'up_num':10, 'low_num':4, 'sample_num':5, 'max_rounds':50}.
    ex_conf (dict, optional): Configuration for generating exclusion volumes. 
        Defaults to {'remove_hs': True, 'ex_dis': 2.0, 'num_ex': 2, 'mode': 'shell', 'rounds': 100}.
    use_ex (bool, optional): Whether to use exclusion volumes. Defaults to True.
    mol_sdf (str, optional): Path to the molecule file in SDF format. Defaults to "".

    Returns:
    list: A list of generated random pharmacophores.

    Raises:
    Exception: If the generation of random pharmacophores fails.
    """
    random_phores = []
    # print(f'random_conf: {random_conf}')
    # print(f'ex_conf: {ex_conf}')
    try:
        random_phores = extract_random_phore_from_origin(phore, **random_conf)
        if use_ex:
            random_phores = [generate_random_exclusion_volume(p, ligand, **ex_conf) for p in random_phores]
    except Exception as e:
        print(f"{phore.id} failed to generate random phore.", e)
        mol = read_molecule(mol_sdf.replace('.sdf', '.mol2'))
        random_phores = generate_random_phore(mol, phore, random_conf, ex_conf, use_ex)
        # raise(e)
    return random_phores


def generate_conformation(cache_path, result, num_conformers=40, split=False, conf_per_file=6000):
    """
    Generates molecular conformations and optionally splits the resulting file.

    Args:
    cache_path (str): The path to the cache directory where intermediate files will be stored.
    result (dict): A dictionary containing molecular data and metadata.
    num_conformers (int, optional): The number of conformations to generate. Default is 40.
    split (bool, optional): Whether to split the resulting conformation file into smaller files. Default is False.
    conf_per_file (int, optional): The maximum number of conformations per split file. Default is 6000.

    Returns:
    None: The function modifies the `result` dictionary in place, adding keys for generated conformations and metadata.
    """
    _id = result['id']
    if 'd' not in result:
        result['d'] = result['l']

    if 'dm' not in result:
        init_pose = result.get('rand', '')
        if init_pose == '':
            init_pose = os.path.join(cache_path, f"{_id}_random.sdf")
            mol = read_molecule(result['d'])
            rand_mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(rand_mol)
            Chem.MolToMolFile(rand_mol, init_pose)

        conf_std = time.time()
        conf_file = os.path.join(cache_path, f"{_id}_conformations.sdf")
        conf_log = os.path.join(cache_path, f"{_id}_conformations.log")
        conf_cmd = CMD_TEMP['conf_gen'].format(init_pose, conf_file, num_conformers, conf_log)
        # print(f"Command: {conf_cmd}")
        status = os.system(conf_cmd)

        if status != 0 or not is_valid_file(conf_file):
            result['status'] = 1
            print(f'[W] Failed to generate conformations for `{_id}`')
        else:
            result['dm'] = conf_file
            result['conf_time'] = time.time() - conf_std

    if split and 'db' not in result:
        counts = len([line for line in open(result['dm'], 'r').readlines() if line.strip() == "$$$$"])
        if counts > conf_per_file:
            split_path = os.path.join(cache_path, 'splits')
            if os.path.exists(split_path) and len(os.listdir(split_path)) > 0:
                shutil.rmtree(split_path)
            result['db'] = split_sdf_file(result['dm'], split_path, conf_per_file)
        else:
            split = False

    if 'dm' in result:
        result['_dm'] = [result['dm']]
        result['toSearch'] = '_dm' if not split else 'db'


def split_sdf_file(sdf_file, out_dir, conf_per_file=6000):
    """
    Splits a large SDF file into smaller files with a specified number of conformers per file.

    Args:
        sdf_file (str): Path to the input SDF file. Can be a regular or gzipped SDF file.
        out_dir (str): Directory where the output SDF files will be saved.
        conf_per_file (int, optional): Number of conformers per output file. Default is 6000.

    Returns:
        list: A list of paths to the generated smaller SDF files.

    Raises:
        OSError: If there is an error creating the output directory or writing to the files.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sdf_file = os.path.abspath(sdf_file)
    sdf_name = os.path.basename(sdf_file).strip(".sdf").strip(".sdf.gz")
    out_dir = os.path.abspath(out_dir)
    out_file = os.path.join(out_dir, sdf_name)
    result = []

    f = gzip.open(sdf_file, 'rt') if sdf_file.endswith('.gz') else open(sdf_file, 'r')
    ids = []
    current_lines = []
    line = f.readline()
    curr_id = ""
    wf = None

    while line:
        if curr_id == "":
            curr_id = line.strip()

        if curr_id != "":
            current_lines.append(line)
            if line.strip() == "$$$$":
                if len(ids) % conf_per_file == 0:
                    if wf is not None: wf.close()
                    new_file = f"{out_file}_{len(ids)//conf_per_file}.sdf"
                    result.append(new_file)
                    wf = open(new_file, 'a')
                wf.write("".join(current_lines))
                ids.append(curr_id)
                curr_id = ""
                current_lines = []
        line = f.readline()
    f.close()
    return result


def copyTo(src, dst):
    dst.append(copy.deepcopy(src))


def is_valid_file(f):
    return os.path.exists(f) and os.path.isfile(f) and os.path.getsize(f) != 0


def check_phore_num(phore_file):
    with open(phore_file, 'r') as f:
        lines = [line for line in f.readlines() if line.strip() != ""]
        num = len(lines)
        return num - 2 if num > 0 else 0


def main():
    args = parse_args()
    # out_dir = os.path.join(args.out_dir, f"{args.task}/{args.dataset}/{args.mode}/{args.baseline}")
    if args.task in ['screen', 'align', 'fishing']:
        if args.task in ['screen', 'align']:
            args.out_dir = os.path.join(args.out_dir, f"{args.task}/{args.dataset}/{args.mode}/{args.baseline}") 

        results = evaluate(args)
        # results = virtual_screening(args.mode, args.baseline, args.dataset, out_dir)

    elif args.task == 'analyze':
        ...
    else:
        raise NotImplementedError(f'The specified task `{args.task}` is not implemented yet.')


if __name__ == "__main__":
    ## Virtual Screening Test Set
    # test()
    
    ## Main Function
    st = time.time()
    # torch.multiprocessing.set_start_method("spawn")
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Current Hostname: $(hostname)")
    print(f'Current PID: {os.getpid()}')
    print(f"Current Command: {' '.join(sys.argv)}")
    main()

    end = time.time()
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Job Finished! {end-st:.3f} seconds cost.")

