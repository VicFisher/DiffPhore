import os
import json
import argparse
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from pandarallel import pandarallel
# from baselines.prepare_datasets import select_ligand_file, prepare_drug_file

BASE = "/home/worker/users/YJL/DiffPhore/programs/baselines"
tools = {'smina': f'{BASE}/smina', 'vina': f'{BASE}/vina', 
         'gnina': f'{BASE}/gnina', 'unidock': 'unidocktools unidock_pipeline'}
config_abbr = {'smina': 'gs', 'gnina': 'gs', 'vina': 'vina', 'unidock': 'uni'}

def docking_run(baseline, dataset, size=20, cpu=8, num_modes=10, num_workers=1,
        prepared_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets", 
        output_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/output"):
    output_dir = os.path.join(output_path, f"align/{dataset}/complex/{baseline}")
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(prepared_path, f"{dataset}")

    size_flag = "" if baseline not in ['vina', 'unidock'] else f"--size_x {size} --size_y {size} --size_z {size}"
    cpu_flag = "" if baseline not in ['vina', 'smina', 'gnina'] else f"--cpu {cpu}"
    num_flag = f"--num_modes {num_modes}"

    if num_workers > 1:
        pandarallel.initialize(nb_workers=num_workers)
        df_pdbids = pd.DataFrame({'pdbid': os.listdir(dataset_dir)})
        df_pdbids.parallel_apply(lambda x: run_docking(x['pdbid'], baseline, dataset_dir, output_dir, size_flag, cpu_flag, num_flag), axis=1)

    else:
        for pdb in os.listdir(dataset_dir):
            run_docking(pdb, baseline, dataset_dir, output_dir)


def target_fishing_run(baseline, drug, smiles="",
        dataset='target_fishing', size=20, cpu=8, num_modes=10, num_workers=30,
        prepared_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets", 
        drug_path="/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/drugs/",
        output_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/output"):

    output_dir = os.path.join(output_path, f"target_fishing/{baseline}/{drug}")
    ligand_file = select_ligand_file(drug, baseline, smiles, drug_path)
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(prepared_path, f"{dataset}")

    size_flag = "" if baseline not in ['vina', 'unidock'] else f"--size_x {size} --size_y {size} --size_z {size}"
    cpu_flag = "" if baseline not in ['vina', 'smina', 'gnina'] else f"--cpu {cpu}"
    num_flag = f"--num_modes {num_modes}"

    if num_workers > 1:
        pandarallel.initialize(nb_workers=num_workers)
        df_pdbids = pd.DataFrame({'pdbid': os.listdir(dataset_dir)})
        df_pdbids.parallel_apply(lambda x: run_docking(x['pdbid'], baseline, dataset_dir, 
                                                       output_dir, ligand_file=ligand_file, 
                                                       size_flag=size_flag, cpu_flag=cpu_flag, 
                                                       num_flag=num_flag), axis=1)
    else:
        for pdb in os.listdir(dataset_dir):
            run_docking(pdb, baseline, dataset_dir, output_dir)


def virtual_screening_run(baseline, target, 
        dataset='virtual_screening', size=20, cpu=8, num_modes=10, num_workers=30,
        prepared_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets", 
        output_path="/home/worker/users/YJL/DiffPhore/experiments/baselines/output"):
    import pandas as pd
    target = target.lower()

    if num_workers > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=num_workers)

    output_dir = os.path.join(output_path, f"virtual_screening/{baseline}/{target}")
    dataset_dir = os.path.join(prepared_path, f"{dataset}")


    ligand_path = os.path.join(prepared_path, f"{dataset}/{target}/inputs")
    print(f"[I] `{target}`-`{baseline}`: {len(os.listdir(ligand_path))} ligands found.")
    ligand_json = os.path.join(prepared_path, f"{dataset}/{target}/{target}_ligand.json")
    ligand_rec = json.load(open(ligand_json, 'r'))

    os.makedirs(output_dir, exist_ok=True)

    size_flag = "" if baseline not in ['vina', 'unidock'] else f"--size_x {size} --size_y {size} --size_z {size}"
    cpu_flag = "" if baseline not in ['vina', 'smina', 'gnina'] else f"--cpu {cpu}"
    num_flag = f"--num_modes {num_modes}"

    if num_workers > 1:
        import pandas as pd
        df_recs = pd.DataFrame(ligand_rec)
        df_recs['ligand_file'] = df_recs.apply(lambda x: select_ligand_file(x['name'], baseline, x['smiles'], prepared=True, drug_path=ligand_path), axis=1)
        df_recs = df_recs[df_recs['ligand_file'] != ""]
        df_recs.parallel_apply(lambda x: run_docking_vs(target, x['name'], baseline, dataset_dir, 
                                                        output_dir, ligand_file=x['ligand_file'], 
                                                        size_flag=size_flag, cpu_flag=cpu_flag, 
                                                        num_flag=num_flag), axis=1)
    else:
        for rec in ligand_rec:
            rec['ligand_file'] = select_ligand_file(rec['name'], baseline, rec['smiles'], ligand_path)
            run_docking_vs(target, rec['name'], baseline, dataset_dir, 
                           output_dir, ligand_file=rec['ligand_file'], 
                           size_flag=size_flag, cpu_flag=cpu_flag, 
                           num_flag=num_flag)


def run_docking(pdb, baseline, dataset_dir, output_dir, ligand_file="", size_flag="", cpu_flag="", num_flag=""):
    try:
        tool = tools[baseline]
        pdb_dir = os.path.join(dataset_dir, pdb)
        output_pdb = os.path.join(output_dir, pdb)
        os.makedirs(output_pdb, exist_ok=True)
        status_file = os.path.join(output_pdb, f"{pdb}.status")
        status = "1" 
        if os.path.exists(status_file):
            with open(status_file, 'r') as fstat:
                status = fstat.readline().strip()

        if status != "0":
            std_time = time.time()
            config_file = os.path.join(pdb_dir, f"{pdb}_{config_abbr[baseline]}.config")
            config_flag = json_to_config_flag(config_file, ligand_file)

            out_flag = ""
            if baseline in ['smina', 'gnina']:
                out_flag = f"--out " + os.path.join(output_pdb, f"{pdb}.sdf")

            elif baseline == 'vina':
                out_flag = f"--out " + os.path.join(output_pdb, f"{pdb}.pdbqt")

            elif baseline == 'unidock':
                out_flag = f"-sd {output_pdb}"

            log_file = os.path.join(output_pdb, f"{pdb}.log")
            log_flag = f" > {log_file} 2>&1"

            commands = f"{tool} {config_flag} {size_flag} {num_flag} {cpu_flag} {out_flag} {log_flag}"
            status = os.system(commands)

            if status != 0:
                print(f"[E] {baseline} failed for `{pdb}`")

            else:
                print(f"[I] {baseline} finished for `{pdb}`")

            with open(status_file, 'w') as f:
                f.write(str(status)+'\n')
                f.write(str(time.time() - std_time)+'\n')

    except Exception as e:
        print(f"[E] {baseline} failed for `{pdb}`. {e}")


def run_docking_vs(target, name, baseline, dataset_dir, output_dir, ligand_file="", size_flag="", cpu_flag="", num_flag=""):
    try:
        tool = tools[baseline]
        target_dir = os.path.join(dataset_dir, target)
        output_path = os.path.join(output_dir, name)
        os.makedirs(output_path, exist_ok=True)
        status_file = os.path.join(output_path, f"{name}.status")
        status = "1" 
        if os.path.exists(status_file):
            with open(status_file, 'r') as fstat:
                status = fstat.readline().strip()

        if status != "0":
            std_time = time.time()
            config_file = os.path.join(target_dir, f"{target}_{config_abbr[baseline]}.config")
            config_flag = json_to_config_flag(config_file, ligand_file)

            out_flag = ""
            if baseline in ['smina', 'gnina']:
                out_flag = f"--out " + os.path.join(output_path, f"{name}.sdf")

            elif baseline == 'vina':
                out_flag = f"--out " + os.path.join(output_path, f"{name}.pdbqt")

            elif baseline == 'unidock':
                out_flag = f"-sd {output_path}"

            log_file = os.path.join(output_path, f"{name}.log")
            log_flag = f" > {log_file} 2>&1"

            status = docking(tool, config_flag, size_flag, num_flag, cpu_flag, out_flag, log_flag)

            if status != 0:
                print(f"[E] {baseline} failed for {target}-{name}")

            else:
                print(f"[I] {baseline} finished for {target}-{name}")

            with open(status_file, 'w') as f:
                f.write(str(status)+'\n')
                f.write(str(time.time() - std_time)+'\n')

    except Exception as e:
        print(f"[E] {baseline} failed for `{pdb}`. {e}")


def docking(tool, config_flag, size_flag, num_flag, cpu_flag, out_flag, log_flag):
    commands = f"{tool} {config_flag} {size_flag} {num_flag} {cpu_flag} {out_flag} {log_flag}"
    status = os.system(commands)
    return status


def json_to_config_flag(json_file, ligand_file=""):
    config = json.load(open(json_file, 'r'))
    if ligand_file != "":
        if 'ligand' in config:
            config['ligand'] = ligand_file
        elif 'ligands' in config:
            config['ligands'] = ligand_file
    config_flag = ' '.join([f'--{key} {value}' for key, value in config.items()])
    return config_flag


def select_ligand_file(drug, baseline, smiles="", prepared=False,
                       drug_path="/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/drugs/"):
    try:
        if not prepared:
            prepare_drug_file(drug, smiles, drug_path)
        if baseline == 'vina':
            ligand_file = os.path.join(drug_path, f"{drug}/{drug}.pdbqt")
        elif baseline == 'unidock':
            ligand_file = os.path.join(drug_path, f"{drug}/{drug}_uni.sdf")
        elif baseline in ['smina', 'gnina']:
            ligand_file = os.path.join(drug_path, f"{drug}/{drug}.sdf")
        else:
            ligand_file = ""
        # assert os.path.exists(ligand_file), f"[E] {ligand_file} does not exist."

    except Exception as e:
        print(f"[E] {drug} {baseline} failed. {e}")
        ligand_file = ""

    if not os.path.exists(ligand_file):
        ligand_file = ""

    return ligand_file


def prepare_drug_file(drug="4OH_Tamoxifen", smiles="", 
    drug_path="/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/drugs"):
    try:
        target_dir = f"{drug_path}/{drug}"
        os.makedirs(target_dir, exist_ok=True)
        drug_file = f"{target_dir}/{drug}.sdf"
        drug_uni_file = f"{target_dir}/{drug}_uni.sdf"
        lig_random_pdb = f"{target_dir}/{drug}.pdb"
        lig_random_pdbqt = f"{target_dir}/{drug}.pdbqt"
        prepare_lig_path="/home/worker/software/ADFR/bin/prepare_ligand"

        mol = None
        if smiles != "" and not os.path.exists(drug_file):
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            Chem.MolToMolFile(mol, drug_file)

        if not os.path.exists(drug_uni_file):
            os.system(f"cp {drug_file} {drug_uni_file}")
            com_prepare_uni = f"unidocktools ligandprep -l {drug_uni_file} -sd {target_dir} > /dev/null"
            os.system(com_prepare_uni)

        if mol is None:
            mol = Chem.SDMolSupplier(drug_file, removeHs=False)[0]

        if not os.path.exists(lig_random_pdbqt):
            Chem.MolToPDBFile(mol, lig_random_pdb)
            com_prepare_vina = f"cd {target_dir} && {prepare_lig_path} -l {lig_random_pdb} -o {lig_random_pdbqt} > /dev/null && cd - > /dev/null"
            status = os.system(com_prepare_vina)
    except Exception as e:
        print(f"[E] Failed to prepare {drug}: {e}")


if __name__ == '__main__':

    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current PID: {os.getpid()}")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Hostname: $(hostname)")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pdbbind', choices=['pdbbind', 'posebusters', 'target_fishing', 'virtual_screening'])
    parser.add_argument('--baseline', type=str, default='vina', choices=['vina', 'smina', 'gnina', 'unidock'])
    parser.add_argument('--nworkers', type=int, default=1)
    parser.add_argument('--task', type=str, default='docking', choices=['docking', 'target_fishing', 'virtual_screening'])
    parser.add_argument('--drug', type=str, default='4OH-Tamoxifen')
    parser.add_argument('--smiles', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    args = parser.parse_args()
    args.smiles = args.smiles.strip(":").strip()
    args.drug = args.drug.strip(":").strip()
    args.target = args.target.strip(":").strip()

    if args.task == 'docking':
        docking_run(args.baseline, args.dataset, num_workers=args.nworkers)
    elif args.task == 'target_fishing':
        target_fishing_run(args.baseline, drug=args.drug, smiles=args.smiles, dataset=args.dataset, num_workers=args.nworkers)
    elif args.task == 'virtual_screening':
        virtual_screening_run(args.baseline, args.target,  args.dataset, num_workers=args.nworkers)
    else:
        raise NotImplementedError("Invalid task, please choose from [docking, target_fishing].")

    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Job done.")
