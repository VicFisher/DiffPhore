import json
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import time
import pandas as pd


def prepare_datasets(source_path, target_path, filelist):
    """
    Preprocess of ligand and receptor for molecular docking
    Args:
        source_path: The source directory of the input files.
        target_path: The destiny directory to store the processed files.
        filelist: The files recording the sample ids for preprocess.
    """
    pdblist = [pdb.strip() for pdb in open(filelist).readlines()]
    for pdb in pdblist:
        try:
            pdb_dir = os.path.join(source_path, pdb)
            ligand_file = os.path.join(pdb_dir, f'{pdb}_ligand.sdf')
            protein_file = os.path.join(pdb_dir, f'{pdb}_protein.pdb')
            process_pdb(pdb, protein_file, ligand_file, target_path)
        except Exception as e:
            print(f"Error processing PDB {pdb}: {e}")


def process_pdb(name, protein_file, ligand_file, target_path, 
                prepare_lig_path="/home/worker/software/ADFR/bin/prepare_ligand",
                prepare_rec_path="/home/worker/software/ADFR/bin/prepare_receptor"):
    """
    Process the input protein file and ligand file for the docking baselines.
    Args:
        name: The id of current input sample.
        protein_file: The protein file in PDB format.
        ligand_file: The ligand file in SDF format.
        target_path: The target path to store the result.
        prepare_lig_path: The script from ADFR to prepare ligand.
        prepare_rec_path: The script from ADFR to prepare receptor.
    """
    try:
        target_dir = os.path.join(target_path, name)
        os.makedirs(target_dir, exist_ok=True)

        vina_config = os.path.join(target_dir, f'{name}_vina.config')
        gs_config = os.path.join(target_dir, f'{name}_gs.config')
        uni_config = os.path.join(target_dir, f'{name}_uni.config')

        if not all([os.path.exists(x) for x in [vina_config, gs_config, uni_config]]):
            vina_dict = {}
            gs_dict = {}
            uni_dict = {}

            gs_dict['autobox_ligand'] = ligand_file
            gs_dict['autobox_add'] = '4'
            
            ## 1. prepare the ligand random conformation
            lig_random_pdb = os.path.join(target_dir, f'{name}_ligand.pdb')
            lig_random_pdbqt = os.path.join(target_dir, f'{name}_ligand.pdbqt')
            lig_random_sdf_uni = os.path.join(target_dir, f'{name}_ligand_uni.sdf')
            lig_random_sdf = os.path.join(target_dir, f'{name}_ligand.sdf')
            ligand_mol = None

            try:
                ligand_mol = Chem.SDMolSupplier(ligand_file, removeHs=False)[0]
                center = ligand_mol.GetConformers()[0].GetPositions().mean(axis=0)
                center_dict = {'center_x': f"{center[0]:.3f}", 'center_y': f"{center[1]:.3f}", 'center_z': f"{center[2]:.3f}"}
                vina_dict.update(center_dict)
                uni_dict.update(center_dict)
                rand_mol = Chem.AddHs(ligand_mol)
                AllChem.EmbedMolecule(rand_mol)
                Chem.MolToPDBFile(rand_mol, lig_random_pdb)
                Chem.MolToMolFile(rand_mol, lig_random_sdf)
                Chem.MolToMolFile(rand_mol, lig_random_sdf_uni)

            except Exception as e:
                print(f"[W] Failed to process ligand file {ligand_file}. Trying to reload molecule from mol2 file and retry. \nError: {e}")

                ligand_mol = Chem.MolFromMol2File(ligand_file.replace('.sdf', '.mol2'))
                center = ligand_mol.GetConformers()[0].GetPositions().mean(axis=0)
                center_dict = {'center_x': f"{center[0]:.3f}", 'center_y': f"{center[1]:.3f}", 'center_z': f"{center[2]:.3f}"}
                vina_dict.update(center_dict)
                uni_dict.update(center_dict)
                rand_mol = Chem.AddHs(ligand_mol)
                AllChem.EmbedMolecule(rand_mol)
                Chem.MolToPDBFile(rand_mol, lig_random_pdb)
                Chem.MolToMolFile(rand_mol, lig_random_sdf)
                Chem.MolToMolFile(rand_mol, lig_random_sdf_uni)

            ## 2. prepare the ligand file
            # 2.1 vina
            com_prepare_vina = f"cd {target_dir} && {prepare_lig_path} -l {lig_random_pdb} -o {lig_random_pdbqt} > /dev/null && cd - > /dev/null"
            status = os.system(com_prepare_vina)
            vina_dict['ligand'] = lig_random_pdbqt
            gs_dict['ligand'] = lig_random_sdf

            # 2.2 unidock
            com_prepare_uni = f"unidocktools ligandprep -l {lig_random_sdf_uni} -sd {target_dir} > /dev/null"
            status = os.system(com_prepare_uni)
            uni_dict['ligands'] = lig_random_sdf_uni

            ## 3. prepare the protein file
            protein_file_clean = os.path.join(target_dir, f'{name}_protein_clean.pdb')
            protein_pdbqt = os.path.join(target_dir, f'{name}_protein.pdbqt')
            protein_pdbqt_uni = os.path.join(target_dir, f'{name}_protein_uni.pdbqt')

            # 3.1 vina-smina-gnina
            com_prepare_protein = f"grep -v 'HETATM' {protein_file} > {protein_file_clean} && {prepare_rec_path} -r {protein_file_clean} -o {protein_pdbqt} -A 'hydrogens' > /dev/null"
            status = os.system(com_prepare_protein)
            vina_dict['receptor'] = protein_pdbqt
            gs_dict['receptor'] = protein_pdbqt

            # 3.2 unidock
            com_prepare_protein_uni = f"unidocktools proteinprep -r {protein_file} -o {protein_pdbqt_uni} > /dev/null"
            status = os.system(com_prepare_protein_uni)
            uni_dict['receptor'] = protein_pdbqt_uni

            # 4. write the config file
            json.dump(vina_dict, open(vina_config, 'w'), indent=4)
            json.dump(gs_dict, open(gs_config, 'w'), indent=4)
            json.dump(uni_dict, open(uni_config, 'w'), indent=4)

    except Exception as e:
        print(f"[E] Failed to process the name `{name}`. Error: {e}")


def prepare_ligand_input(target, target_dir, outpath, nworkers=1):
    """
    Prepare the ligand from DUD-E dataset
    args:
        target: Specific target from DUD-E dataset.
        target_dir: The directory of each target.
        outpath: The output path to store the processed files
        nworkers: The number of workers to process the input file. Default as 1.
    """
    active_smi = os.path.join(target_dir, target, 'actives_final.ism')
    decoy_smi = os.path.join(target_dir, target, 'decoys_final.ism')
    outpath = os.path.join(outpath, target)
    prepared_ligand_path = os.path.join(outpath, 'inputs')
    os.makedirs(prepared_ligand_path, exist_ok=True)
    os.makedirs(outpath, exist_ok=True)

    actives = [smi.strip() for smi in open(active_smi, 'r').readlines()]
    actives = [{'smiles': smi.split()[0], 'name': smi.split()[-1], 'label': 1} for smi in actives if smi != '']
    decoys = [smi.strip() for smi in open(decoy_smi, 'r').readlines()]
    decoys = [{'smiles': smi.split()[0], 'name': smi.split()[-1], 'label': 0} for smi in decoys if smi != '']
    ligands = actives + decoys
    json.dump(ligands, open(os.path.join(outpath, f'{target}_ligand.json'), 'w'), indent=4)
    print(f'[I] `{target}`: {len(ligands)} ({len(actives)} actives & {len(decoys)} decoys) ligands to be prepared.')
    if nworkers > 1:
        df_ligand = pd.DataFrame(ligands)
        df_ligand.parallel_apply(lambda l: prepare_drug_file(l['name'], l['smiles'], prepared_ligand_path), axis=1)

    else:
        for l in ligands:
            prepare_drug_file(l['name'], l['smiles'], prepared_ligand_path)
    print(f"[I] `{target}`: All ligands are prepared.")


def prepare_vs_dataset(complex_dir, target_path, outpath, nworkers=36):
    """
    Prepare the virtual screening dataset.
    """
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=nworkers)
    for target in os.listdir(complex_dir):
        try:
            target = target.lower()
            protein_file = os.path.join(complex_dir, target, f'protein.pdb')
            ligand_file = os.path.join(complex_dir, target, f'ligand.sdf')
            process_pdb(target, protein_file, ligand_file, outpath)
            prepare_ligand_input(target, target_path, outpath, nworkers=nworkers)

        except Exception as e:
            print(f"[E] Failed to process `{target}`: {e}")


def prepare_drug_file(drug="4OH_Tamoxifen", smiles="", 
    drug_path="/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/drugs"):
    """
    Prepare the drug file for target fishing
    """
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


def pose_prediction_prepare():
    """
    Prepare for pose prediction
    """
    # 1. prepare the pdbbind dataset
    print(f"[I] Starting to prepare the PDBBind dataset...")
    pdbbind_dir = '/home/worker/users/YJL/DiffPhore/data/PDBBind/all'
    pdbbind_out = '/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets/pdbbind'
    pdbbind_list = '/home/worker/users/YJL/DiffPhore/data/splits/timesplit_test'
    prepare_datasets(pdbbind_dir, pdbbind_out, pdbbind_list)
    print(f"[I] PDBBind dataset preparation is done.")

    # 2. prepare the posebusters dataset
    print(f"[I] Starting to prepare the PoseBusters dataset...")
    posebusters_dir = '/home/worker/users/YJL/DiffPhore/data/PoseBusters/all'
    posebusters_out = '/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets/posebusters'
    posebusters_list = '/home/worker/users/YJL/DiffPhore/data/splits/posebusters_test_all'
    prepare_datasets(posebusters_dir, posebusters_out, posebusters_list)
    print(f"[I] PoseBusters dataset preparation is done.")


def target_fishing_prepare():
    """
    Prepare for target fishing
    """
    import pandas as pd
    from pandarallel import pandarallel
    nworkers = 30
    pandarallel.initialize(nb_workers=nworkers)
    df_ifp = pd.read_csv('/home/worker/users/YJL/DiffPhore/experiments/diffphore_inference/target_fishing/IFPTarget.csv')
    df_ifp['pdbid'] = df_ifp['pdbid'].map(lambda x: x.lower())
    # df_ifp['source_path'] = df_ifp['protein_file'].map(lambda x: os.path.dirname(os.path.dirname(x)))
    out_path = "/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets/target_fishing"

    df_ifp.parallel_apply(lambda x: process_pdb(x['pdbid'], x['protein_file'], x['ligand_file'], out_path), axis=1)


def virtual_screening_prepare():
    """
    Prepare for virtual screening.
    """
    complex_dir = '/home/worker/users/YJL/DiffPhore/data/DUD_E/crystal_selection/'
    target_path = '/home/worker/users/YJL/DiffPhore/data/DUD_E/targets/'
    outpath = '/home/worker/users/YJL/DiffPhore/experiments/baselines/prepared_datasets/virtual_screening/'
    prepare_vs_dataset(complex_dir, target_path, outpath)

if __name__ == '__main__':

    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current PID: {os.getpid()}")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Hostname: $(hostname)")
    ## prepare the datasets for pose prediction
    # pose_prediction_prepare()

    # prepare the datasets for target fishing
    # target_fishing_prepare()

    # prepare the datasets for virtual screening
    virtual_screening_prepare()


    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] All jobs finished.")
