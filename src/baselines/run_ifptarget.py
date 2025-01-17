import json
import os, sys, math, time, argparse
import pandas as pd
from pandarallel import pandarallel

def split_index(index_file, batch_dir, batch_size=10):
    """
    Split the index in the index file into batches
    """
    os.makedirs(batch_dir, exist_ok=True)
    with open(index_file, 'r') as f:
        recs = f.readlines()
        headers = [r for r in recs if r.startswith('//')]
        contents = [r for r in recs if not r.startswith('//')]

    length = len(contents)
    batch = math.ceil(length / batch_size)
    filenames = []

    for i in range(batch):
        batch_file = os.path.join(batch_dir, f'batch_{i}.txt')
        if not os.path.exists(batch_file):
            with open(batch_file, 'w') as f:
                f.write(''.join(headers))
                f.write(''.join(contents[i*batch_size:(i+1)*batch_size]))
        filenames.append(batch_file)

    return filenames


def _run(drug_file, index_file, target_path, result_dir, work_dir, rec_file, 
         ifptarget_bin, vina_bin, status_file, log_file, overwrite=False):
    """
    To run the IFPTarget for target fishing.
    Args:
        drug_file (str): The drug file for the target fishing.
        index_file (str): The index file for the target fishing.
        target_path (str): The path to the target file.
        result_dir (str): The directory to store the results.
        work_dir (str): The working directory.
        rec_file (str): The receptor file.
        ifptarget_bin (str): The path to the IFPTarget binary.
        vina_bin (str): The path to the Vina binary.
        status_file (str): The file to store the status of the run.
        log_file (str): The file to store the log of the run.
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.

    Return:
        dict: A dictionary containing the cost (time taken), status (exit status), and rec_file (receptor file path).
    """

    # ifptarget_bin = os.path.abspath(ifptarget_bin)
    # vina_bin = os.path.abspath(vina_bin)
    # drug_file = os.path.abspath(drug_file)
    # index_file = os.path.abspath(index_file)
    # target_path = os.path.abspath(target_path)
    # result_dir = os.path.abspath(result_dir)
    # work_dir = os.path.abspath(work_dir)
    # status_file = os.path.abspath(status_file)
    # log_file = os.path.abspath(log_file)

    status = 110
    cost = 0
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            stat = f.readlines()
            if stat:
                try:
                    status = stat[0].strip()
                    cost = float(stat[1].strip())
                except:
                    print(f"[W] Failed to read status file for `{os.path.basename(drug_file)}`: `{os.path.basename(index_file)}`, re-running.")
                    status = 110

    if status != '0' or overwrite:
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.dirname(rec_file), exist_ok=True)
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        command = f"{ifptarget_bin} -l {drug_file} -t {target_path} -i {index_file}"
        command += f" -p {vina_bin} -s 0.0 -c 3 -w {work_dir} -r {result_dir} -rf {rec_file} > {log_file} 2>&1"
        st = time.time()
        print(f'[I] Command for `{os.path.basename(drug_file)}`: `{os.path.basename(index_file)}`: {command}')
        status = os.system(command)
        cost = time.time() - st

        with open(status_file, 'w') as f:
            f.write(f"{status}\n")
            f.write(f"{cost}\n")

        if status == 0:
            print(f"[I] Task finished for `{os.path.basename(drug_file)}`: `{os.path.basename(index_file)}`.")
        else:
            print(f"[E] Task failed for `{os.path.basename(drug_file)}`: `{os.path.basename(index_file)}`.")

    else:
        print(f"[I] Task finished for `{os.path.basename(drug_file)}`: `{os.path.basename(index_file)}` before, skipped.")
    return {'cost': cost, 'status': status, 'rec_file': rec_file}


def run(drug, drug_path, index_file, output_dir, target_path, ifptarget_bin, vina_bin, 
        overwrite=False, nworkers=1, batch=None):
    """
    Run the IFP target prediction pipeline for a given drug.
    Args:
        drug (str): Name of the drug.
        drug_path (str): Path to the directory containing the drug files.
        index_file (str): Path to the index file.
        output_dir (str): Directory where the output will be stored.
        target_path (str): Path to the target file.
        ifptarget_bin (str): Path to the IFP target binary.
        vina_bin (str): Path to the Vina binary.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        nworkers (int, optional): Number of parallel workers to use. Defaults to 1.
        batch (tuple, optional): Tuple specifying the batch range to process. Defaults to None.

    Return:
        list: List of records generated by the pipeline.

    """
    drug_file = os.path.join(drug_path, f'{drug}/{drug}.pdbqt')
    index_path = os.path.join(output_dir, 'index_files')
    index_files = split_index(index_file, index_path, batch_size=10)
    if batch is not None:
        index_files = index_files[batch[0]: batch[1]]

    result_dir = os.path.join(output_dir, drug, 'results')
    work_dir = os.path.join(output_dir, drug, 'work')
    rec_dir = os.path.join(output_dir, drug, 'rec_files')
    log_dir = os.path.join(output_dir, drug, 'log_files')
    result_json = os.path.join(output_dir, drug, 'results.json')

    if nworkers > 1:
        pandarallel.initialize(nb_workers=nworkers)

        df_index = pd.DataFrame([{'batch': i, 'index_file': index} for i, index in enumerate(index_files)])
        df_index['result_dir'] = df_index['batch'].apply(lambda x: f'{result_dir}/batch_{x}/')
        df_index['work_dir'] = df_index['batch'].apply(lambda x: f'{work_dir}/batch_{x}/')
        df_index['rec_file'] = df_index['batch'].apply(lambda x: f'{rec_dir}/batch_{x}.txt')
        df_index['log_file'] = df_index['batch'].apply(lambda x: f'{log_dir}/batch_{x}.log')
        df_index['status_file'] = df_index['batch'].apply(lambda x: f'{log_dir}/batch_{x}.status')
        df_index['records'] = df_index.parallel_apply(lambda x: _run(drug_file, x['index_file'], target_path, x['result_dir'], 
                                               x['work_dir'], x['rec_file'], ifptarget_bin, vina_bin, x['status_file'], x['log_file'], overwrite), axis=1)
        records = df_index['records'].tolist()

    else:
        records = []
        for i, index_file in enumerate(index_files):
            _result_dir = os.path.join(result_dir, f'batch_{i}')
            _work_dir = os.path.join(work_dir, f'batch_{i}')
            _rec_file = os.path.join(rec_dir, f'batch_{i}.txt')
            _log_file = os.path.join(log_dir, f'batch_{i}.log')
            _status_file = os.path.join(log_dir, f'batch_{i}.status')
            record = _run(drug_file, index_file, target_path, _result_dir, _work_dir, 
                          _rec_file, ifptarget_bin, vina_bin, _status_file, _log_file, overwrite)
            records.append(record)

    json.dump(records, open(result_json, 'w'), indent=4)
    print(f'Finished {drug}')
    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug', type=str, required=True)
    parser.add_argument('--drug_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--index_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ifptarget_bin', type=str, required=True)
    parser.add_argument('--vina_bin', type=str, required=True)
    parser.add_argument('--nworkers', type=int, default=1)
    parser.add_argument('--batch', type=int, nargs='+', default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}]")
    print(f"Current PID: {os.getpid()}")
    print(f"Current Working Dir: {os.getcwd()}")
    os.system("echo Hostname: $(hostname)")
    args = parse_args()
    run(args.drug, args.drug_path, args.index_file, args.output_dir, args.target_path, 
        args.ifptarget_bin, args.vina_bin, overwrite=False, nworkers=args.nworkers, batch=args.batch)

    print(f"[{time.strftime('%Y/%m/%d-%H:%M:%S')}] Finished.")

