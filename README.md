# Knowledge-Guided Diffusion Model for 3D Ligand-Pharmacophore Mapping

Official implementation of [Knowledge-Guided Diffusion Model for 3D Ligand-Pharmacophore Mapping](to be published) by Jun-Lin Yu, Cong Zhou, Xiang-Gen Liu, Guo-Bo Li. 

Pharmacophores are abstractions of essential chemical interaction patterns, holding an irreplaceable position in drug discovery. Despite the availability of many pharmacophore-based tools, the adoption of deep learning for pharmacophore-guided drug discovery remains relatively rare. We herein propose a novel knowledge-guided diffusion framework for ‘on-the-fly’ 3D ligand-pharmacophore mapping, named DiffPhore. It comprises a knowledge-guided ligand-pharmacophore mapping encoder, a diffusion conformation generator, and a calibrated conformation sampler. By training on two newly established benchmark datasets of pharmacophore-ligand pairs, DiffPhore achieved state-of-the-art performance in predicting ligand active conformations, surpassing traditional pharmacophore tools and several advanced docking methods. It also manifested superior virtual screening power for both lead discovery and target fishing. 
If you encounter any question, don't hesitate to open an issue or send us an email at vicfisher6072@163.com, liuxianggen@scu.edu.cn and liguobo@scu.edu.cn.

![mapping](figs/mapping.gif)

# Installation
## Requirements
DiffPhore is developed on the Rocky Linux 9.2 OS, CUDA 12.1 with Anaconda 23.3.1 and main packages as follows:
- Python 3.9.16
- PyTorch 1.12.1
- PyTorch Geometric 2.1.0.post1
- RDKit 2021.03.1

## Command
You can install these packages via the `environment.yml` file by running the following commands:
```
git clone https://github.com/VicFisher/DiffPhore.git
cd DiffPhore/src
conda env create -f environment.yml
conda activate diffphore
```
Note that the pytorch_geometric package installation might be problematic, you can try to install it manually through local `.whl` files.

# Usage
## 1. Calculate pharmacophore files using [AncPhore](https://ancphore.ddtmlab.org/)
Please refer to the [AncPhore](https://ancphore.ddtmlab.org/) website for more details.
Here, we provide a pharmacophore file as an example (`examples/phore/sQC_QFA_complex.phore`), which is the one used for virtual screening of sQC in the paper.

## 2. Run DiffPhore
### Pharmacophore mapping for a single ligand-pharmacophore pair
```
python src/inference.py --phore examples/phore/sQC_QFA_complex.phore --ligand examples/ligands/STK936575.sdf --cache_path data/caches --out_dir examples/output/1 --model_dir weights/diffphore_calibrated_warmuped_ft --sample_per_complex 40 --batch_size 20 --num_workers 6
```
`--phore` option specifies the pharmacophore file generated by AncPhore. <br />
`--ligand` specifies the ligand file in MOL/SDF format (only the first molecule will be used) or a text file contains SMILESs (each line as 'CCCCC xxx').<br />
`--cache_path` specifies the directory for storing the cached dataset of the inputs.<br />
`--out_dir` specifies the output directory. <br />
`--model_dir` specifies the directory containing the DiffPhore model weights.<br />
`--sample_per_complex` specifies the number of samples to be generated for each ligand-pharmacophore pair.<br />
`--inference_steps` specifies the number of denoising steps (default is 20)<br />
`--batch_size` specifies the batch size for the model inference.<br />
`--num_workers` specifies the number of workers for the data loader.<br />
`--fitness` specifies the fitness score to be used for ranking the poses, default is 1 (DfScore1). Options: DfScore1, DfScore2, DfScore3, DfScore4, DfScore5 (specific to target fishing task, or specify `--target_fishing True` instead). This depends on the AncPhore program contained in the `programs` folder.<br />
Other options can be found in the code.<br />
The output directory will contain the pharmacophore mapping results (mapping process, ranked poses and mapping summary files).<br />


### Pharmacophore mapping for multiple ligand-pharmacophore pairs (in the cases of virtual screening or target fishing)
```
python src/inference.py --phore_ligand_csv examples/task_file.csv --cache_path data/caches --out_dir examples/output/2 --model_dir weights/diffphore_calibrated_warmuped_ft --sample_per_complex 40 --batch_size 20 --num_workers 6
```
`--phore_ligand_csv` specifies the csv file containing the ligand and pharmacophore input information. The csv file should contains the header 'ligand_description' and 'phore', which require the same file format as `--ligand` and `--phore` options.<br />

### 3. Expected output
The pharmacophore mapping results, including aligned ligand structures ranked by corresponding fitness scores and some summary files,  are located in the output directory specified with `--out_dir`. It contains the following directories or files:
| File/Directory | Description | 
| ----------- | ----------- |
|mapping_process/|The cachce directory for the original aligned ligand structures and pharmacophore fitness calculation.|
|ranked_poses/| The cachce directory for "SDF" files containing aligned ligand poses ranked by the specified pharmacophore fitness score.|
|inference_metric.json| The "JSON" file recording the 'id', 'fitness' and 'run_time' for each input ligand-pharmacophore pair.|
|ranked_results.csv| The results ranked by maximium fitness score, which is useful for virtual screening task.|

# Training 
## 1. Warm-up training with LigPhoreSet
LigPhoreSet comprises >280,000 unique ligands derived from ZINC20 dataset and >800,000 corresponding ligand-pharmacophore pairs, which makes DiffPhore capture generalizable ligand-pharmacophore mapping (LPM) patterns across a broad chemical and pharmacophoric space. A demo command for this training stage is as follows:
```
python src/train.py --run_name diffphore_warmup --dataset zinc --lr 1e-3 --num_conv_layers 4 --ns 20 --nv 10 --phoretype_match True --consider_norm True --n_epochs 40 
```

## 2. Refinement training with CpxPhoreSet
CpxPhoreSet contains about 1,5000 imperfectly-matching ligand-pharmacophore pairs derived from crystal complex structures of PDBBind, which refine the model for understanding biased LPMs and gaining deeper insights into the induced-fit effects of ligand-target interactions. A demo command for this training stage is as follows:
```
python src/train.py --run_name diffphore_calibrated_warmup --restart_dir results/diffphore_warmup --dataset pdbbind --lr 1e-3 --num_conv_layers 4 --ns 20 --nv 10 --phoretype_match True --consider_norm True --dynamic_coeff 6 --epoch_from_infer 400 --rate_from_infer 0.6 --n_epochs 800 
```

**Note**: The datasets, including **LigPhoreSet** and **CpxPhoreSet**, will be made available after the publication of our paper. The training commands provided above are illustrative and not executable in their current form. A complete training script will be released alongside the datasets upon publication.

# Citation
Yu, J.; Zhou, C.; Ning, X.; Mou, J.; Meng, F.; Wu, J.; Chen, Y.; Liu, X.*; Li, G.-B*. Knowledge-Guided Diffusion Model for 3D Ligand-Pharmacophore Mapping (under review)<br />
*To whom correspondence should be addressed.

# Lisense
The code and model weights are released under MIT license. See the [LICENSE](LICENSE) file for details.


