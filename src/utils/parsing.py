
from argparse import ArgumentParser,FileType


def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='../results', help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--cache_path', type=str, default='../data/cache', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='../data/PDBBind/', help='Folder containing original structures')
    parser.add_argument('--split_train', type=str, default='../data/splits/timesplit_no_lig_overlap_train', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='../data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='../data/splits/timesplit_test', help='Path of file defining the split')
    parser.add_argument('--test_sigma_intervals', type=str2bool, default=False, help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=5, help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--train_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps for inference on val')
    parser.add_argument('--num_inference_complexes', type=int, default=100, help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--inference_earlystop_metric', type=str, default='valinf_rmsds_lt2', help='This is the metric that is addionally used when val_inference_freq is not None')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max', help='Whether to maximize or minimize metric')
    parser.add_argument('--project', type=str, default='diffphore_train', help='')
    parser.add_argument('--run_name', type=str, default='', help='')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=16, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', type=str2bool, default=False, help='pin_memory arg of dataloader')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Whether to overwrite exsiting run.')
    parser.add_argument('--debug', type=str2bool, default=False, help='Whether to run in debug mode.')
    parser.add_argument('--config_mode', type=str, default='model', choices=['model', 'all'], help='Choose to load model-related config or load all parameters.')
    parser.add_argument('--pretrain_model_pt', type=str, default=None, help='Pretrained model checkpoint.')


    # Training arguments
    parser.add_argument('--model_type', type=str, default='diff', choices=['diff', 'tank'], help='The type of model to use, choose from ["diff", "tank"].')
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=40, help='Patience of the LR scheduler')
    parser.add_argument('--lr_decay_factor', type=float, default=0.9, help='The decay factor of the learning rate.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None, help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', type=str2bool, default=True, help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs to warm up training with less sample.')
    parser.add_argument('--warmup_propotion', type=float, default=0.03, help='Propotion of samples to warm up.')
    parser.add_argument('--warmup_number', type=int, default=20000, help='Number of samples to warm up.')
    parser.add_argument('--train_propotion', type=float, default=0.03, help='Propotion of samples to train with less sample.')
    parser.add_argument('--train_number', type=int, default=200000, help='Number of samples to train with less sample.')
    parser.add_argument('--valid_warmup_propotion', type=float, default=0.03, help='Propotion of samples to warm up.')
    parser.add_argument('--valid_warmup_number', type=int, default=1000, help='Number of samples to warm up.')
    parser.add_argument('--valid_propotion', type=float, default=0.03, help='Propotion of samples to train with less sample.')
    parser.add_argument('--valid_number', type=int, default=10000, help='Number of samples to train with less sample.')
    parser.add_argument('--random_select', type=str2bool, default=False, help='Whether to select part of the dataset to train the model.')
    parser.add_argument('--keep_origin_config', type=str2bool, default=False, help='Whether to keep original config when restart a model training.')
    parser.add_argument('--model_ckpt', type=str, default='last_model.pt', help='Choose the trained model checkpoint.')
    parser.add_argument('--ro5_filter', type=str2bool, default=False, help='Filter the dataset with Rule of five.')
    parser.add_argument('--load_optimizer', type=str2bool, default=True, help='Whether to load the state dict of the optimizer.')
    parser.add_argument('--freeze_epoch', type=int, default=0, help='The epochs to freeze the process layers of the model, which is useful when fine-tuning.')
    parser.add_argument('--finetuning', type=str2bool, default=False, help='Whether to fine tune.')
    parser.add_argument('--fitscore', type=str2bool, default=True, help='Whether to calculate the fitscore.')
    parser.add_argument('--store_ranked_pose', type=str2bool, default=False, help='Whether to store the ranked docking poses.')
    parser.add_argument('--keep_update', type=str2bool, default=False, help='Whethter to keep the updates in the process of sampling and reverse process.')


    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='If positive, the number of training and validation complexes is capped')
    parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', type=str2bool, default=True, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--consider_ex', type=str2bool, default=True, help='Whether to consider the exclusion volumes')
    parser.add_argument('--neighbor_cutoff', type=float, default=5., help='The cutoff to link exclusion volumes to other nodes.')
    parser.add_argument('--ex_connected', type=str2bool, default=True, help='Whether to connect exlusion volumes to ordinary pharmacophore types.')
    parser.add_argument('--use_las_constrains', type=str2bool, default=True, help='Whether to use LAS constraints.')
    parser.add_argument('--use_phore_rule', type=str2bool, default=True, help='Whether to use mannually added phore matching rule to train the model.')
    parser.add_argument('--save_single', type=str2bool, default=False, help='Whether to save each data point into pickle file.')
    parser.add_argument('--chembl_path', type=str, default='../data/ChEMBL/', help='ChEMBL dataset path')
    parser.add_argument('--zinc_path', type=str, default='../data/ZINC/', help='ZINC dataset path')
    parser.add_argument('--dataset', type=str, default='pdbbind', choices=['pdbbind', 'chembl', 'zinc'], help='Choose the dataset.')
    parser.add_argument('--use_sdf', type=str2bool, default=True, help='Use the `sdf` format ligand to generate phore files.')
    parser.add_argument('--near_phore', type=str2bool, default=False, help='Whether the generated exclusion volumes near the pharmacophores.')
    parser.add_argument('--reject', type=str2bool, default=False, help='Whether to reject samples when T_ < R_ or T_ < Theta_ or R_ < Theta_ at training.')
    parser.add_argument('--reject_rate', type=float, default=0.3, help='The reject rate of when --reject set as True.')
    parser.add_argument('--flag', type=str, default="", help='The extra detail of the dataset.')
    parser.add_argument('--phore_path', type=str, default=None, help='The pharmacophore dataset path.')
    parser.add_argument('--contrastive', type=str2bool, default=False, help='Whether to use the contrastive model to compute features for the model')
    parser.add_argument('--contrastive_model_dir', type=str, default="", help='The path to the contrastive model.')
    parser.add_argument('--delta_t', type=float, default=0.05, help='The noise sampling from inference result of previous step minus `delta_t`.')
    parser.add_argument('--rate_from_infer', type=float, default=0.0, help='The possibility to sample from inference results.')
    parser.add_argument('--epoch_from_infer', type=int, default=300, help='The starting epoch of sampling from inference results.')
    parser.add_argument('--dynamic_coeff', type=float, default=0, help='Dynamic coefficient factor, default 0. 0 means dynamic probability is not considered.')
    parser.add_argument('--max_phore_num', type=int, default=999, help='Maximium number of pharmacophores allowed.')
    parser.add_argument('--min_phore_num', type=int, default=0, help='Minium number of pharmacophores allowed.')
    parser.add_argument('--fitscore_cutoff', type=float, default=0, help='Minium fitscore of complex samples from PDBBind allowed.')
    

    # Diffusion
    parser.add_argument('--tr_weight', type=float, default=0.33, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=0.33, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=0.33, help='Weight of torsional loss')
    parser.add_argument('--rot_sigma_min', type=float, default=0.1, help='Minimum sigma for rotational component')
    parser.add_argument('--rot_sigma_max', type=float, default=1.65, help='Maximum sigma for rotational component')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=30, help='Maximum sigma for translational component')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--no_torsion', type=str2bool, default=False, help='If set only rigid matching')
    parser.add_argument('--new', type=str2bool, default=False, help='If use MPL layers to update the weight of each vector in ligand-phore-cross layer.')


    # Tank 
    parser.add_argument("--consider_affinity", default=True, type=str2bool, help="Whether to consider affinity when using TankPhore Model.")
    parser.add_argument("--pred_dis", default=True, type=str2bool, help="Whether to predict distance distribution.")
    parser.add_argument("--contact_weight", default=1.0, type=float, help="The weight of contact loss")
    parser.add_argument("--affinity_weight", default=0.01, type=float, help="The weight of affinity loss")
    parser.add_argument("--pose_weight", default=5.0, type=float, help="The pose weight when calculating contact_loss")

    # Model
    parser.add_argument('--confidence_mode', type=str2bool, default=False, help='The training mode of the model')
    parser.add_argument('--confidence_dropout', type=float, default=0.0, help='The dropout rate of the confidence model')
    parser.add_argument('--confidence_no_batchnorm', type=str2bool, default=False, help='Whether not to use batchnorm in confidence model')
    parser.add_argument('--by_total', type=str2bool, default=False, help='The confidence model calculate the loss from the total fitscore or ph_overlap & ex_overlap')
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', type=str2bool, default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', type=str2bool, default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', type=str2bool, default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=25, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', type=str2bool, default=False, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')
    parser.add_argument('--consider_norm', type=str2bool, default=True, help='Whether to consider the norm of the pharmacophore points')
    parser.add_argument('--auto_phorefp', type=str2bool, default=False, help='Whether to automatically generate the phore fingerprint')
    parser.add_argument('--angle_match', type=str2bool, default=True, help='Whether to consider phore-ligand norm direction match')
    parser.add_argument('--phoretype_match', type=str2bool, default=True, help='Whether to consider pharmacophore type match.')
    parser.add_argument('--cross_distance_transition', type=str2bool, default=True, help='Whether to consider cross-distance as a weight of vector')
    parser.add_argument('--phore_direction_transition', type=str2bool, default=True, help='Whether to consider mannual phore direction as a weight of vector')
    parser.add_argument('--phoretype_match_transition', type=str2bool, default=True, help='Whether to consider pharmacophore type match as a weight of vector')
    parser.add_argument('--phore_rule', type=str2bool, default=True, help='Whether to use mannually added phore matching rule to train the model.')
    parser.add_argument('--ex_factor', type=float, default=-2.0, help='The direction factor of exclusion volumes to atoms.')
    parser.add_argument('--boarder', type=str2bool, default=False, help='Whether to consider the exclusion volume boarder.')
    parser.add_argument('--by_radius', type=str2bool, default=False, help='Consider the clash by atom radius.')
    parser.add_argument('--clash_tolerance', type=float, default=0.4, help='The tolerance of the clash.')
    parser.add_argument('--clash_cutoff', type=float, nargs='+', default=[1.0, 2.0, 3.0, 4.0, 5.0], help='The cutoff to consider the clash to exclusion volumes.')
    parser.add_argument('--use_att', type=str2bool, default=False, help='The cutoff to consider the clash to exclusion volumes.')
    parser.add_argument('--use_phore_match_feat', type=str2bool, default=False, help='Whether to consider the pharmacophore type match attribute for the cross edge.')
    parser.add_argument('--atom_weight', type=str, choices=['softmax', 'sigmoid', 'atomwise', 'phore'], default='softmax', help='The way of calculating the edge weight.')
    parser.add_argument('--trioformer_layer', type=int, default=1, help='The layers of Trioformer block.')
    parser.add_argument('--return_node', type=str2bool, default=True, help='Whether to use the node attribute as the output of contrastive model.')
    parser.add_argument('--norm_by_ph', type=str2bool, default=False, help='Whether to normalize the result with overlap volume.')
    parser.add_argument('--dist_for_fitscore', type=str2bool, default=False, help='Whether to incorporate the distance when predicting fitscore.')
    parser.add_argument('--angle_for_fitscore', type=str2bool, default=False, help='Whether to incorporate the angle matching when predicting fitscore.')
    parser.add_argument('--type_for_fitscore', type=str2bool, default=False, help='Whether to incorporate the pharmacophore type matching when predicting fitscore.')
    parser.add_argument('--sigmoid_for_fitscore', type=str2bool, default=False, help='Whether to use sigmoid as the activation function when predicting fitscore.')
    parser.add_argument('--readout', type=str, default='mean', help='The way to readout for the fitscore.')
    parser.add_argument('--as_exp', type=str2bool, default=False, help='To calculate the intermediate values for fitscore regression with exp() func.')
    parser.add_argument('--scaler', type=float, default=100.0, help='The scaler for the atom weight.')
    parser.add_argument('--multiple', type=str2bool, default=True, help='Whether multiple the atom weight and the total weight as the final weight for each cross edge.')

    args = parser.parse_args()
    # args = parser.parse_known_args()[0]

    args.phore_rule = args.angle_match or args.phoretype_match
    return args


def str2bool(inp):
    inp = inp.lower()
    if inp in ['y', 'yes', 'true', 't']:
        return True
    else:
        return False