import argparse
import json
import math
from argparse import Namespace

def create_parser():
    # Variable Space
    parser = argparse.ArgumentParser(description="Train and evaluate a model",
                                     epilog="by Tengfei Xue txue4133@uni.sydney.edu.au")
    # Paths
    parser.add_argument('--input_path', type=str, default='./TrainData/outliers_data/DEBUG_kp0.1/h5_np15/',
                        help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./ModelWeights', help='Save trained models')
    # Hyperlocal representations
    parser.add_argument('--k', type=int, default=20, help='Local streamlines (k_local) the number of neighbor streamlines (in streamline level)')
    parser.add_argument('--k_ds_rate', type=float, default=0.1, help='1 means no downsample. downsample the tractography when calculating pairwise distance matrix for local streamlines.')
    parser.add_argument('--k_point_level', type=int, default=5, help='The number of neighbor points (in point level) on one streamline')
    # Training parameters
    parser.add_argument('--save_step', type=int, default=1, help='The interval of saving weights')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for Adam')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')
    parser.add_argument('--scheduler', type=str, default='wucd', help='type of learning rate scheduler')
    parser.add_argument('--dvae_config_path', type=str, default='../config/dvae', help='version of dVAE')
    parser.add_argument('--dvae_weight_path', type=str, default='../weight/dvae', help='version of dVAE')
    parser.add_argument('--step_size', type=int, default=5, help='Period of learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart (for wucd)')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases Ti after a restart (for wucd)')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--sample_pts', type=int, default=150, help='the number of epochs')
    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--loss', type=str, default='focal', help='loss name')
    parser.add_argument('--num_fiber_per_brain', type=int, default=1000, help='The number of fibers each brain')
    parser.add_argument('--num_point_per_fiber', type=int, default=15, help='The number of points each fiber')
    parser.add_argument('--use_tracts_training', default=False, action='store_true', help='Convert cluster labels into tracts during training')
    parser.add_argument('--use_tracts_testing', default=False, action='store_true', help='Convert cluster labels into tracts during testing')
    parser.add_argument('--use_dvae', default=False, action='store_true', help='to use dvae or not')
    parser.add_argument('--use_cnn', default=False, action='store_true', help='to use cnn or not')
    parser.add_argument('--use_pointnet', default=False, action='store_true', help='to use pointnet or not')
    parser.add_argument('--save_args_only', default=False, action='store_true', help='Save args only, not perform training')
    parser.add_argument('--cal_equiv_dist', default=False, action='store_true', help='Calculate equivalent distance for pairwise distance matrix')
    parser.add_argument('--recenter', default=False, action='store_true', help='Recenter the data use the center of mass')
    parser.add_argument('--num_features', type=int, default=3, help='Number of input features')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    
    return parser


def load_args(path, args):
    params_set_in_testing = ['aug_times', 'out_path']  # For the parameter name in this list, input the parameter value from test.py or test_realdata.py
    with open(path, 'r') as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key,value in saved_json_dict.items():
            if key in params_set_in_testing:
                print('Skip loading {} from training args'.format(key))
                continue
            args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def load_args_in_testing_only(path, args):
    """Only load augments that are used in testing"""
    params_set_in_testing = ['aug_times', 'out_path']  # For the parameter name in this list, input the parameter value from test.py or test_realdata.py
    with open(path, 'r') as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key,value in saved_json_dict.items():
            if key in params_set_in_testing:
                print('Skip loading {} from training args'.format(key))
                continue
            if key in args_dict.keys():  # only if arguments also appear at testing, we then load them from training.
                args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def save_args(path, args):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(args.__dict__)