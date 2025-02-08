#%%
import nibabel as nib
import whitematteranalysis as wma
import vtk
import os
import sys
sys.path.append('../')
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
from utils.logger import create_logger
from utils.funcs import cluster2tract_label, unify_path, makepath, fix_seed, obtain_TractClusterMapping, array2vtkPolyData
from utils.cli import create_parser, load_args
from train_test.train import load_model, results_logging, train_val_test_forward
from dataloader.dataloader import RealData
from train_test.test import test_paths

def load_file(file_name, num_points=15):
    '''
    Read the trk file and convert to vtk polydata, then do feat_ras
    '''
    trk_file = nib.streamlines.load(file_name)
    streamlines = trk_file.streamlines

    pd_tractography = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    for streamline in streamlines:
        lines.InsertNextCell(len(streamline))
        for point in streamline:
            point_id = points.InsertNextPoint(point)
            lines.InsertCellPoint(point_id)

    pd_tractography.SetPoints(points)
    pd_tractography.SetLines(lines)

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=num_points)
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))

    return feat

def test_realdata_DL_net(net):
    """test the network"""
    test_predicted_lst = []
    # test
    with torch.no_grad():
        for j, data in enumerate(test_loader, start=0):
            _, _, test_predicted_lst = \
                train_val_test_forward(j, data, net, 'test_realdata', -1, [], test_predicted_lst,
                                       args, device, args.num_classes, epoch=1)

    return test_predicted_lst


def center_tractography(input_path, feat_RAS, out_path=None, logger=None, tractography_name=None,save_data=False):
    """Recenter the tractography to atlas center
        feat_RAS: [n_fiber, n_point, n_feat]"""
    HCP_center = np.load(input_path)  
    test_subject_center = np.mean(feat_RAS, axis=0)
    displacement = HCP_center - test_subject_center
    c_feat_RAS = feat_RAS + displacement 
    if save_data:
        recenter_path = os.path.join(out_path, 'recentered_tractography')
        makepath(recenter_path)
        feat_RAS_pd = array2vtkPolyData(c_feat_RAS)
        wma.io.write_polydata(feat_RAS_pd, os.path.join(recenter_path, 'recentered_{}'.format(tractography_name)))
        logger.info('Saved recentered tractography to {}'.format(recenter_path))
    return c_feat_RAS

if __name__ == '__main__':
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Variable Space
    parser = create_parser()
    args = parser.parse_args()
    # input from test.py keyboard
    args_path = args.out_path_base + '/cli_args.txt'
    # input from train.py keyboard, in cli.txt
    args = load_args(args_path, args)
    # fix seed
    fix_seed(args.manualSeed)
    # paths
    test_paths()
    # Tract cluster mapping
    ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
    # Record the training process and values
    logger = create_logger(args.out_log_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    # load data
    feat_RAS=load_file(args.tractography_path)
    c_feat_RAS = center_tractography(args.HCP_center, feat_RAS, out_path=args.out_path, logger=logger, tractography_name=args.tractography_name, save_data=args.save_data)
    # Real data processing
    test_realdata = RealData(c_feat_RAS,args,logger=logger) 
    test_loader = torch.utils.data.DataLoader(test_realdata, batch_size=args.test_realdata_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_realdata, batch_size=args.test_realdata_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_realdata, batch_size=args.test_realdata_batch_size, shuffle=False)
    test_realdata_size = len(test_realdata)
    args.weight_path = os.path.join(args.out_path, 'best_org_f1_model.pth')
    DL_model = load_model(args, num_classes=args.num_classes, device=device, test=True)  
    predicted_lst = test_realdata_DL_net(DL_model)

