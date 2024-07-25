#%% 
from pathlib import Path
from dipy.io.streamline import load_tractogram, save_tractogram, load_trk
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.data import get_fnames
import torch
import csv
import pandas as pd
import h5py
# from modules.utils import *
# from modules.pointClassifier import *
import datetime
from pprint import pprint
import os
import nibabel as nib

#%% Read and Load trks for 1 subject
class datagen():
    def __init__(self):
        """placeholder init function"""
        return
    
    def merge_trks(self, input_files:list, output_file_path:str): 
        """For merging left and right part of the same tract bundle or for merging different CC sub-parts
        In some cases, also to merge the whole brain tractograms"""
        # For merging .trks only
        import subprocess
        command = ['track_merge']
        command.extend(input_files)
        command.extend([output_file_path])
        print(command)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    def load_trks(self, cap:float, subject, trk_path:str, cap_streamlines=1000): 
        """load all the trks in a given folder of .trk files
        conditions: load 20% streamlines for tracts with more than cap_streamlines streamlines
        input: .trk
        output: dict of bundles
        
        includes stratified sampling"""
        i = 0
        bundles = {}
        for path in Path(trk_path).iterdir(): 
            if 'CC.trk' in str(path):
                os.rename(str(path), str(path).replace('CC.trk', 'CC_0.trk'))
                print('renamed to CC_0')
                break
            else: 
                pass
            if subject + '.trk' in str(path):
                os.remove(str(path))
        
        for path in Path(trk_path).iterdir():
            # iterate over the path where hcp842 is stored and take out all tractograms and name of the bundles
            if path.is_file() and path.suffix == '.trk':
                # try:
                #     if 'CC.trk' in str(path):
                #         os.rename(str(path), str(path).replace('CC.trk', 'CC_0.trk'))
                #         print('renamed to CC_0')
                # except:
                #     pass
                if subject in str(path):
                    pass

                bundle_name = str(path).replace(".trk", "")
                bundle_name = bundle_name.split('/')[-1]
                try:
                    bundle_track = load_tractogram(str(path), 'same', bbox_valid_check=False)
                except:
                    # print(path)
                    continue
                bundle_streams = bundle_track.streamlines
                len_bundle = len(bundle_streams)
                
                if len_bundle<=cap_streamlines:
                    cap_bundle_streams = select_random_set_of_streamlines(bundle_streams, int(len_bundle))
                else:
                    #sampling only 20% of the stramlines if the number of streamlines is less than equal to cap_streamlines
                    cap_bundle_streams = select_random_set_of_streamlines(bundle_streams, int(cap*len_bundle))

                bundles[bundle_name]  = [i,cap_bundle_streams]
                i = i + 1
        return bundles

    def tractogram_descriptor(self, bundles:dict, path:str):
        """bundles as a dict with each key representing [label, [all streamlines of that key]]
        @bundles: dict output from load_trks
        @path: path for writing the csv file
        here keys are labels, and dictionary elements are attached to each key
        features:
        get the max length of streamline, max no. of streamlines, description of streamline"""
        bundle_tensors = {}
        get_tensor = lambda x: [torch.from_numpy(x[i]) for i in range(len(x))]
        get_max_length = lambda x: max([len(s) for s in x])
        get_min_length = lambda x: min([len(s) for s in x])
        lcnv = pd.read_csv('/home/ang/Documents/GitHub/TractoBERT/csv/label_convention.csv')
        bundles_info = [] 
        for b in bundles.keys():
            bundle_tensors[b] = [bundles[b][0], get_tensor(bundles[b][1])]
            max_len_bundle = get_max_length(bundle_tensors[b][1])
            min_len_bundle = get_min_length(bundle_tensors[b][1])
            label_int = int(lcnv[lcnv['tract'] ==b]['label'])
            bundles_info.append([b, label_int, len(bundles[b][1]), max_len_bundle, min_len_bundle])

        if os.path.isfile(path): # removing multiple occurrences of the .csv file
            os.remove(path)
        else: 
            pass
        with open(path, "a", newline="") as file:
            writer = csv.writer(file)
            for info in bundles_info:
                writer.writerow(info)

        return bundle_tensors

    # Now selecting the bundles that need to be selected either automated OR manual

    def get_keys_from_dict(self, original_dict, selected_keys):
        """for selecting the tracts or bundle names from the total dictionary of bundles
        returns: new_dict with all the selected streamlines """
        # keys from the dictionary of streamlines
        new_dict = {key: original_dict[key] for key in selected_keys if key in original_dict}
        return new_dict

    def prune_tracts(self, df, bundle_tensors, nsmallest=None, nlargest=None, add_tracts=None, remove_tracts=None):
        """use only on of the argument, else everything is set to None
        nlargest and nsmallest are automatic and add_tracts, remove_tracts are manual"""
        if nsmallest is not None:
            minor_tracts = df.nsmallest(nsmallest, ['streamlines']).tract.to_list()
            df = df[~df['tract'].isin(minor_tracts)]
            return self.get_keys_from_dict(bundle_tensors, df['tract'].to_list())
        elif nlargest is not None: 
            dominant_tracts = df.nlargest(nlargest, ['streamlines']).tract.to_list()
            df = df[df['tract'].isin(dominant_tracts)]
            return self.get_keys_from_dict(bundle_tensors, df['tract'].to_list())
        elif add_tracts is not None:
            return self.get_keys_from_dict(bundle_tensors, add_tracts)
        elif remove_tracts is not None:
            df = df[~df['tract'].isin(remove_tracts)]
            return self.get_keys_from_dict(bundle_tensors, df['tract'].to_list())
        
    def create_hdf5_from_dictionary(self, data_dict, file_path):
        # specifically for bundles stored in the dictionary format
        i = 0 
        with h5py.File(file_path, 'w') as hdf5_file:
            for key, value in data_dict.items():
                label = value[0]
                for idx, item in enumerate(value[1]):
                    dataset_name = f'{key}__{label}_{idx}'  # Creating a unique name for the dataset
                    hdf5_file.create_dataset(dataset_name, data=item)
                i = i+1

    def trk2hdf5(self, subject:str, interp_size:int, omit_labels:list, select_labels:list, FOLDER_PATH:str, TRK_PATH:str, cap:float, nlargest=None, nsmallest=None, add_tracts=None, remove_tracts=None ):
        """
        @subject: subject ID or subject name
        @interp_size: interpolation size for the streamlines, defining the length of the final streamline (dimensions)
        @omit_labels: labels which need to be omitted even after removing some tracts from the source trk for experimentation purposes
        @FODLER_PATH: write folder for the .h5 of the data
        @TRK_PATH: source directory where all the trks are stored
        @nlargest: selecting largest tracts
        @remove_tracts: in some cases removing extra tracts which are usually present in the ATLAS"""

        bundles = self.load_trks(cap, subject, TRK_PATH)
        print("All capped tracts loaded")
        bundle_tensors = self.tractogram_descriptor(bundles, "csv/" + subject + ".csv")
        print("Tensors created and Tracts Analysed")
        df = pd.read_csv( "csv/" + subject + ".csv", names=['tract', 'label', 'streamlines', 'max_len', 'min_len'], header=None)
        cap_bundles = self.prune_tracts(df, bundle_tensors, nlargest= nlargest, nsmallest=nsmallest, add_tracts=add_tracts, remove_tracts=remove_tracts)
        print("Pruning done")
        file_path = FOLDER_PATH + "_" + subject + ".h5"
        try:
            os.remove( file_path)
        except: 
            pass
        self.create_hdf5_from_dictionary(cap_bundles, file_path)
        print("Stratified Sampled Streamlines [All Tracts] at {0} of major tracts".format(cap), file_path)
        with h5py.File(file_path, 'r') as hdf5_file:
            streamlines, labels = get_numpy_data(hf = hdf5_file, points_per_streamline= interp_size, omit_labels=omit_labels, select_labels=select_labels)
        if select_labels is not None: 
            assert(len(np.unique(labels)) == len(select_labels))
            # print("check get_numpy_data function in utils only")
        return streamlines, labels, file_path


# %%