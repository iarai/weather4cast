# Author: Pedro Herruzo
# Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
# IARAI licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import numpy as np
import netCDF4
import pandas as pd
import json
import h5py

# ----------------------------------
# Loading utils: blacklist & data splits
# ----------------------------------
def get_triple_idxs_w_blacklist(days, bins_to_predict=32, day_bins=96, len_seq_in=4, black_list_path='blacklist.json', 
                                verbose=False):

    triples = []
    with open(black_list_path) as data_file:
        # convert key 'dates' to integer
        black_list = json.load(data_file)
        black_list = {int(k): v for k, v in black_list.items()}
    
    # range for the input sequence
    in_range = day_bins-(len_seq_in+bins_to_predict-1)
    
    # for each day 'd', AND each starting timebin index 'i'
    for d in days:
        for i in range(in_range):
            seq_in_black = False
            
            # if day is in the blacklist AND any input frame in the sequence is missing
            # -->  we will block this sequence
            if d in black_list.keys():
                
                seq_bins = [i+j for j in range(len_seq_in)]
                timebins_in_black = [idx for idx in seq_bins if idx in black_list[d]]
                
                if len(timebins_in_black)>0:
                    seq_in_black = True
                    if verbose:
                        print(i, seq_bins, timebins_in_black, black_list[d])
            
            # only consider sequences with input frames not in the black_list
            if not seq_in_black:
                
                # even if input sequence is not in the black_list
                # we will only consider the triplet if the output is not in the blacl_list
                for o in range(bins_to_predict):
                    out_in_black = False
                    if d in black_list.keys():
                        if i+len_seq_in+o in black_list[d]:
                            out_in_black = True
                    
                    if not out_in_black:
                        triples.append((d, i, o))
                    else:
                        if verbose:
                            print("--> output")
                            print((d, i, i+o, black_list[d]))
            else: 
                if verbose:
                    print("input: ")
                    print(d, i)
                    print()

    return triples

def get_test_triplets(days, test_sequences, bins_to_predict=32):
    triples = []
    
    for d in days:
        day_id = str(d)[-3:]
        i = test_sequences[day_id]['bins_in']['0']['id_bin']
        for o in range(bins_to_predict):
            triples.append((d, i, o))
    return triples

def get_time():
    return ['{}{}{}{}00'.format('0'*bool(i<10), i, '0'*bool(j<10), j) for i in np.arange(0, 24, 1) for j in np.arange(0, 60, 15)]

def read_splits(path_splits, path_test_split=''):
    """ read dates splits with pandas and test_splits with json """
    df = pd.read_csv(path_splits, index_col=0)
    with open(path_test_split) as data_file:    
        test_splits = json.load(data_file)
    return df, test_splits

def get_next_day(sorted_days, current_day):
    """ get the next day """
    pos_date = np.argwhere(sorted_days==current_day)
    assert len(pos_date)==1, f" Error: date {current_day} not in the list"
    
    pos_date = pos_date[0, 0]
    assert pos_date+1<len(sorted_days), f" Error: date {current_day} is the last date, there is not a next day"

    return sorted_days[pos_date+1]

# ----------------------------------
# h5 input/output
# ----------------------------------
def unpack_h5_like_netCDF(data, add_offset, scale_factor, **kwargs):
    # unpack the variable into float32 as netCDF does
    # shttps://unidata.github.io/netcdf4-python/#Variable.set_auto_maskandscale
    #print(scale_factor, add_offset)
    data = data*scale_factor + add_offset
    return data

def scale_h5_variables(data, variables, process):
    """ preprocess h5 files form disk """
    for i, tgt_var in enumerate(variables):
        #print(tgt_var)
        data[:, i] = unpack_h5_like_netCDF(data[:, i], **process[tgt_var])
        data[:, i] = preprocess(data[:, i], **process[tgt_var])
        
    #assert 0 <= data.min() and data.max() <= 1, f"Error, the scale of the variables is wrong"
    return data

def write_data(data, filename):
    """ write data in gzipped h5 format with type uint16 """
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data=data, dtype=np.uint16, compression='gzip', compression_opts=9)
    f.close()
    
def read_h5_predictions(file_path):
    """ read h5 file and cast to float32 """
    
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key][:]
    
    data = data.astype(np.float32)

    return data

def read_and_normalize_h5(file_path, target_vars, preprocess):
    """ read and normalize a h5 file """
    data = read_h5_predictions(file_path)
    data = scale_h5_variables(data, target_vars, preprocess)
    return data

# ----------------------------------
# PREPROCESSING
# ----------------------------------

def postprocess(data, fill_value, max_value, add_offset, scale_factor):
    """ scales 'v' to the original scale ready to save into disk
    """

    # 1. scale data to the original range
    data = data*(max_value*scale_factor - add_offset) + add_offset
    
    # 2. pack the variable into an uint16 valid range (as netCDF does)
    # shttps://unidata.github.io/netcdf4-python/#Variable.set_auto_maskandscale
    data = (data-add_offset)/scale_factor
    
    # 3. Cast the data to integer
    # this step must be used to get back the original integer saved in the input files
    data = np.uint16(np.round(data, 0))
    assert data.max() <= max_value, f"Error, postprocess of the variables is wrong"
    
    return data

def postprocess_fn(data, variables, process):
    """ post process each variable separately """
    for i, tgt_var in enumerate(variables):
        data[:, i] = postprocess(data[:, i], **process[tgt_var])
        
    #data = data.astype(np.uint16)
    return data

def preprocess(data, fill_value, max_value, add_offset, scale_factor):
    """ scale it into [0, 1] 
    """
    for v in [fill_value, max_value, add_offset, scale_factor]:
        #print(v, type(v))
        pass
    return (data-add_offset)/(max_value*scale_factor - add_offset)

def preprocess_fn(data, fill_value, max_value, add_offset, scale_factor):
    """ returns a processed numpy array 
        from a given numpy masked array 'v' 
    """
    data = np.float32(data)

    # scale it into [0, 1]
    data = preprocess(data, fill_value, max_value, add_offset, scale_factor)

    # fill NaNs with 'fill_value'
    data = data.filled(fill_value)

    assert 0 <= np.nanmin(data) and np.nanmax(data) <= 1, f"Error, the scale of the variables is wrong"
    
    return data

# ----------------------------------
# DIMENSIONS
# ----------------------------------

def time_2_channels(w, height=-1, width=-1):
    """ move channels and sequence to the end, then combine them """
    if height==-1 or height==-1:
        height, width = w.shape[-2:]

    w = np.reshape(w, (-1, height, width)) # preserve spatial dimensions
    return w

def channels_2_time(w, seq_time_bins, n_channels, height, width):
    """ unroll time and channels """
    w = np.reshape(w, (seq_time_bins, n_channels, height, width)) 
    
    return w

def mk_crop_Dataset_netcdf4(product, x_start, y_start, size=256):
    """ crop a squared region size
        provide upper-left corner with (x_start, y_start)
    """
    return product[y_start: y_start+size, x_start: x_start+size]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor 
        credit: https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
    """
    return np.eye(num_classes, dtype='uint8')[y]

def preprocess_1hot_ct(v, fill_value, max_value):
    """ returns a processed numpy array 
        from a given numpy masked array 'v' 
        converting to 1-hot-encoding & move channels at the beginning
    """

    # get data filling NaNs with the value 'fill_value'
    v = v.filled(fill_value)
        
    v = to_categorical(v, max_value)

    v = np.moveaxis(v, -1, 0)
    
    return v

# ----------------------------------
# load products or sequences with netCDF4
# ----------------------------------
def get_prod_name(product):
    """ this is just a terrible hack since folder containing this product 
        and files have a slightly different name 
    """
    if product=='ASII':
        product = 'ASII-TF'
    return product

def read_netcdf4(product, day_in_year, time, root=''):
    if root=='':
        print("reading from shared file systen, you might want to inform the root variable to boost I/O")
        root='/iarai/public/t4c/aemet/nwcsaf.org/NATIVO'

    file = f'S_NWC_{get_prod_name(product)}_MSG*_Europe-VISIR_*T{time}Z.nc'
    path = f'{root}/{day_in_year}/{product}/{file}'
    
    file_path = glob.glob(path)
    assert len(file_path)==1, f" Error with file in {path} ----> all these files were found: {file_path}"
    file_path = file_path[0]
    ds = netCDF4.Dataset(file_path, 'r')

    return ds

def get_file_netcdf4(product, day_in_year, time, attributes,
                     root='', crop=None, params_process=None, 
                     ct_1hot=None, target=False):
    """ open a *.nc file and return only the specified attributes 
        - one_hot_ct allows to 1-hot encoding 'ct'
    """
    ds = read_netcdf4(product, day_in_year, time, root)
    ds_vars, ds_masks = {}, {}
    
    for attr in attributes:
        #ds.variables[attr].set_auto_scale(False) # uncomment & don't scale in preprocess_fn to see raw data
        v = ds.variables[attr][...]
        
        if crop is not None:
            v = mk_crop_Dataset_netcdf4(v, **crop)

        if isinstance(v.mask, np.bool_):
            v.mask = np.zeros(v.shape)
        ds_masks[attr] = v.mask
        
        if params_process is not None:
            if attr == 'ct' and ct_1hot is not None:
                v = preprocess_1hot_ct(v, **params_process[attr])
            else:
                v = preprocess_fn(v, **params_process[attr])
                
        else: # return raw value with NaNs where a mask is found
             v = v.filled(np.nan)
            
        ds_vars[attr] = v
            
    return ds_vars, ds_masks

def get_products_netcdf4(day_in_year, time, products, path,
                 attrs_order=['ct', 'ctth_pres', 'crr_intensity'],
                 crop=None, preprocess=None, ct_1hot=None, debug=False, 
                 target=False):
    """ loads all products and attributes into a single tensor 
        sorted by attrs_order
        
        returns 
            - numpy tensor with shape (attributes, ny, nx)
            - attrs_order
    """
    
    prods, masks = {}, {}
    for product, attributes in products.items():
        
        prod, mask = get_file_netcdf4(product, day_in_year, time, attributes, 
                                      root=path, crop=crop, params_process=preprocess, 
                                      ct_1hot=ct_1hot, target=target)
        prods = dict(prods, **prod)
        masks = dict(masks, **mask)
    
    if debug:
        for sorted_var in attrs_order:
            print(sorted_var, prods[sorted_var].shape)
    
    # to numpy
    if ct_1hot is not None: 
        v = 'ct'
        prods, masks = prods[v], masks[v]
        attrs_order = [ct_1hot[key] for key in sorted(ct_1hot.keys(), reverse=False)]
    else:
        prods = np.asarray([prods[sorted_var] for sorted_var in attrs_order])
        masks = np.asarray([masks[sorted_var] for sorted_var in attrs_order])

    return prods, masks, attrs_order.copy()

def get_sequence_netcdf4(len_seq, in_start_id, day_id, products, path, target_vars, 
                         hhmmss=get_time(), crop=None, preprocess=None, ct_1hot=None, 
                         day_bins=96, sorted_dates=None):
    """ input doesn't need the mask """
    
    sequence = []
    seq_info = {'day_in_year': [], 'time_bins': [], 'masks': []}
    already_next_day = True

    for time_bin in range(in_start_id, in_start_id+len_seq):
        if time_bin>=day_bins: # this is to load next days' timebin in the test split
            time_bin = time_bin % day_bins
            if not already_next_day:
                print(f'Since input sequence goes from {in_start_id} to {in_start_id+len_seq}')
                print(f'we should need to update day {day_id}...')
                next_day_id = get_next_day(sorted_dates, day_id)
                print(f'to {next_day_id}, but files are all in the folder of the former day.\n')
                #already_next_day = True
                
        prod, masks, channels = get_products_netcdf4(day_id, hhmmss[time_bin], products, path, 
                                                     target_vars, crop, preprocess, ct_1hot)
        sequence.append(prod)
        seq_info['day_in_year'].append(day_id)
        seq_info['time_bins'].append(time_bin)
        #seq_info['masks'].append(masks)
    
    #seq_info['masks'] = np.asarray(seq_info['masks'])
    seq_info['channels'] = channels

    # to numpy
    sequence = np.asarray(sequence)
        
    return sequence, seq_info