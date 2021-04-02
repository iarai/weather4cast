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

import numpy as np
import xarray as xr

# ----------------------------------
# preprocess - static features
# ----------------------------------
def get_copies(l, n_copies):
    """ return the array duplicated for each sample in the sequence """
    arr = np.asarray([l for i in range(n_copies)])    
    return arr

def _norm(x, max_v, min_v):
    """ we assume max_v > 0 & max_v > min_v """
    return (x-min_v)/(max_v-min_v)

def normalize_latlon(latlons):
    norm_latlon = {'lat': {'max_v': 86, 'min_v': 23}, # it does not start from the equator & does not reah the pole
                   'lon': {'max_v': 76, 'min_v': -76}} # it does not consider full earth
    
    latlons[0] = _norm(latlons[0], **norm_latlon['lat'])
    latlons[1] = _norm(latlons[1], **norm_latlon['lon'])

    return latlons

def crop_Dataset(product, x_start, y_start, size=256):
    """ crop a squared region size
        provide upper-left corner with (x_start, y_start)
    """
    return product.isel(nx=slice(x_start, x_start+size), 
                        ny=slice(y_start, y_start+size))

def mk_crop_np(product, x_start, y_start, size=256):
    """ crop a squared region size^2
        provide upper-left corner with (x_start, y_start)
    """
    return product[y_start:y_start+size, x_start:x_start+size]

# ----------------------------------
# load extra information - static features
# ----------------------------------
def get_elevation(n_copies, crop=None, path='', shape=[1019, 2200], norm=True):

    altitudes = np.fromfile(path, dtype=np.float32)
    altitudes = altitudes.reshape(shape[0], shape[1])
    max_alt = altitudes.max()
    
    if crop is not None:
        altitudes = mk_crop_np(altitudes, **crop)
        
    if norm:
        # make under see level 0
        altitudes[altitudes<0] = 0

        # normalize
        altitudes = altitudes/max_alt
    
    return np.expand_dims(get_copies(altitudes, n_copies), axis=1), ['altitudes']

def get_lat_lon(n_copies, crop=None, path='', atts = ['latitude', 'longitude'], norm=True):

    latlons = xr.open_dataset(path)
    
    if crop is not None:
        latlons = crop_Dataset(latlons, **crop)
    
    # get only the values form the netcdf4 file
    latlons = [latlons[att][0].values for att in atts]
    
    if norm:
        latlons = normalize_latlon(latlons)
    
    return get_copies(latlons, n_copies), atts.copy()

def get_static(attributes, n_copies, paths, crop=None, channel_dim=1, norm=True):
    
    statics, descriptions = [], []
    funcs = {'l': get_lat_lon, 'e': get_elevation}
    
    for feature in attributes:
        if feature in 'le':
            data, channels = funcs[feature](n_copies, crop=crop, path=paths[feature], norm=norm)
            statics.append(data)
            descriptions += channels
    
    if len(statics)!=0:
        statics = np.concatenate(statics, axis=channel_dim)
    
    return statics, descriptions