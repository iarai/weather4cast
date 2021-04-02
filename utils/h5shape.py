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
import sys, getopt
import h5py


def load_test_file(file_path):
    """
    Given a file path, loads test file (in h5 format).
    Returns: tensor of shape (number_of_test_cases = 5, 3, 3, 496, 435) 
    """
    # load h5 file
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])

    # get relevant test cases
    data = data[0:]
    data = np.stack(data,axis=0)
    # transpose
    return data

def print_shape(data):
    print(data.shape)


if __name__ == '__main__':

    # gather command line arguments.
    infile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:", ["infile="])
    except getopt.GetoptError:
        print('usage: h5shape -i <path to h5 file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: h5shape -i <path to h5 file>')
            sys.exit()
        elif opt in ("-i","--infile"):
            infile = arg

    data = load_test_file(infile)
    print_shape(data)