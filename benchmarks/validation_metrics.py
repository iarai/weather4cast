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

class LeadTimeEval():
    """ This class helps to evaluate how a model performs across the prediction time horizon. 
        It will save the metrics per time lead predicted and create a plot with all of them.
    """
    
    def __init__(self, len_seq_in=4, bins_to_predict=32, n_channels=3):
        
        self.len_seq = len_seq_in
        self.n_bins = bins_to_predict
        self.n_channels = n_channels
        self.errors = {}
        
        self.index = ['day_in_year', 'in_start_id', 'channel']
        self.cols = self.index + [j for j in range(self.n_bins)]
        
    def get_numpy(self, x):
        return x.detach().cpu().numpy()
        
    def update_errors(self, err, metadata):
        """ Updates errors per channel for a particular 'date', 'starting time bin' and the 'lead time' predicted

        Note:
            - err.shape = (batch_size, channels_size)
            - metadata has to contain (day_in_year, lead_time, time_bins) 
        """
        
        days = self.get_numpy(metadata['out']['day_in_year'][0])
        lead_times = self.get_numpy(metadata['out']['lead_time'][0])
        target_times = self.get_numpy(metadata['out']['time_bins'][0])

        j = 0
        for d, lead_t, tgt_t, e in zip(days, lead_times, target_times, err):
            start_t = tgt_t - self.len_seq - lead_t
            #str_print = f'{j}- day={d}, lead_t={lead_t}, tgt_t={tgt_t}, start_t={start_t}, err={e}'
            #print(str_print)

            if d not in self.errors.keys():
                self.errors[d] = {}

            if start_t not in self.errors[d].keys():
                self.errors[d][start_t] = {}

            if lead_t not in self.errors[d][start_t].keys():
                self.errors[d][start_t][lead_t] = e # e.shape = channels_size
            else:
                print(f"Error, this lead_time={lead_t} was already updated in day={d} start_t={start_t}")

            j += 1
            
    def __update_channel_errors(self, errors, row_channels):
        """ Updates the errors per channel

        Args:
            errors (dict): errros of each variable
            row_channels (list): placeholder to save the errors per variable

        Returns:
            list: filled placeholder to save the errors per variable
        """
        for id_chn in range(self.n_channels):
            row_channels[id_chn].append(errors[id_chn])
        return row_channels
        
    def __get_lead_time_array(self, data, n_bins):
        """ builds a list of lists containing all predictions per 'date', 'starting time bin', 'variable', and 'lead time'
        """
        rows = []

        for id_date in data.keys():
            for id_start in data[id_date].keys():
                row_channels = []
                for id_chn in range(self.n_channels):
                    row_channels.append([id_date, id_start, id_chn])
                
                for j in range(n_bins):
                    # if the lead time is informed
                    if j in data[id_date][id_start].keys():
                        errors = data[id_date][id_start][j]
                        
                        row_channels = self.__update_channel_errors(errors, row_channels)
                    else:
                        row_channels = self.__update_channel_errors([np.NaN]*self.n_channels, row_channels)
                        
                for row in row_channels:
                    rows.append(row)

        return rows

    def get_lead_time_errors_df(self):
        """ Builds a spreadsheet containing all 32 predictions for all 'dates', 'starting time bins', and 'varriables'

        Returns:
            pandas.DataFrame: each rows contains the 32 predictions for a particular 'date' and 'starting time bin'
        """
        import pandas as pd
        
        rows = self.__get_lead_time_array(self.errors, self.n_bins)

        df = pd.DataFrame(rows, columns=self.cols)
        df = df.set_index(self.index).sort_index()

        return df

    def get_lead_time_metrics(self, root, title, region='', y_label='mse', x_label='lead times'):
        """ creates a plot for the prediction horizon

        Args:
            root (str): path to save the plots
            title (str): title of the plot
            region (str, optional): Region where the errors belong to. Defaults to ''.

        Returns:
            list: errors, standard deviations
        """
        import matplotlib.pyplot as plt
        fname = f'{root}/lead_times_mse_{region}.csv'
        fname_fig = f'{root}/lead_times_mse_fig_{region}'
        
        df = self.get_lead_time_errors_df()
        df.to_csv(fname, encoding='utf-8')
        print("saved errors to disk:", fname)
        
        errs, std = df.mean(), df.std()

        fig = plt.figure(figsize=(20,10))
        plt.errorbar(np.arange(len(errs)), errs, std, fmt='ok', lw=3)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xticks(np.arange(self.n_bins), np.arange(self.n_bins))
        plt.title(title)
        
        fig.savefig(fname_fig)
        plt.show()
        plt.close(fig)

        return list(errs), list(std)
        
    
        