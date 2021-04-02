

![Regions](/images/weather4cast_v1000-26.png?raw=true "Weather4cast competition")

# [Weather4cast](https://www.iarai.ac.at/weather4cast/): Multi-sensor weather forecasting competition & benchmark dataset

- Study satellite multi-channel weather movies.
- Predict weather products in various earth regions.
- Apply transfer learning to new earth regions.

## Contents
- [Weather4cast: Multi-sensor weather forecasting competition & benchmark dataset](#weather4cast-multi-sensor-weather-forecasting-competition--benchmark-dataset)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Get the data](#get-the-data)
  - [Submission guide](#submission-guide)
  - [Data summary](#data-summary)
  - [Start here](#start-here)
  - [Benchmarks](#benchmarks)
    - [Generate a submission](#generate-a-submission)
    - [Train/evaluate a UNet](#trainevaluate-a-unet)
  - [Cite](#cite)

## Introduction

The aim of our core competition is to predict the next 32 images (8h ahead in 15 minutes intervals) in our weather movies, which encode four different variables: (i) Temperature from the [Cloud Top Temperature](https://www.nwcsaf.org/ctth2) or the ground [Skin Temperature](https://www.nwcsaf.org/ishai_description) if there are no clouds, (ii) [Convective Rainfall Rate](https://www.nwcsaf.org/crr3), (iii) [Probability of Occurrence of Tropopause Folding](https://www.nwcsaf.org/asii-tf), and (iv) [Cloud Mask](https://www.nwcsaf.org/cma3). Each image is an observation of these 4 channels in a 15 minutes period where pixels correspond to a spatial area of ~3km x 3km, and there are 6 regions of 256 x 256 pixels to provide test predictions. From these regions, 3 of them contain also training and validation data for learning purposes but the other 3 only inference is requested, to assess the Transfer Learning capabilities of models.

![Regions](/images/stage1_regions.png?raw=true "Train/Validation/test Regions")

The submission format in each day of the test set is a multi-dimensional array (tensor) of shape (32, 4, 256, 256) and the objective function of all submitted tensors (one for each day in the test set and region) is the **mean squared error** of all pixel channel values to pixel channel values derived from true observations. We note that we normalize these pixel channel values to lie between 0 and 1 by dividing the pixel channel value by the maximum value of each variable (see below).

There are **two competitions** running in parallel that expect independent submission (participants can join one or both of them):
- [Core Competition](https://www.iarai.ac.at/weather4cast/competitions/weather4cast-2021/): Train your models on these regions with the provided data and submit predictions on the test subset.
- [Transfer Learning Competition](https://www.iarai.ac.at/weather4cast/competitions/w4c21-transfer/): Only the test subset is provided for these regions, test the generalization capacity of your models.

## Get the data
You can download the data once registered in the competition.
- Core Competition [Join and get the data](https://www.iarai.ac.at/weather4cast/forums/forum/competition/weather4cast-2021/)
- Transfer Learning Competition [Join and get the data](https://www.iarai.ac.at/weather4cast/forums/forum/competition/weather4cast-2021-transfer-learning/): 
## Submission guide

Currently, the competition data provided comes in a zip file that has the following folder structure.
```
+-- RegionX -- ...
+-- RegionY 
        +-- training -- ... (~96 files per variable)
        +-- validation -- ... (~96 files per variable)
        +-- test -- (4 files each variable)
            +-- 2019047
            +-- ... 
            +-- 2019074
                +-- ASII -- ...
                +-- CMA -- ...
                +-- CRR -- ...
                +-- CTTH
                    + -- S_NWC_CTTH_MSG4_Europe-VISIR_20190216T170000Z.nc
                    + -- S_NWC_CTTH_MSG4_Europe-VISIR_20190216T171500Z.nc
                    + -- S_NWC_CTTH_MSG4_Europe-VISIR_20190216T173000Z.nc
                    + -- S_NWC_CTTH_MSG4_Europe-VISIR_20190216T174500Z.nc
```
Each region has three splits training/validation/test, and each split has a folder yyyyddd that corresponds to the day number in that year *day_in_year*, e.g. 2019365 would refer to the last day in 2019. Each *day_in_year* has 4 folders containing the weather variables. All 15-minute period images available for that day are contained inside. We note that there is a maximum of 96 files for training/validation (4 images/hour * 24 hour), and exactly 4 files in test (1 hour as input for the next 32 requested consecutive images).

Each of the files S_NWC_`variable`_MSG4_Europe-VISIR_`yyyymmdd`T`hhmm`00Z.nc is a [netCDF](https://unidata.github.io/netcdf4-python/) encoding the respective requested target variable and other attributes that might help, for the same region of 256 x 256 pixels in the same 15-minutes interval. 

For the submission, we expect a zip file back that, when unpacked, decomposes into the following folder structure:
```
+-- RegionX -- ...
+-- RegionY 
        +-- test -- (1 file per day, encoding 32 images of 4 channels each)
            + -- 2019047.h5
            + -- ... 
            + -- 2019074.h5
```
where now each [h5](https://docs.h5py.org/en/stable/quick.html) file `yyyyddd`.h5 contains a uint16 tensor of shape (32, 4, 256, 256) that contains the predictions of 32 successive images following the sequence of 4 images given in the corresponding input test folders for all regions of the competition data. 
Note that each variable should be in its own range since we will scale it as mentioned above.

To check if the shape of a predicted tensor is appropriate (32, 4, 256, 256), the following script should give us exactly that:
```
python3 utils/h5file.py -i path_to_RegionY/test/yyyyddd.h5
```

To generate the compressed folder, `cd` to the parent folder containing the regions folders (RegionX, ..., RegionY) and zip all regions together like in the following example:
```
user@comp:~/tmp_files/core-competition-predictions$ ls
R1/ R2/ R3/
user@comp:~/tmp_files/core-competition$ zip -r ../core-predictions.zip .
...
user@comp:~/tmp_files/core-competition$ ls ../
core-competition-predictions/   transfer-learning-competition-predictions/  core-predictions.zip
```
Please, delete any file that makes your submission not to match the requested structure. An example of finding and deleting non-expected files is shown below: 
```
user@comp:~/tmp_files/core-competition-predictions$ find . -wholename *nb* -print
user@comp:~/tmp_files/core-competition-predictions$ find . -wholename *DS* -print
./.DS_Store
./R1/.DS_Store
./R3/.DS_Store
./R2/.DS_Store
user@comp:~/tmp_files/core-competition-predictions$ find . -wholename *DS* -delete
user@comp:~/tmp_files/core-competition-predictions$ find . -wholename *DS* -print
```

The submission file can be uploaded in the corresponding following submission link:
- [Core Competition](https://www.iarai.ac.at/weather4cast/competitions/weather4cast-2021/?submissions)
- [Transfer Learning Competition](https://www.iarai.ac.at/weather4cast/competitions/w4c21-transfer/?submissions)

## Data summary

The following table summarizes the data provided, detailing the requested target variables and all other variables provided. For a detailed explanation of each of the variables, see the link provided in the Introduction. Please, in order to keep the size of the provided prediction files small, just deliver them in uint16 format with each variable within its own range. When computing the score,  we will divide each channel by its maximum value to have them in the interval [0, 1]:

Name | Folder | Variables | Target Variable | type | Range | Scale Factor | Offset
:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
Temperature at Ground/Cloud | CTTH |  `temperature` <br /> `ctth_tempe` <br /> `ctth_pres` <br /> `ctth_alti` <br /> `ctth_effectiv` `ctth_method` <br /> `ctth_quality` <br /> `ishai_skt` <br /> `ishai_quality` | `temperature` | uint16 | 0-11000 | 10 | 0
Convective Rainfall Rate | CRR | `crr` <br /> `crr_intensity` <br /> `crr_accum` <br /> `crr_quality` | `crr_intensity` | uint16 | 0-500 | 0.1 | 0
Probability of Occurrence of Tropopause Folding | ASII | `asii_turb_trop_prob` <br /> `asiitf_quality` | `asii_turb_trop_prob` | uint8 | 0-100 | 1 | 0
Cloud Mask | CMA | `cma_cloudsnow` <br /> `cma` <br /> `cma_dust` <br /> `cma_volcanic` <br /> `cma_smoke` <br /> `cma_quality` | `cma` | uint8 | 0-1 | 1 | 0
Cloud Type | CT | `ct` <br /> `ct_cumuliform` <br /> `ct_multilayer` <br /> `ct_quality` | `None` | uint8 | `None` | `None` | `None`

Cloud Type is provided since it has rich information that might help the models but no variable is required from this product. For the other products, we expect predictions for [`temperature, crr_intensity, asii_turb_trop_prob, cma`] in this order for the channel dimension in the submitted tensor.

Data obtained in collaboration with AEMet - Agencia Estatal de Meteorología/ NWC SAF.

## Start here
We provide an introduction notebook in `utils/1. Onboarding.ipynb` where we cover all basic concepts for the competition from ~*scratch*:

1. How to read, explore, and visualize netCDF4 files
2. Load and transform context variables: *altitudes* and *latitude*/*longitude* 
3. Data split training/validation/test, and list of days with missing time-bins
4. Data Loader Example
5. Generate a valid submission for the Persistence model

Furthermore, you can find all explained methods in the notebook ready to use in the files `utils/data_utils.py` and `utils/context_variables.py`, so you can import them out of the box.

The code assumes that if you download the regions for the core or transfer learning competition, they are located like follows:
```
+-- data
    +-- w4c-core-stage-1 
        +-- R1
        +-- R2
        +-- R3
    +-- w4c-transfer-learning-stage-1
        +-- R4
        +-- R5
        +-- R6
    +-- static
        +-- Navigation_of_S_NWC_CT_MSG4_Europe-VISIR_20201106T120000Z.nc
        +-- S_NWC_TOPO_MSG4_+000.0_Europe-VISIR.raw
```
Please, provide the path to the parent folder `data` as the argument `data_path` of the function `get_params(...)` in `config.py`. 

Just in the same way, if you consider using the provided [static context variables](https://www.iarai.ac.at/weather4cast/forums/topic/weather4cast-2021-static-channels-common-files-for-any-competition/), provide the parent folder of the files `data/static` as the argument `static_data_path` of the function `get_params(...)`.


## Benchmarks

### Generate a submission

We provide a notebook (`2. Submission_UNet.ipynb`) where we show how to create a submission using pre-trained [UNet](https://arxiv.org/pdf/1505.04597.pdf) models, in particular, we will produce 3 sets of predictions:

* A valid submission for the core-competition (R1-3) using pre-trained UNets per region i.e. individual models per region
* A valid submission for the transfer-learning-competition (R4-6) using a single UNet trained on region R1
* Use the ensamble of models trained in regions R1-3 to generate a valid submission for the transfer-learning-competition (R4-6) by averaging their predictions

The weights needed to generate such submission for the UNets can be downloaded once registered to the competition [here](https://www.iarai.ac.at/weather4cast/forums/forum/competition/). The notebook uses an architecture and a [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) class defined in `weather4cast/benchmarks/`, but it is not required to understand them when learning how to generate the submissions from a pre-trained model.

### Train/evaluate a UNet
We provide a script (`3-train-UNet-example.py`) with all necessary code to train a UNet model from scratch or fine tune from any of the provided checkpoints for `2. Submission_UNet.ipynb`. The same code with the flag `-m val` can be used to evaluate a model on the validation data split.

To use it, set the correct data paths in `config.py` and use the following syntax:

```
user@comp:~/projects/weather4cast-participants/utils$ python 3-train-UNet-example.py --h                                    
usage: 3-train-UNet-example.py [-h] [-g GPU_ID] [-r REGION] [-m MODE]
                               [-c CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -g GPU_ID, --gpu_id GPU_ID
                        specify a gpu ID. 1 as default
  -r REGION, --region REGION
                        region_id to load data from. R1 as default
  -m MODE, --mode MODE  choose mode: train (default) / val
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        init a model from a checkpoint path. '' as default
                        (random weights)
```
Few examples:
```
cd utils

    # a.1) train from scratch
python 3-train-UNet-example.py --gpu_id 1 --region R1

    # a.2) fine tune a model from a checkpoint
python 3-train-UNet-example.py --gpu_id 2 --region R1 -c 'epoch=03-val_loss_epoch=0.027697.ckpt'

    # b.1) evaluate an untrained model (with random weights)
python 3-train-UNet-example.py --gpu_id 3 --region R1 --mode val

    # b.2) evaluate a trained model from a checkpoint
python 3-train-UNet-example.py --gpu_id 4 --region R1 --mode val -c 'epoch=03-val_loss_epoch=0.027697.ckpt'
```

Every time we run the script PyTorch Lightning will create a new folder `lightning_logs/version_i/`, increasing version `i` automatically. This is where the model parameters and checkpoints will be saved together with the files explained below.

The class `LeadTimeEval` in `benchmarks/validation_metrics.py` is used by the `UNet` model to store the error per variable in the evaluation of the validation data split. After the evaluation all errors are shown by the standard output. Furthermore, a plot `lead_times_mse_fig_R1.png` with the evolution of the mean error across time (from 1 to 32 future predictions) is produced saving also the values to disk `lead_times_mse_fig_R1.csv`, in the respective `lightning_logs/version_i/` folder. The latter can be used to compare different models across time. 

![Lead Times](/images/lead_times_mse_fig_R1.png "Lead Times")
The image above shows the mean error per time bin (y-axis) and its standard deviation up to 8 hours (32 time bins ahead, x-axis). The further the prediction the worst the error. The title of the picture indicates that this model used latitude/longitude and elevations (l-e), and indicates the mean error per variable averaging all 32 lead times.

## Cite

When referencing the data or the models provided, please cite [this
paper]().

```
@InProceedings{tbd}
```