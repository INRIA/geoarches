## General presentation

This codebase is the code for running and training [ArchesWeather](https://arxiv.org/abs/2405.14527).

Below is an exemple of a 10-day rollout for the ArchesWeather-M model initialized on January 1st, 2020 (with rollout steps of 24h).
![rollout_vid](https://huggingface.co/gcouairon/ArchesWeather/resolve/main/vid_rollout_jan1st.mp4)


## Installation

### Environment

```sh
conda create --name weather python=3.10
conda activate weather
pip install -r requirements.txt

mkdir sblogs
```

We recommend making the following symlinks in the codebase folder:
```sh
ln -s /path/to/data/ data
ln -s /path/to/models/ modelstore
ln -s /path/to/evaluation/ evalstore
ln -s /path/to/wandb/ wandblogs
```
Where `/path/to/models/` is where the trained models are stored, and `/path/to/evaluation/` is a folder used to store intermediate outputs from evaluating models. You can also simply create folders if you want to store data in the same folder.


### Data

The ``dl_era.py`` scripts downloads data from WeatherBench as netcdf files, because it was originally used on a system that could not handle the many files of the zarr storage system.
You can download the full dataset sequentially via `python dl_era.py`. If you wish to download the dataset in parrallel using multiple workers, you can download specific years with the script, e.g. via

```sh
python dl_era.py --clim # to download climatology for ACC metrics
python dl_era.py --year 2019,2020,2021 # to download specific years
```
You should download data from Weatherbench for years 1979 to 2021 (included). By default the dataset will be downloaded to `data/era5_240/`.


### Download model

```sh
mkdir modelstore/archesweather-M
src=https://huggingface.co/gcouairon/ArchesWeather/resolve/main
tgt=modelstore/archesweather-M
wget -O $tgt/archesweather-M_weights.pt $src/archesweather-M_weights.pt 
wget -O $tgt/archesweather-M_config.yaml $src/archesweather-M_config.yaml 
```

You can run a similar command to download the ArchesWeather-S model.


## ArchesWeather Inference

Here is a quick snippet on how to load an ArchesWeather model and perform inference:

```python
from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)

# load model and dataset
device = 'cuda:0'
fg = OmegaConf.load('modelstore/archesweather-M/archesweather-M_config.yaml')

ds = instantiate(cfg.dataloader.dataset, 
                    path='data/era5_240/full/', 
                    domain='test') # the test domain is year 2020

backbone = instantiate(cfg.module.backbone)
module = instantiate(cfg.module.module, backbone=backbone, dataset=ds)

ckpt = torch.load('modelstore/archesweather-M/archesweather-M_weights.pt', map_location='cpu')
module.load_state_dict(ckpt)
module = module.to(device).eval()


# make a batch
batch = {k:(v[None].to(device) if hasattr(v, 'to') else [v]) for k, v in ds[0].items()}
output = module.forward(batch)

# denormalize output
denorm_pred = ds.denormalize(output, batch)

# get per-sample main metrics from WeatherBench
from evaluation.deterministic_metrics import headline_wrmse
denorm_batch = ds.denormalize(batch)
metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')

# average metrics
metrics_mean = {k:v.mean(0) for k, v in metrics.items()}

#plot prediction
plt.imshow(denorm_pred['next_state_surface'][0, 2, 0].detach().cpu().numpy())

```

Multistep inference:

```python

multistep = 10
norm_batch = {k:(v.to(device) if hasattr(v, 'to') else v) for k, v in ds[0].items()}


#alternatively
traj = dict(traj_surface=[], traj_level=[])
for i in range(multistep):
    pred = module.forward(norm_batch)
    denorm_pred = ds.denormalize(pred, norm_batch)
    norm_batch = ds.normalize_next_batch(pred, norm_batch)

    traj['traj_surface'].append(denorm_pred['next_state_surface'].cpu().detach())
    traj['traj_level'].append(denorm_pred['next_state_level'].cpu().detach())


```

## Codebase logic

The codebase uses pytorch lightning, hydra, and logs data to Weights and Biases by default. For submission to SLURM it uses the submitit package. 

the configs are stored in `configs` folder. 
On each computing infrastructure, you can define the following alias
```sh
alias train='python submit.py cluster=example-slurm'
alias debug='python train_hydra.py cluster=example-slurm'

```
where `example-slurm` is the file in `configs/cluster` that contains information about how jobs should be started.

*train* submits the job to SLURM while *debug* starts the job directly. *train* will log to Weights and Biases by default, unlike *debug*.

## Training ArchesWeather

Example command on how to train ArchesWeather:

```python
train module=forecast-archesweather dataloader=era5-w
```

The target module is `lightning_modules.forecast.ForecastModule`, which is initialized with a backbone model defined in `backbones/archesweather`.

To override parameters:

```python
train module=forecast-gco dataloader=era5-w \
"++name=archesweather-s" \
"++module.backbone.depth_multiplier=1" \
```

The training script handles SLURM pre-emption: when a job is pre-empted, the script saves a checkpoint and requeues a job that will resume the current run.

By default, if you try to start a run that has the same name as a previous run, the script will check if the configurations for module and datasets are the same. If yes, it will resume the previous run, if not it will issue an error message and exit.


## External resources

Many thanks to the authors of WeatherLearn for adapting the Pangu-Weather pseudocode to pytorch. The code for our model is mostly based on their codebase.

[WeatherBench](https://sites.research.google/weatherbench/)

[WeatherLearn](https://github.com/lizhuoq/WeatherLearn/tree/master)

