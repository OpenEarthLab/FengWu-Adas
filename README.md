# FengWu-Adas: Towards an end-to-end artificial intelligence driven global weather forecasting system

This repo contains the official PyTorch codebase of <a href="https://arxiv.org/abs/2312.12462" target="_blank">FengWu-Adas</a>. 

## Codebase Structure

- `configs` contains the experiment configurations and trained model checkpoints.
- `data` contains the all the data used for model training and evaluation.
    - `era5` contains the ERA5 data, which can be downloaded from the official website of <a href="https://cds.climate.copernicus.eu" target="_blank">Climate Data Store</a>. 
    - `gdas` contains the real conventional observations of GDAS, which can be downloaded from the <a href="https://www.ncei.noaa.gov/has/HAS.DsSelect" target="_blank">Archive Information Request System</a>.
    - `igra` contains the processed IGRA radiosonde observations for $z500$ and $t850$ variables. The raw data and documentation are publicly available at the official website of <a href="https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive" target="_blank">NCEI</a>.
- `datasets` contains the dataset of ERA5 and observations, and corresponding mean and standard deviation values for standardization.
- `models` contains the Adas model and its basic modules, and the forecast model <a href="https://arxiv.org/abs/2304.02948" target="_blank">FengWu</a> (ONNX version).
- `utils` contains the files that support some basic needs.
- `train.py` and `inference.py` provide training and evaluation pipelines.

We provide the <a href="https://drive.google.com/file/d/1oePzgyY18qDzpJAq328arLMBIvCIdVzo/view?usp=sharing" target="_blank">ONNX model</a> of FengWu with 721×1440 resolution for making forecasts, which can also be found and downloaded in the <a href="https://github.com/OpenEarthLab/FengWu" target="_blank">official repository</a> of FengWu.

## Setup

First, download and set up the repo

```
git clone https://github.com/OpenEarthLab/FengWu-Adas.git
cd FengWu-Adas
```

Then, download and put the ERA5 data, GDAS observations and forecast model `FengWu.onnx` into corresponding positions according to the codebase structure.

Deploy the environment given below

```
python version 3.8.18
torch==1.13.1+cu117
```

## Training

We support multi-node and multi-gpu training. You can freely adjust the number of nodes and GPUs in the following commands.

To train the Adas model with the default configuration of ideal experiment with 10% simulated observations, just run

```
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=29500 train.py
```

You can freely choose the experiment you want to perform by changing the command parameters. For example, if you want to use 4 GPUs to train the model for 10 epochs with GDAS observations, you can run

```
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=29500 train.py --obs_type='gdas' --max_epoch=10 --world_size=4
```

## Evaluation

The commands for testing are the same as for training. You can use 1 GPU on 1 node to evaluate the analysis performance of FengWu-Adas in ideal experiment with 10% simulated observations through

```
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=29500 inference.py
```

The best checkpoint saved during training will be loaded to evaluate the RMSE metrics for all variables on ERA5. The evaluation of end-to-end forecast performance can be achieved by adjusting the `pred_len` parameter, which is equal to one sixth of the forecast hours because of the single-step foreacast interval of FengWu. For example, if you want to evaluate the forecast RMSE over 10 days on IGRA dataset when assimilating GDAS observations, you can run

```
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=29500 inference.py --obs_type='gdas' --pred_len=40 --eval_data='igra'
```

In addition, We have also provided the trained model checkpoints so that you can perform evaluation directly without training. Below are their download links:

<a href="https://drive.google.com/file/d/17PPX7SwU_6QOzrkppOjT_z48kzw2RsOk/view?usp=sharing" target="_blank">Adas checkpoint for GDAS conventional observations</a> 

<a href="https://drive.google.com/file/d/15NldueBhAs49AkHjApL9_FthJDOPhvOa/view?usp=sharing" target="_blank">Adas checkpoint for 10% simulated observations</a> 

<a href="https://drive.google.com/file/d/1cE8Jh0Vw5Xc1oK1qnS34GRVITxgH4fQX/view?usp=sharing" target="_blank">Adas checkpoint for 1% simulated observations</a> 

<a href="https://drive.google.com/file/d/1Q2i-smK00_LlQgSKcVFQTvChigmzMvFy/view?usp=sharing" target="_blank">Adas checkpoint for 0.1% simulated observations</a> 

Please download the checkpoints and put them in the corresponding directories of `configs`, and then rename them as `best_model.pth`.

## BibTeX
```bibtex
@article{chen2023towards,
  title={Towards an end-to-end artificial intelligence driven global weather forecasting system},
  author={Chen, Kun and Bai, Lei and Ling, Fenghua and Ye, Peng and Chen, Tao and Chen, Kang and Han, Tao and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2312.12462},
  year={2023}
}
```
