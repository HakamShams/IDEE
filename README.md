[![Website](https://img.shields.io/badge/Website-IDE-CBBD93.svg?logo=Leaflet)](https://hakamshams.github.io/IDE/)
[![Paper](https://img.shields.io/badge/Paper-Openreview-CBBD93.svg?logo=openaccess)](https://openreview.net/forum?id=DdKdr4kqxh)
[![ArXiv](https://img.shields.io/badge/ArXiv-2410.24075-CBBD93.svg?logo=arxiv)](https://arxiv.org/abs/2410.24075)
[![Synthetic](https://img.shields.io/badge/Code-Synthetic_data-purple.svg?logo=github)](https://github.com/HakamShams/Synthetic_Multivariate_Anomalies)
![Python 3.10](https://img.shields.io/badge/python-3.10-purple.svg)
![License MIT](https://img.shields.io/badge/license-MIT-purple.svg)

<img align="left" src="docs/images/NeurIPS-logo.svg"  width="100" style="margin-right: 10px;"> <img align="left" src="docs/images/era5.png" width="45">
# Identifying Spatio-Temporal Drivers of Extreme Events (IDEE)

Computer Vision Group, Institute of Computer Science III, University of Bonn.

This is the code to reproduce the results presented in the paper:
["**Identifying Spatio-Temporal Drivers of Extreme Events**"](https://arxiv.org/abs/2410.24075) by [Mohamad Hakam Shams Eddin](https://hakamshams.github.io/), and [Juergen Gall](http://pages.iai.uni-bonn.de/gall_juergen/). Accepted at [NeurIPS'24](https://neurips.cc/Conferences/2024).

The code for generating the synthetic data is available at [Synthetic_Multivariate_Anomalies](https://github.com/HakamShams/Synthetic_Multivariate_Anomalies).

### [Website](https://hakamshams.github.io/IDE/) | [Paper](https://openreview.net/forum?id=DdKdr4kqxh)

[![video presentation](docs/images/video_cover_ide.jpg)](https://www.youtube.com/watch?v=_AD5moplxB0)

<br />

The spatio-temporal relations of extreme events impacts and their drivers in climate data are not fully understood and there is a need of machine learning approaches to identify such spatio-temporal relations from data. 
The task, however, is very challenging since there are time delays between extremes and their drivers, and the spatial response of such drivers is inhomogeneous. 
In this work, we propose a first approach and benchmarks to tackle this challenge. 
Our approach is trained end-to-end to predict spatio-temporally extremes and spatio-temporally drivers in the physical input variables jointly. 
We assume that there exist precursor drivers, primarily as anomalies in assimilated land surface and atmospheric data, for every observable impact of extremes. 
By enforcing the network to predict extremes from spatio-temporal binary masks of identified drivers, the network successfully identifies drivers that are correlated with extremes. 
We evaluate our approach on three newly created synthetic benchmarks where two of them are based on remote sensing or reanalysis climate data and on two real-world reanalysis datasets. 

## Poster

<table><tr><td>
  <img src="docs/poster/Shams_Gall.png">
</td></tr></table>

## Setup

For pip with a virtual environment:
```
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```
Pytorch has to be updated to be computable with your GPU driver.

If you want to use Mamba:
```
pip install mamba-ssm
pip install mamba-ssm[causal-conv1d]
```

## Code

The code has been tested under Pytorch 1.12.1 and Python 3.10.6 on Ubuntu 20.04.5 LTS with NVIDIA A100 & A40 GPUs and GeForce RTX 3090 GPU.

### Configuration

The main config file [config.py](config.py) includes the parameters for training and testing and models.

To train on the synthetic data:
```
  train_synthetic.py
```
For testing on the synthetic data:
```
  test_synthetic.py
```
Similarly, to train on the real-world data i.e., ERA5 Land:
```
  train_ERA5_Land.py
```
and for testing on ERA5 Land:
```
  test_ERA5_Land.py
```

The code is implemented with a simple DataParallel for multi-GPUs.
Training on real-world different than the one in the paper, requires fine-tuning for the hyper-parameters and weighting.

### Backbones
- [x] [Video Swin Transformer](models/encoder/Swin_3D.py)
- [x] [Vision Mamba](models/encoder/Mamba.py)
- [x] [3D CNN](models/encoder/CNN_3D.py)

### Baselines

- [Multiple instance learning](Baselines_MIL):
  - [x] DeepMIL
  - [x] ARNet
  - [x] RTFM
  - [ ] MGFN
- [One Class](Baselines_OneClass):
  - [x] SimpleNet
- [Reconstruction based](Baselines_Reconstruction):
  - [x] UniAD
  - [x] STEALNET

### Pretrained models

#### Synthetic CERRA Reanalysis

| name           | backbone               | resolution  | config                                                                                                                                                                                                                      | model                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|:---------------|:-----------------------|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepmil_exp3   | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=13r1xUVPEGDQFyRatn1J_PaOSB0CuloXM&export=download&authuser=1&confirm=t&uuid=725e5824-e074-47db-bdb0-afa27128772f&at=AENtkXYzb3RE1DDyHrx4mEGcXvD8:1732626689185)   | [checkpoint](https://drive.usercontent.google.com/download?id=1yquMBr7m45ggMS9X85YXhFYjgHL6ZPYr&export=download&authuser=1&confirm=t&uuid=1c6dfc74-17fa-413d-b59b-ba5377c53959&at=AENtkXYdCNXWkJHWyagWonL6G76z:1732626687306)                                                                                                                                                                                                                                | 
| arnet_exp3     | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1-So4MGCIWfs3lrScmbfj_Z93YHiveXAY&export=download&authuser=1&confirm=t&uuid=cdf07cfe-02ac-44a5-a596-f87abb704eb9&at=AENtkXYzZ0a6HjmbUklp0TURG2em:1732626595265)   | [checkpoint](https://drive.usercontent.google.com/download?id=1IACZtEd7X8bYwRgRqdBRQoBcbgekhDn1&export=download&authuser=0)                                                                                                                                                                                                                                                                                                                                  |
| rtfm_exp3      | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1aLnHix43c8KlquIpwCBf86oLmTaPUUTo&export=download&authuser=1&confirm=t&uuid=8743ac77-f22e-4ff7-bca5-9a01b79320cf&at=AENtkXbyM5dNBDfpsddwZFfd8Dwl:1732626863652)   | [checkpoint](https://drive.usercontent.google.com/download?id=1oMaGeBusPV4UvhaNC9ztxvGhfSWfM3JR&export=download&authuser=1&confirm=t&uuid=3f5376f2-b44d-41d9-b643-6af3a2b6abc8&at=AENtkXYtwSnHlDKT4iYkxTXfj4Qg:1732626862334)                                                                                                                                                                                                                                | 
| simplenet_exp3 | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1X-mRLLfTO6ou-Fq4lvCMdjteIC7aNiTn&export=download&authuser=1&confirm=t&uuid=cb4709c1-6721-4fb2-9e32-daa3af05a605&at=AENtkXZ6hwQ4vWTBKvrVOd-QlWaC:1732626922662)   | [checkpoint](https://drive.usercontent.google.com/download?id=1cpNeVCH2py1nmwftQnEKxHVwFPkl8mX2&export=download&authuser=1&confirm=t&uuid=3789ef6f-c351-4503-bcba-4a176ba89c04&at=AENtkXa8JBbb41oXuj8TN_tYzGkF:1732626925363) / [backbone](https://drive.usercontent.google.com/download?id=1AFQ8iCihS9iEjn0YGARco09mrMjFXQLV&export=download&authuser=1&confirm=t&uuid=8d7d889e-6b39-4fa5-b7e6-0b42aa0bd5d7&at=AENtkXbWX5SkTW9PecovtpytUlxM:1732626921592)  | 
| stealnet_exp3  | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1GECL3CeHaP34bxYeIibuKqm_t-bsenx6&export=download&authuser=1&confirm=t&uuid=97b0cf53-f4fb-44ce-ad71-de2431c856d4&at=AENtkXbT8XZh60L8TOUqOGmcsF1u:1732627043536)   | [checkpoint](https://drive.usercontent.google.com/download?id=1p8rS6drdNGmHIPcS3CF0lwduJX0nsEls&export=download&authuser=1&confirm=t&uuid=49b45321-3d39-4f3d-a780-337689adf73a&at=AENtkXYO5u_j6CN_LTVYHP9XfLen:1732627042676)                                                                                                                                                                                                                                | 
| uniad_exp3     | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1Z3Yo_kpTbfKlNy2YkggSfAqgjWkBYQ0d&export=download&authuser=1&confirm=t&uuid=9fadc492-786e-4d6f-a232-f2906cce2ac4&at=AENtkXbPeXrWjmTU7NPxBu4rwQjO:1732627051937)   | [checkpoint](https://drive.usercontent.google.com/download?id=1uy14ax8tdOi8reIYuTtT183qkieRgQQG&export=download&authuser=1&confirm=t&uuid=017c97bd-5fa3-4486-a651-9c3d5dbb77c7&at=AENtkXZXy59wjVo4qzazxWcINEs0:1732627051085)                                                                                                                                                                                                                                | 
| ide_swin_exp3  | Video Swin Transformer | 200x200     | [config](https://drive.usercontent.google.com/download?id=1MiPEYOcU6anDH2jzDLMAGTT7S-8RyQ5M&export=download&authuser=1&confirm=t&uuid=872cfc82-66ea-427a-a777-590a5ac807c3&at=AENtkXbRp2Bt2rRTfFWO_QY2mCjI:1732626803143)   | [checkpoint](https://drive.usercontent.google.com/download?id=18BduWcbMVPZJSt8yvv33yeSRkFfP8yCM&export=download&authuser=1&confirm=t&uuid=ddee8e8c-cdac-43f7-8fa6-bf97765e220b&at=AENtkXY0K7w5qPPCbBl3NXcjPRjc:1732626802452)                                                                                                                                                                                                                                | 
| ide_mamba_exp3 | Vision Mamba           | 200x200     | [config](https://drive.usercontent.google.com/download?id=1poplqvESUrNxB9WlphxS22GfTI5Ed1b7&export=download&authuser=1&confirm=t&uuid=4e6ae531-bfaa-4a9a-a22f-2b4c05f33834&at=AENtkXYTzDvbopofIjTTBmJjxYZ5:1732626775545)   | [checkpoint](https://drive.usercontent.google.com/download?id=1vwOSJ2T2dP_J58SWuLL_zmnqMl26-HLQ&export=download&authuser=1&confirm=t&uuid=3e511f73-d3cd-4532-8817-c97d5f6bb7c3&at=AENtkXaG1lXYbAnh3iM06awiAJZM:1732626695552)                                                                                                                                                                                                                                | 

#### Real-world CERRA Reanalysis

| name      | backbone               | region  | resolution  | config                                                                                                                                                                                                                     | model                                                                                                                                                                                                                          |
|:----------|:-----------------------|:--------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ide_CERRA | Video Swin Transformer | Europe  | 512x832     | [config](https://drive.usercontent.google.com/download?id=1uMjyz1tUS-3PbR7oNf26KMqdNqcsbS8z&export=download&authuser=1&confirm=t&uuid=d178582a-9adf-46d4-945f-cb7ddda7d9cb&at=AENtkXYj09fTrWygtB_hemX5i5g2:1732627313610)  | [checkpoint](https://drive.usercontent.google.com/download?id=1W68gao80Z3qLM3HoHZF8wRLYrzsdMTjj&export=download&authuser=1&confirm=t&uuid=e99cfc00-68a8-4bc5-baf9-1b62ecfcdb5b&at=AENtkXaxoOzhm5G4bfTEaiRc9v9V:1732627312891)  | 

#### Real-world ERA5-Land Reanalysis

| name    | backbone               | region         | cordex  | resolution  | config                                                                                                                                                                                                                     | model                                                                                                                                                                                                                          |
|:--------|:-----------------------|:---------------|:--------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ide_AFR | Video Swin Transformer | Africa         | AFR-11  | 804x776     | [config](https://drive.usercontent.google.com/download?id=1JAaH-uafMmUOX3ntFJrfgPhgUlB1uk3v&export=download&authuser=1&confirm=t&uuid=1bf44ff8-afad-4ad6-8a08-0739bdf5ecfa&at=AENtkXb9BcfdA4elY868tShZiBC1:1732627640576)  | [checkpoint](https://drive.usercontent.google.com/download?id=1siLytAg99S5lLSlqyiHDpWjhi0bGUfA5&export=download&authuser=1&confirm=t&uuid=9ccd9911-c76c-4be2-98c2-05778b2a67a6&at=AENtkXZg_LaUPTKF7l8O7qKTtt5G:1732627639554)  |
| ide_CAS | Video Swin Transformer | Central Asia   | CAS-11  | 400x612     | [config](https://drive.usercontent.google.com/download?id=1XqvdNMveZWVcdXNbgtZI7SKZX0r75zrk&export=download&authuser=1&confirm=t&uuid=3baef401-2a05-4ed9-9811-1da4eda7c14f&at=AENtkXZGv9NlP8vqrS94mvOJT-7P:1732627645569)  | [checkpoint](https://drive.usercontent.google.com/download?id=1gS-IaJ1mJ6P94S3_schkb2MDNph_2ssw&export=download&authuser=1&confirm=t&uuid=ccb1c8b5-3b6b-46c3-ad89-67352979729e&at=AENtkXZT0qSUGz6wip11Vh_RbvSB:1732627644470)  |
| ide_EAS | Video Swin Transformer | East Asia      | EAS-11  | 668x812     | [config](https://drive.usercontent.google.com/download?id=10sO-uQMEq7uWdwEF19_Xh0OI5usP6ToV&export=download&authuser=1&confirm=t&uuid=e877d70f-3fb5-4d37-aae8-09b900d08114&at=AENtkXaiz8o-ItljjhN0nrDU7YzE:1732627649975)  | [checkpoint](https://drive.usercontent.google.com/download?id=1aaWPzQUJx3BWq7r6VJXhFdXzbdNmXyXE&export=download&authuser=1&confirm=t&uuid=8f2c765a-c315-4aa2-9e68-92013b6d24e6&at=AENtkXYV4dktNIGor4uNOBonHa_g:1732627649162)  |
| ide_EUR | Video Swin Transformer | Europe         | EUR-11  | 412x424     | [config](https://drive.usercontent.google.com/download?id=1RpPPqxqElJ3YO2t86nZdo-mc5NS5zYnN&export=download&authuser=1&confirm=t&uuid=ebbcefad-5f88-47fd-a5b8-0930d9c6a46a&at=AENtkXawVoL8zC_hzTyp3BDj7z6l:1732627709089)  | [checkpoint](https://drive.usercontent.google.com/download?id=1KnI7j4n2QNx0dyHCyOm9JFHlMBM7A_J8&export=download&authuser=1&confirm=t&uuid=7b87c742-2329-47b3-9a4a-a72de2d84050&at=AENtkXZ1aBLXk64aAU2_bJaPvGhL:1732627707551)  | 
| ide_NAM | Video Swin Transformer | North America  | NAM-11  | 520x620     | [config](https://drive.usercontent.google.com/download?id=1X_CM52v2hD2DOZ2DmPcReX5DqgUyKDEw&export=download&authuser=1&confirm=t&uuid=1f8f2adb-e536-492f-aad2-44739e9cc1d2&at=AENtkXZWxTXy7hk8FAwE6kv90caF:1732627714098)  | [checkpoint](https://drive.usercontent.google.com/download?id=1TodH-aNwo_j5nIflDTgXtKzfphroWpvU&export=download&authuser=1&confirm=t&uuid=8cda2f1a-98f5-4327-ae75-8f9df1d21f7b&at=AENtkXagfYFv2_vTDl9MOFqBPzIi:1732627713100)  | 
| ide_SAM | Video Swin Transformer | South America  | SAM-11  | 668x584     | [config](https://drive.usercontent.google.com/download?id=1VjYXWDqVnQO_Csptt2NQNc0TlQKb1gBA&export=download&authuser=1&confirm=t&uuid=e943340d-dda3-4770-be9d-acc916b41396&at=AENtkXZgzLamN4p9O8o5S-bFSjoP:1732627718290)  | [checkpoint](https://drive.usercontent.google.com/download?id=1TFRSz2BXY9HlL2hyY64BLSFdErzYbsuo&export=download&authuser=1&confirm=t&uuid=9363a58a-53f6-45b9-9569-9f6bc34616c7&at=AENtkXaRrqI-rJVmg4aHNsiX3OIK:1732627717496)  | 


### Structure
```
├── Baselines_MIL
│   ├── config.py
│   ├── dataset
│   │   └── Synthetic_dataset.py
│   ├── log
│   ├── models
│   │   ├── agent
│   │   │   └── Swin_3D.py
│   │   ├── build_arnet.py
│   │   ├── build_deepmil.py
│   │   ├── build_mgfn.py
│   │   ├── build_rtfm.py
│   │   ├── classifier
│   │   │   ├── ARNet.py
│   │   │   ├── DeepMIL.py
│   │   │   ├── MGFN.py
│   │   │   └── RTFM.py
│   │   ├── encoder
│   │   │   ├── CNN_3D.py
│   │   │   ├── Mamba.py
│   │   │   └── Swin_3D.py
│   │   └── losses.py
│   ├── test_mil_synthetic.py
│   ├── train_arnet_synthetic.py
│   ├── train_deepmil_synthetic.py
│   ├── train_mgfn_synthetic.py
│   ├── train_rtfm_synthetic.py
│   └── utils
│       └── utils_train.py
├── Baselines_OneClass
│   ├── config.py
│   ├── dataset
│   │   └── Synthetic_dataset.py
│   ├── log
│   ├── models
│   │   ├── build_simplenet.py
│   │   ├── encoder
│   │   │   ├── CNN_3D.py
│   │   │   ├── Mamba.py
│   │   │   └── Swin_3D.py
│   │   └── losses.py
│   ├── test_simplenet_synthetic.py
│   ├── train_simplenet_synthetic.py
│   └── utils
│       └── utils_train.py
├── Baselines_Reconstruction
│   ├── config.py
│   ├── dataset
│   │   └── Synthetic_dataset.py
│   ├── log
│   ├── models
│   │   ├── build_steal.py
│   │   ├── build_uniad.py
│   │   ├── initializer.py
│   │   └── losses.py
│   ├── test_steal_synthetic.py
│   ├── test_uniad_synthetic.py
│   ├── train_steal_synthetic.py
│   ├── train_uniad_synthetic.py
│   └── utils
│       └── utils_train.py
├── config.py
├── dataset
│   ├── CERRA_dataset.py
│   ├── ERA5_Land_dataset.py
│   └── Synthetic_dataset.py
├── docs
│   ├── images
│   │   ├── era5.png
│   │   ├── NeurIPS-logo.svg
│   │   ├── neurips-navbar-logo.svg
│   │   └── video_cover_ide.jpg
│   └── poster
│       └── Shams_Gall.png
├── log
├── models
│   ├── build.py
│   ├── classifier
│   │   └── CNN_3D.py
│   ├── codebook
│   │   ├── FSQ.py
│   │   ├── LatentQuantize.py
│   │   ├── LFQ.py
│   │   ├── Random_VQ.py
│   │   └── VQ.py
│   ├── encoder
│   │   ├── CNN_3D.py
│   │   ├── Mamba.py
│   │   └── Swin_3D.py
│   └── losses.py
├── README.md
├── scripts
│   ├── download_cerra.sh
│   ├── download_era5_land.sh
│   ├── download_noaa_cerra.sh
│   ├── download_noaa_era5_land.sh
│   └── download_synthetic.sh
├── test_CERRA.py
├── test_ERA5_Land.py
├── test_synthetic.py
├── train_CERRA.py
├── train_ERA5_Land.py
├── train_synthetic.py
├── utils
│   └── utils_train.py
└── vis
    ├── visualize_CERRA_data.py
    ├── visualize_ERA5-Land_data.py
    ├── visualize_NOAA_data.py
    └── visualize_synthetic_data.py
```

## Dataset

- The full data set can be obtained from [https://doi.org/10.60507/FK2/RD9E33](https://doi.org/10.60507/FK2/RD9E33) (~ 1.1 TB after decompression).
- The data can be also downloaded via scripts found in [scripts](scripts) i.e., you can download the synthetic data via [script/download_synthetic.sh](scripts/download_synthetic.sh) (~46 GB):
  ```
   wget --continue  https://bonndata.uni-bonn.de/api/access/datafile/7506 -O Synthetic.7z
  ```
  To extract the files you need the 7-Zip packge:
  ```
   sudo apt update
   sudo apt install p7zip-full
  ```
  and to extract:
  ```
   7za x Synthetic.7z
  ```
  
  CERRA files are large, so they are split into two files. To extract CERRA files just run:
  ```
   7za x CERRA.7z.001
  ```
  7z will find CERRA.7z.002 automatically.

- You can visualize the data using the scripts in [vis](vis).

### Citation
If you find our work useful, please cite:

```
@inproceedings{
eddin2024identifying,
title={Identifying Spatio-Temporal Drivers of Extreme Events},
author={Mohamad Hakam Shams Eddin and Juergen Gall},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=DdKdr4kqxh}
}
```

### Acknowledgments


This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) within the Collaborative Research Centre SFB 1502/1–2022 - [DETECT](https://sfb1502.de/) - [D05](https://sfb1502.de/projects/cluster-d/d05) and by the Federal Ministry of Education and Research (BMBF) under grant no. 01IS24075C RAINA.

### License

The code is released under MIT License. See the [LICENSE](LICENSE) file for details.
