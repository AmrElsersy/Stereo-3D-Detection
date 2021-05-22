[![Made withPython](https://img.shields.io/badge/Made%20with-python-407eaf?style=for-the-badge&logo=python)](https://www.python.org/)
[![Made withPytorch](https://img.shields.io/badge/Made%20with-pytorch-ee4c2c?style=for-the-badge&logo=pytorch)](https://www.pytorch.org/)
[![Made withCuda](https://img.shields.io/badge/Made%20with-cuda-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)
[![Made withAnaconda](https://img.shields.io/badge/Made%20with-anaconda-43b049?style=for-the-badge&logo=anaconda)](https://www.anaconda.com/) <br>
![Supports_windows](https://img.shields.io/badge/windows-0078D6?style=for-the-badge&logo=windows)
![Supports_linux](https://img.shields.io/badge/linux-white?style=for-the-badge&logo=linux)
![Supports_macos](https://img.shields.io/badge/macos-black?style=for-the-badge&logo=macos)

# 3D Detection Stereo Based

This repository containts a real time **3D depth estmiation** using stereo camera on [KITTI Benchmark](http://www.cvlibs.net/datasets/kitti/)

## Dependencies

- [CUDA >= 10.0](https://developer.nvidia.com/Cuda-Toolkit)
- [Pytorch >= 1.0](https://pytorch.org/)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/)
- Linux/Mac/Widnows

## Installation

NOTE: this installlation is compatible with linux only, nearly will support windows

1. you must install CUDA local on your system environment, follow this [link](https://developer.nvidia.com/Cuda-downloads)
2. you must instakk cuDNN local in your system environment, follow this [link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
3. you should build a specific environment so we recommend to install [anaconda](https://docs.anaconda.com/anaconda/install/)
4. Install the dependencies for demos and visualizaiotn
	
   - Open your anaconda terminal
   - Create new conda enviroment with python 3.8.5 by running this command ```conda create --name obj_det python=3.8.5```
   - Activate your enviroment ```conda activate obj_det```
   - Install dependencies following this commands
   	
	```shell script
		conda install -c pytorch torchvision=0.8.2
		pip install PyQt5 vtk tqdm matplotlib==3.3.3 easydict==1.9
		pip install mayavi
		conda install scikit-image shapely
		conda install -c conda-forge opencv
	```
	
   - Then Navigate to Models/AnyNet/models/spn_t1 to activate spn layer 
   - If you are windows user, open git bash cmd and activate the enviroment and run`sh make.sh`
   - If you are Linuex user, open terminal and activate the enviroment and run `./make.sh` 
   
## Dataset Preparation

You need to make data directory first and construct dataset as following

```
Stereo-3D-Detection
├── checkpoints
├── data
│   ├── kitti
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & image_3
│   │   │── testing
├── Models
├── utils_classes
├── .
├── .
```
## Checkpoints Preparation

You can download all checkpoints from this [Drive](https://drive.google.com/drive/folders/1QOAIldySCMdQuJ99SOOAmz-ckgPo-XgM?usp=sharing)

```
Stereo-3D-Detection
├── checkpoints
│   ├── anynet.tar
│   ├── sfa.pth
├── data
├── Models
├── utils_classes
├── .
├── .
```

## Demo

- To go from stereo to 3D object detection
```shell script
python sfa_demo.py
```
Note: you can navigate between images by pressing any key, and to exit press ESC

- To generate demo video, be sure you adjusted the path in [`sfa_demo.py`](./sfa_demo.py) then run:
```shell script
python sfa_demo.py --generate_video
```

## Evaluation

1. you need To generate predictions pickle file by running:
```shell script
python sfa_demo.py --generate_pickle
```
2. Then, to evaluate the resulted prediction run [`evaluation.py`](./evaluation.py)
```shell script
python evaluation.py
```
