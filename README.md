[![Made withPython](https://img.shields.io/badge/Made%20with-python-407eaf?style=for-the-badge&logo=python)](https://www.python.org/)
[![Made withPytorch](https://img.shields.io/badge/Made%20with-pytorch-ee4c2c?style=for-the-badge&logo=pytorch)](https://www.pytorch.org/)
[![Made withCuda](https://img.shields.io/badge/Made%20with-cuda-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-downloads)
[![Made withAnaconda](https://img.shields.io/badge/Made%20with-anaconda-43b049?style=for-the-badge&logo=anaconda)](https://docs.anaconda.com/anaconda/install/) <br>
![Supports_windows](https://img.shields.io/badge/windows-0078D6?style=for-the-badge&logo=windows)
![Supports_linux](https://img.shields.io/badge/linux-white?style=for-the-badge&logo=linux)
![Supports_macos](https://img.shields.io/badge/macos-black?style=for-the-badge&logo=macos)

# 3D Detection Stereo Based

This repository containts a real time **3D depth estmiation** using stereo camera on [KITTI Benchmark](http://www.cvlibs.net/datasets/kitti/)

<hr>

![ezgif com-gif-maker](https://user-images.githubusercontent.com/35613645/127658809-faf09e66-48de-4c60-8733-8b6842529bf6.gif)


## Dependencies

- [CUDA >= 10.0](https://developer.nvidia.com/Cuda-Toolkit)
- [Pytorch >= 1.0](https://pytorch.org/)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/)

<hr>

## Installation	
- Open your anaconda terminal
- Create new conda enviroment with python 3.8.5 by running this command 
```
conda create --name obj_det python=3.8.5
```
- Activate your enviroment `conda activate obj_det`
- Install dependencies following this commands
 
```shell script
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   pip install PyQt5 vtk tqdm matplotlib==3.3.3 easydict==1.9 tensorboard
   pip install mayavi
   conda install scikit-image shapely
   conda install -c conda-forge opencv
```
	
- Then Navigate to `Models/AnyNet/models/spn_t1` to activate SPN layer.
- For windows, Open git bash terminal, Activate the enviroment then run `sh make.sh`
- For Linux  , Open the terminal, Activate the enviroment then run  `./make.sh` 
<hr>
   
## Dataset Preparation

You need to make data directory first and construct dataset folder as following :

```
Stereo-3D-Detection
├── checkpoints
├── data
│   ├── <dataset folder>
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & image_3
│   │   │── testing
├── Models
├── utils_classes
├── .
├── .
```

<hr>

## Checkpoints Preparation

You can download all checkpoints from this [Drive](https://drive.google.com/drive/folders/1QOAIldySCMdQuJ99SOOAmz-ckgPo-XgM?usp=sharing)

```
Stereo-3D-Detection
├── checkpoints
│   ├── anynet.tar
│   ├── sfa.pth
├── data
├── .
```

<hr>

## Demo

- To go from stereo to 3D object detection
```shell script
python demo.py
```
#### OPTIONS:

- Choose a mode:
   - Regular mode, Add no option -> You can navigate between images by pressing any key, and to exit press ESC
   - To evaluate the pipeline, Add `--evaluate`
   - To generate a video, Be sure that you have adjusted the path in [`demo.py`](https://github.com/AmrElsersy/Stereo-3D-Detection/blob/aeb7f0b0b15da3ed7534f3b7346aa01011a71950/demo.py#L115), Then add `--generate_video`
      - To generate the video with bev view, Add `--with_bev`
- To Spasify How often to print time durations In case of video, Add `--print_freq <no>`
- Data path is set to `data/kitti` by default, To change it add `--data_path <datapath>`
- Anynet checkpoint path is set to `checkpoints/anynet.tar` by default, To change it add `--pretrained_anynet <checkpoint path>`
- SFA checkpoint path is set to `checkpoints/sfa.pth` by default, To change it add `--pretrained_sfa <checkpoint path>`

<hr>

## Training

### Train Anynet Model
You have to organize your own dataset as the following format
```
Stereo-3D-Detection
├── checkpoints
├── data
│   ├── <dataset>
│   │   │── training
│   │   │   ├──disp_occ_0 & image_2 & image_3
├── .
├── .
```
Incase of `.npy` Disparities:
```
Stereo-3D-Detection
├── checkpoints
├── data
│   ├── <dataset>
│   │   │── training
│   │   │   ├──disp_occ_0_npy & image_2 & image_3
├── .
├── .
```
Command:
```shell script
python train_anynet.py  --maxdisp <default: 192> \ 
                        --datatype <2012/2015/other> \
                        --data_path <datapath> \
                        --save_path <default: 'results/train_anynet'> \
                        --pretrained_path <pretrained checkpoint path> \
                        --train_file <train file path if exist> \
                        --validation_file <validation file path> \
                        --with_spn <Activates Anynet last layer [RECOMMENDED]>
```
###### OPTIONS:
- If disparity files are .npy format, Add `--load_npy` 
- If you want to evaluate your pretrained checkpoint without training, Add `--evaluate`
- In case of datatype 2012/2015, Add `--split_file`
- In case of training datatype of other, and want to train on specefic file names `--train_file`
- In case of testing datatype of other, and want to validate/test on specefic file names `--validation_file`
- If you want to start from specefic index, you can use this flag `--index <no>`

<hr>

##### To train on Kitti Object:  
```shell script
python train_anynet.py  --maxdisp 192 \
                        --datatype other \
                        --data_path data/kitti/ \
                        --pretrained_path checkpoints/anynet.tar \
                        --train_file data/kitti/imagesets/train.txt \
                        --validation_file data/kitti/imagesets/val.txt \
                        --with_spn --load_npy
``` 
##### To train on Kitti 2015:
```shell script
python train_anynet.py  --maxdisp 192 \
                        --datatype 2015 \
                        --save_path results/kitti2015 \
                        --data_path data/path-to-kitti2015/training/ \
                        --pretrained_path checkpoints/anynet.tar  \
                        --split_file data/path-to-kitti2015/split.txt \
                        --with_spn
```
##### To train on Kitti 2012:
```shell script
python train_anynet.py  --maxdisp 192 \
                        --datatype 2012 \
                        --save_path results/kitti2012 \
                        --data_path data/path-to-kitti2012/training/ \
                        --pretrained_path checkpoints/anynet.tar  \
                        --split_file data/path-to-kitti2012/split.txt \
                        --with_spn
```

<hr>

### Train SFA Model
You have to organize your own dataset as the following format
```
Stereo-3D-Detection
├── checkpoints
├── data
│   ├── <dataset>
│   │   │── ImageSets
│   │   │   ├── train.txt & test.txt & val.txt
│   │   │── training
│   │   │   ├── velodyne & calib & label_2
├── .
├── .
```
Command:
```shell script
python train_sfa.py
```
###### OPTIONS:
- By default data path is set to 'data/kitti', To change it use `--data_path <datapath>`
- By default pretrained path is set to 'checkpoints/sfa.pth', To change it use `--pretrained_path <pretrained checkpoint path>`
- By default the name used for saved files is set to 'fpn_resnet_18', To change it use `--saved_fn <name>`
- By default the batch size is set to 2, To change it use `--batch_size <no>`
- You can adjust how often to print/save checkpoint/ Tensorboard freq through these flags `--print_freq <no>` `--checkpoint_freq <no>` `--tensorboard_freq <no>`
- If you want to evaluate your pretrained checkpoint without training, Add `--evaluate`

NOTE: The text files in ImageSets are split files, you can find the split files of Kitti object dataset [here](https://drive.google.com/drive/folders/1acbucPC1Sni57j8KUUQdvmyl0E-sX_aB?usp=sharing)

<hr>

## KITTI Evaluation
To evaluate the model on testing data on KITTI submition
```
python demo.py --testing --save_objects objects.pkl

```
```
python submit_to_kitti.py
```
Then compress the label_2 folder in testing directory and submit it on KITTI 

<hr>

## Utils

### Generate disparity

To generate disparity from point cloud, Be sure your folder structure is like this:
```
├── checkpoints
├── data
│   ├── <dataset>
│   │   │── training
│   │   │   ├── image_2 & velodyne & calib
├── .
├── .
```
Then run this command:
```
python ./tools/generate_disp.py --datapath <datapath>
```
#### OPTIONS:
- There is `--limit <no>` flag if you dont to limit who much of the dataset you want to convert 
- Data path is set to `data/kitti/training` by default, To change it add `--data_path <datapath>`
NOTE: When specifiying your data path make it relative to Stereo-3D-Detection directory

This will generate 2 disaprity folders at the data path location `generated_disp/disp_occ_0` and `generated_disp/disp_occ_0_npy`, you can use any. But we recommend to use `.npy` files 

<hr>

### Generate disparity/depth

To generate point cloud from disparity/depth, Be sure your folder structure is like this:
```
├── checkpoints
├── data
│   ├── <dataset>
│   │   │── training
│   │   │   ├── disp_occ_0 & calib
├── .
├── .
```
Then run this command:
```
python ./tools/generate_lidar.py --datapath <datapath>
```

#### OPTIONS:
- If your converting depth images, use this flag `--is_depth`
- Data path is set to `data/kitti/training` by default, To change it add `--data_path <datapath>`
- There is `--limit <no>` flag if you dont to limit who much of the dataset you want to convert
NOTE: When specifiying your data path make it relative to Stereo-3D-Detection directory

This will generate a velodyne folder at the data path location `generated_lidar/velodyne` 

<hr>

### Visualize point cloud

To View a point cloud file `.bin`, You can use View_bin.py file in tools folder. Just copy it in point cloud folder, then run:
```
python view_bin.py
```
#### OPTIONS:
- By default, it will show you image `000000.bin`, but you can specify the image you want by using `--image <image no>` flag

<hr>

### Profiling

We added another way to track how long each function take and how frequent it have been called. you can see this by running :
```
sh profiling.sh
```
Then you will find your results in new generated file `profiling.txt`. 

<hr>

## Citation
```
@article{wang2018anytime,
  title={Anytime Stereo Image Depth Estimation on Mobile Devices},
  author={Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and Van Der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1810.11408},
  year={2018}
}
```

```bash
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection}},
  year =         {2020}
}
```

