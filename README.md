# GcGAN: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping

### Paper

#### [H. Fu, M. Gong, C. Wang, K. Batmanghelich, K. Zhang and D. Tao: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping. Accepeted to CVPR 2019.](https://arxiv.org/abs/1809.05852)
Huan Fu and Mingming Gong contribute equally to the project and the paper.


### Introduction
The codes have been modified from [CycleGAN-Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN), and have been tested successfully on CentOS release 6.9, Cuda 9.1, Tesla V100, Anaconda python3, Pytorch 0.4.1. 

The codes are only for research purposes. You may also need to follow the instructions of [CycleGAN-Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN).

### Usage
1. Clone the respository:
```
git clone https://github.com/hufu6371/GcGAN.git
cd $GcGAN_ROOT
```
2. Download dataset (Cityscapes):
```
sh ./scripts/download_cyclegan_dataset.sh cityscapes
```
3. Traning and Test (parsing2city):
```
sh ./train_gcgan.sh
sh ./test_gcgan.sh
```
4. Evaluation (parsing2city):  
```
Install pycaffe
Download the pre-trained FCN caffe model following the instructions stated in CycleGAN and Pix2Pix
cd evaluation/parsing2city
python evaluate.py
```

### Pretrained models
1. [KITTI](https://drive.google.com/open?id=180QRn5su1Yf5d-WNqE0jELPNuOpQMjNR)
2. [NYUV2](https://drive.google.com/file/d/1PkxkzWwZthjnJGtaPlTS5qTrj-Tka7eX/view?usp=sharing)

### Scores on the evaluation servers
1. [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
2. [ScanNet](http://dovahkiin.stanford.edu/adai/)

### Results on ScanNet
The evaluation scripts and the groundtruth depth maps for KITTI and NYU Depth v2 are contained in the zip files. You may also need to download the predictions from [Eigen et al.](https://cs.nyu.edu/~deigen/depth/) for the center cropping used in our evaluation scripts.
1. [ScanNet](https://drive.google.com/file/d/12EB_UrmNQZj8VvEUVVxwl1VBQFPB9hdv/view?usp=sharing)
2. [KITTI](https://drive.google.com/open?id=18z_FpbHWmU-tX19n2FWQMwpzCmuuOsMb)
3. [NYU Depth v2](https://drive.google.com/open?id=1uRqOkCbJLwHWyx4oz19N6MQgrOSZQo6H)

### Citation
```
@inproceedings{FuCVPR18-DORN,
  TITLE = {{Deep Ordinal Regression Network for Monocular Depth Estimation}},
  AUTHOR = {Fu, Huan and Gong, Mingming and Wang, Chaohui and Batmanghelich, Kayhan and Tao, Dacheng},
  BOOKTITLE = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  YEAR = {2018}
}
```
### Contact
Huan Fu: hufu6371@uni.sydney.edu.au


