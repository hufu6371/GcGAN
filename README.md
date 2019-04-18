# GcGAN: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping

### Paper

#### [H. Fu, M. Gong, C. Wang, K. Batmanghelich, K. Zhang and D. Tao: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping. Accepeted to CVPR 2019.](https://arxiv.org/abs/1809.05852)
Huan Fu and Mingming Gong contribute equally to the project and the paper.


### Introduction
The shared code is a Pytorch implemention of our CVPR19 paper (GcGAN), and is modified from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN). The code has been tested successfully on CentOS release 6.9, Cuda 9.1, Tesla V100, Anaconda python3, Pytorch 0.4.1. 

This code is only for research purposes. You may also need to follow the instructions of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN).

### Usage
1. Clone the respository:
```
git clone https://github.com/hufu6371/DORN.git
```
2. Build and link to pycaffe:
```
cd $DORN_ROOT
edit Makefile.config
build pycaffe
export PYTHONPATH=$DORN_ROOT/python:$DORN_ROOT/pylayer:$PYTHONPATH
```
3. Download our pretrained models:
```
mv cvpr_kitti.caffemodel $DORN_ROOT/models/KITTI/
mv cvpr_nyuv2.caffemodel $DORN_ROOT/models/NYUV2/
```
4. Demo (KITTI and NYUV2):  
```
python demo_kitti.py --filename=./data/KITTI/demo_01.png --outputroot=./result/KITTI
python demo_nyuv2.py --filename=./data/NYUV2/demo_01.png --outputroot=./result/NYUV2
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


