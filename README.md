# GcGAN: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping

### Paper

#### [H. Fu, M. Gong, C. Wang, K. Batmanghelich, K. Zhang and D. Tao: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping.](https://arxiv.org/abs/1809.05852) Accepeted to CVPR 2019.
[Huan Fu](https://hufu6371.github.io/huanfu/)  and [Mingming Gong](https://mingming-gong.github.io/) contribute equally to the project and the paper.

Pretrained Models will avaliable soon.


### Introduction
The codes have been modified from [CycleGAN-Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN), and have been tested successfully on CentOS release 6.9, Cuda 9.1, Tesla V100, Anaconda python3, Pytorch 0.4.1. 

The codes are only for research purposes. You may also need to follow the instructions of [CycleGAN-Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DistanceGAN](https://github.com/sagiebenaim/DistanceGAN).

### Usage
1. Clone the respository:
```
git clone https://github.com/hufu6371/GcGAN.git
cd $GcGAN_ROOT
```
2. Download the dataset (Cityscapes):
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
Download the pre-trained caffe model following the instructions stated in CycleGAN and Pix2Pix
cd evaluation/parsing2city
python evaluate.py
```

### Tips
1. For city2parsing, we do not apply scale augmentation. (--loadSize 128, --fineSize 128)
2. For parsing2city, we set the hyperparameter "identity" as 0.3. For others, the hyperparameter "identity" is 0.5.
3. For horse2zebra, winter2summer, summer2winter, we do not share parameters for G_{XY} and G_{\tilde{X}\tilde{Y}}. (--model gc_gan_cross)
4. For svhn2mnist, the training scripts are located in the "models/mnist_to_svhn" directory. Please follow the instructions stated in DistanceGAN for training.
5. The evaluation scripts are located in the "$GcGAN_ROOT/evalutation" directory.
6. All the models are trained on 4 GPU cards.


### Citation
```
@inproceedings{FuCVPR19-GcGAN,
  TITLE = {{Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping}},
  AUTHOR = {Fu, Huan and Gong, Mingming and Wang, Chaohui and Batmanghelich, Kayhan and Zhang, Kun and Tao, Dacheng},
  BOOKTITLE = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  YEAR = {2019}
}
```
### Contact
Huan Fu: hufu6371@uni.sydney.edu.au


