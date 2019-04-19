import os
import sys
import cv2
import caffe
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from util import *
from cityscapes import cityscapes
import glob
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--caffemodel_dir", type=str, default='./caffemodel/', help="Where the FCN-8s caffemodel stored")
parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")
args = parser.parse_args()

labels = [{'name':'road', 'catId':0, 'color': (128, 64, 128)},
          {'name':'sidewalk', 'catId':1, 'color': (244, 35, 232)},
          {'name':'building', 'catId':2, 'color': (70, 70, 70)},
          {'name':'wall', 'catId':3, 'color': (102, 102, 156)},
          {'name':'fence', 'catId':4, 'color': (190, 153, 153)},
          {'name':'pole', 'catId':5, 'color': (153, 153, 153)},
          {'name':'traffic_light', 'catId':6, 'color': (250, 170, 30)},
          {'name':'traffic_sign', 'catId':7, 'color': (220, 220, 0)},
          {'name':'vegetation', 'catId':8, 'color': (107, 142, 35)},
          {'name':'terrain', 'catId':9, 'color': (152, 251, 152)},
          {'name':'sky', 'catId':10, 'color': (70, 130, 180)},
          {'name':'person', 'catId':11, 'color': (220, 20, 60)},
          {'name':'rider', 'catId':12, 'color': (255, 0, 0)},
          {'name':'car', 'catId':13, 'color': (0, 0, 142)},
          {'name':'truck', 'catId':14, 'color': (0, 0, 70)},
          {'name':'bus', 'catId':15, 'color': (0, 60, 100)},
          {'name':'train', 'catId':16, 'color': (0, 80, 100)},
          {'name':'motorcycle', 'catId':17, 'color': (0, 0, 230)},
          {'name':'bicycle', 'catId':18, 'color': (119, 11, 32)},
          {'name':'ignore', 'catId':19, 'color': (0, 0, 0)}]


mean_value = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)


def rgb2label(label_rgb):
    label_rgb = label_rgb[:,:,::-1]
    label = np.zeros((label_rgb.shape[0], label_rgb.shape[1]))
    label_dis = np.zeros((20, label_rgb.shape[0], label_rgb.shape[1]), dtype=np.float32)
    for j in xrange(20):
        color = labels[j]['color']
        label_diff = np.abs(label_rgb - color)
        label_diff = np.sum(label_diff, axis=2)
        label_dis[j,:,:] = label_diff

    label = np.argmin(label_dis, axis=0)
    return label

def main():

    n_cl = 19
    #label_frames = CS.list_label_frames(args.split)
    label_frames = glob.glob('/path/to/GcGAN/datasets/cityscapes/testB/*jpg');
    imgs = glob.glob('/path/to/GcGAN/results/parsing2city/test_latest/images/*fake_B.png');
    label_frames = sorted(label_frames)
    imgs = sorted(imgs)
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(args.caffemodel_dir + '/deploy.prototxt',
                    args.caffemodel_dir + 'fcn-8s-cityscapes.caffemodel',
                    caffe.TEST)

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]

        label_color = cv2.imread(label_frames[i])
        img = cv2.imread(imgs[i]);

        label = rgb2label(label_color)
        label = cv2.resize(label, (2048, 1024), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32)
        img = img - mean_value
        img = img.transpose(2,0,1)

        out = segrun(net, img)
        #out = cv2.resize(out, (256, 256), interpolation=cv2.INTER_NEAREST)
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)

    print 'pix acc: {0},  mean acc: {1},  mean IoU: {2}'.format(mean_pixel_acc, mean_class_acc, mean_class_iou)

main()
