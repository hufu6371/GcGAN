import os
import glob
import numpy as np
import cv2
import pdb

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

reals = glob.glob('./datasets/cityscapes/testB/*jpg')
fakes = glob.glob('./results/city2parsing_rot/test_latest/images/*fake_B.png')
#reals = glob.glob('./results/sat2map_pretrained/latest_test/images/fake_B/*B.jpg')
#fakes = glob.glob('./results/sat2map_pretrained/latest_test/images/fake_B/*A.png')
reals = sorted(reals)
fakes = sorted(fakes)
num_imgs = len(reals)

CM = np.zeros((19,19), dtype=np.float32)
# test
for i in range(num_imgs):
    print('{0}'.format(i))
    real = cv2.imread(reals[i])
    fake = cv2.imread(fakes[i]) 

    real = cv2.resize(real, (128, 128), interpolation=cv2.INTER_NEAREST)
    fake = cv2.resize(fake, (128, 128), interpolation=cv2.INTER_NEAREST)

    pred = fake
    label = real


    label_dis = np.zeros((20, 128, 128), dtype=np.float32)
    pred_dis = np.zeros((20, 128, 128), dtype=np.float32)

    for j in range(20):
        color = labels[j]['color']
        label_diff = np.abs(label - color)
        pred_diff = np.abs(pred - color)

        label_diff = np.sum(label_diff, axis=2)
        pred_diff = np.sum(pred_diff, axis=2)

        label_dis[j,:,:] = label_diff
        pred_dis[j,:,:] = pred_diff

    label_id = np.argmin(label_dis, axis=0)
    pred_id = np.argmin(pred_dis, axis=0)

    for j in range(19):
        coord = np.where(label_id == j)
        pred_j = pred_id[coord]
        for k in range(19):
            CM[j,k] = CM[j, k] + np.sum(pred_j == k)


pix_acc = 0
mean_acc = 0
mean_IoU = 0

count = 0
for i in range(19):
    count = count + CM[i, i]
pix_acc = count / np.sum(CM)


count = 0
for i in range(19):
    temp = CM[i, :]
    count = count + CM[i,i]/(np.sum(temp) + 1e-6)
mean_acc = count/19

count = 0
for i in range(19):
    temp_0 = CM[i, :]
    temp_1 = CM[:, i]
    count = count + CM[i, i]/(np.sum(temp_0) + np.sum(temp_1) - CM[i, i] + 1e-6)

mean_IoU = count/19

print('{0}, {1}, {2}'.format(pix_acc, mean_acc, mean_IoU))
