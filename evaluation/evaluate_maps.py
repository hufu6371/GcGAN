import os
import sys
import cv2
import glob
import numpy as np
import pdb

reals = glob.glob('/path/to/GcGAN/datasets/maps/testB/*jpg')
fakes = glob.glob('/path/to/results/photo2map/test_latest/images/*fake_B.png')

reals = sorted(reals)
fakes = sorted(fakes)

num_imgs = len(reals)
corr_count = 0.0
pix_count = 0.0

RMSE = 0.0
#RMSE & Pix acc
for i in range(num_imgs):

    print('{0}'.format(i))
    real = cv2.imread(reals[i])
    fake = cv2.imread(fakes[i]) 

    real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
    fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

    real = real.astype(np.float32)
    fake = fake.astype(np.float32)
    diff = np.abs(real - fake)

    max_diff = np.max(diff, axis=2)
    #delta = 5.0
    correct = max_diff < 5
    corr_count = corr_count + np.sum(correct)
    pix_count = pix_count + 256**2

    diff = (diff**2)/(256**2)
    diff = np.sum(diff)
    rmse = np.sqrt(diff)
    RMSE = RMSE + rmse

RMSE = RMSE/num_imgs
acc = corr_count/pix_count

print('rmse: {0}, acc: {1}'.format(RMSE, acc))
