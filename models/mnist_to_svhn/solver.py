import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim
from model import G12, G21
from model import D1, D2
import pdb

class Solver(object):
  def __init__(self, config, svhn_loader, mnist_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.g12_gc = None
        self.g21_gc = None
        self.d1_gc = None
        self.d2_gc = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.d_gc_optimizer = None
        self.use_reconst_loss = config.use_reconst_loss
        self.use_distance_loss = config.use_distance_loss
        self.use_self_distance = config.use_self_distance
        self.num_classes = config.num_classes
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.lambda_distance_A = config.lambda_distance_A
        self.lambda_distance_B = config.lambda_distance_B
        self.config = config
        self.criterionGc = torch.nn.L1Loss()
        self.lambda_gc = config.lambda_gc
        self.build_model()

    def build_model(self):
      """Builds a generator and a discriminator."""
        self.g12 = G12(self.config, conv_dim=self.g_conv_dim)
        self.g21 = G21(self.config, conv_dim=self.g_conv_dim)
        self.d1 = D1(conv_dim=self.d_conv_dim)
        self.d2 = D2(conv_dim=self.d_conv_dim)
        self.d1_gc = D1(conv_dim=self.d_conv_dim)
        self.d2_gc = D2(conv_dim=self.d_conv_dim)

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        d_gc_params = list(self.d1_gc.parameters()) + list(self.d2_gc.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        self.d_gc_optimizer = optim.Adam(d_gc_params, self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
          self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()
            self.d1_gc.cuda()
            self.d2_gc.cuda()

    def merge_images(self, sources, targets, k=10):
      _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
          i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
      """Converts numpy to variable."""
        if torch.cuda.is_available():
          x = x.cuda()
        return Variable(x)

    def to_data(self, x):
      """Converts variable to numpy."""
        if torch.cuda.is_available():
          x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
      """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.d_gc_optimizer.zero_grad()

    def distance(self, A, B):
      return torch.mean(torch.abs(A - B))

    def get_individual_distance_loss(self, A_i, A_j, AB_i, AB_j, A_to_AB):

      distance_in_A = self.distance(A_i, A_j)
        distance_in_AB = self.distance(AB_i, AB_j)

        if self.normalize_distances:
          if A_to_AB:
            distance_in_A = (distance_in_A - self.expectation_A) / self.std_A
                distance_in_AB = (distance_in_AB - self.expectation_B) / self.std_B
            else:
              distance_in_A = (distance_in_A - self.expectation_B) / self.std_B
                distance_in_AB = (distance_in_AB - self.expectation_A) / self.std_A

        return torch.abs(distance_in_A - distance_in_AB)

    def get_self_distances(self, A, AB, A_to_AB=True):

      A_half_1, A_half_2 = torch.chunk(A, 2, dim=2)
        AB_half_1, AB_half_2 = torch.chunk(AB, 2, dim=2)

        l_distance_A = \
            self.get_individual_distance_loss(A_half_1, A_half_2,
                AB_half_1, AB_half_2, A_to_AB)

            return l_distance_A

    def get_distance_losses(self, A, AB, A_to_AB=True ):

      As = torch.split(A, 1)
        ABs = torch.split(AB, 1)

        loss_distance_A = 0.0
        num_pairs = 0
        min_length = len(As)

        for i in range(min_length - 1):
          for j in range(i + 1, min_length):
            num_pairs += 1
                loss_distance_A_ij = \
                    self.get_individual_distance_loss(As[i], As[j],
                        ABs[i], ABs[j], A_to_AB)
                    loss_distance_A += loss_distance_A_ij

        loss_distance_A = loss_distance_A / num_pairs
        return loss_distance_A

    def get_std(self, num_items, vars, expectation):

      num_pairs = 0
        std_sum = 0.0

        # If self distance computed std for top and bottom half
        if self.use_self_distance:
          for i in range(num_items):
            var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                std_sum += np.square(self.as_np(self.distance(var_half_1, var_half_2)) - expectation)
            return np.sqrt(std_sum / num_items)

        # Otherwise compute std for all pairs of images
        for i in range(num_items - 1):
          for j in range(i + 1, num_items):
            num_pairs += 1
                std_sum += np.square(self.as_np(self.distance(vars[i], vars[j])) - expectation)

        return np.sqrt(std_sum / num_pairs)

    def get_expectation(self, num_items, vars):

      num_pairs = 0
        distance_sum = 0.0

        # If self distance computed expectation for top and bottom half
        if self.use_self_distance:
          for i in range(num_items):
            # Split image to top and bottom half
                var_half_1, var_half_2 = torch.chunk(vars[i], 2, dim=2)
                distance_sum += self.as_np(self.distance(var_half_1, var_half_2))
            return distance_sum / num_items

        # Otherwise compute expectation for all pairs of images
        for i in range(num_items - 1):
          for j in range(i + 1, num_items):
            num_pairs += 1
                distance_sum += self.as_np(self.distance(vars[i], vars[j]))

        return distance_sum / num_pairs

    def set_expectation_and_std(self):

      max_items = self.config.max_items

        A_vars = []
        B_vars = []
        num_vars_A = 0
        num_vars_B = 0

        mnist_iter = iter(self.mnist_loader)
        for step in range(len(mnist_iter)):

          if step >= max_items:
            break

          mnist, m_labels = mnist_iter.next()
            A = Variable(mnist, volatile=True)

            if A.size()[0] != self.config.batch_size:
              continue

            A_vars.append(A)
            num_vars_A += 1

        svhn_iter = iter(self.svhn_loader)
        for step in range(len(svhn_iter)):

          if step >= max_items:
            break

          svhn, s_labels = svhn_iter.next()
            B = Variable(svhn, volatile=True)

            if B.size()[0] != self.config.batch_size:
              continue

            B_vars.append(B)
            num_vars_B +=1
        self.expectation_A = self.get_expectation(num_vars_A, A_vars).astype(float)
        self.expectation_B = self.get_expectation(num_vars_B, B_vars).astype(float)
        self.std_A = self.get_std(num_vars_A, A_vars, self.expectation_A).astype(float)
        self.std_B = self.get_std(num_vars_B, B_vars, self.expectation_B).astype(float)

        print('Expectation for dataset A: %f' % self.expectation_A)
        print('Expectation for dataset B: %f' % self.expectation_B)
        print('Std for dataset A: %f' % self.std_A)
        print('Std for dataset B: %f' % self.std_B)


    def as_np(self, data):
      return data.cpu().data.numpy()

    def get_gc_loss(self, AB, AB_gc):
      loss_gc = 0.0

        size = 32

        inv_idx = torch.arange(size-1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc*self.lambda_gc
        #loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc


    def test(self, svhn_test_loader, mnist_test_loader):
      svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        svhn_test_iter = iter(svhn_test_loader)
        mnist_test_iter = iter(mnist_test_loader)
        index = 0

        #fixed_svhn = self.to_var(svhn_test_iter.next()[index])
        #fixed_mnist = self.to_var(mnist_test_iter.next()[index])
        g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(40000))
        g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(40000))
        self.g12 = G12(1, conv_dim=64) 
        self.g21 = G21(1, conv_dim=64) 
        self.g12.load_state_dict(torch.load(g12_path))
        self.g21.load_state_dict(torch.load(g21_path))

        self.g12.cuda()
        self.g21.cuda()

        for i in range(len(svhn_test_iter)):
          fixed_svhn = self.to_var(svhn_test_iter.next()[index])
            fake_mnist = self.g21(fixed_svhn)

            fake_mnist = self.to_data(fake_mnist)
            svhn = self.to_data(fixed_svhn)


            merged = self.merge_images(svhn, fake_mnist)
            path = os.path.join(self.sample_path, 'sample-%d-s-m.png' %(i))
            scipy.misc.imsave(path, merged)
            print ('saved %s' %path)


    def train(self, svhn_test_loader, mnist_test_loader):
      svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter)) -1

        # fixed mnist and svhn for sampling
        svhn_test_iter = iter(svhn_test_loader)
        mnist_test_iter = iter(mnist_test_loader)
        fixed_svhn = self.to_var(svhn_test_iter.next()[0])
        fixed_mnist = self.to_var(mnist_test_iter.next()[0])
        self.normalize_distances = not self.config.unnormalized_distances

        if (self.use_self_distance or self.use_distance_loss) and self.normalize_distances:
          self.set_expectation_and_std()

        for step in range(self.train_iters+1):
          # reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
              mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)

            size = 32
            inv_idx = torch.arange(size-1, -1, -1).long().cuda()

            # load svhn and mnist dataset
            svhn, s_labels = svhn_iter.next()
            svhn_gc = svhn.clone()
            svhn_gc = self.to_var(svhn_gc)
            svhn_gc = torch.index_select(svhn_gc, 2, inv_idx)
            #svhn_gc = self.rot90(svhn_gc, 0)
            svhn, s_labels = self.to_var(svhn), self.to_var(s_labels).long().squeeze()

            mnist, m_labels = mnist_iter.next() 
            mnist_gc = mnist.clone()
            mnist_gc = self.to_var(mnist_gc)
            mnist_gc = torch.index_select(mnist_gc, 2, inv_idx)
            #mnist_gc = self.rot90(mnist_gc, 0)
            mnist, m_labels = self.to_var(mnist), self.to_var(m_labels)

            #============ train D ============#

            # train with real images
            self.reset_grad()
            out = self.d1(mnist)
            d1_loss = torch.mean((out-1)**2)

            out = self.d2(svhn)
            d2_loss = torch.mean((out-1)**2)

            d_mnist_loss = d1_loss
            d_svhn_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_svhn = self.g12(mnist)
            out = self.d2(fake_svhn)
            d2_loss = torch.mean(out**2)

            fake_mnist = self.g21(svhn)
            out = self.d1(fake_mnist)
            d1_loss = torch.mean(out**2)

            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            #===========train gc D==========#
            # train with real images
            self.reset_grad()
            out = self.d1_gc(mnist_gc)
            d1_gc_loss = torch.mean((out-1)**2)

            out = self.d2_gc(svhn_gc)
            d2_gc_loss = torch.mean((out-1)**2)

            d_mnist_gc_loss = d1_gc_loss
            d_svhn_gc_loss = d2_gc_loss
            d_real_gc_loss = d1_gc_loss + d2_gc_loss
            d_real_gc_loss.backward()
            self.d_gc_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_svhn_gc = self.g12(mnist_gc)
            out = self.d2_gc(fake_svhn_gc)
            d2_gc_loss = torch.mean(out**2)

            fake_mnist_gc = self.g21(svhn_gc)
            out = self.d1_gc(fake_mnist_gc)
            d1_gc_loss = torch.mean(out**2)

            d_fake_gc_loss = d1_gc_loss + d2_gc_loss
            d_fake_gc_loss.backward()
            self.d_gc_optimizer.step()


            #============ train G ============#

            # train mnist-svhn-mnist cycle
            self.reset_grad()
            fake_svhn = self.g12(mnist)
            out_svhn = self.d2(fake_svhn)
            reconst_mnist = self.g21(fake_svhn)

            gen_loss_A = torch.mean((out_svhn-1)**2)
            g_loss = gen_loss_A



            if self.use_reconst_loss:
              reconst_loss_A = torch.mean((mnist - reconst_mnist) ** 2)
                g_loss += reconst_loss_A
            if self.use_distance_loss:
              dist_A = self.get_distance_losses(mnist, fake_svhn, A_to_AB=True) * self.lambda_distance_A
                g_loss += dist_A
            elif self.use_self_distance:
                dist_A = self.get_self_distances(mnist, fake_svhn, A_to_AB=True) * self.lambda_distance_A
                g_loss += dist_A


            fake_svhn_gc = self.g12(mnist_gc)
            out_svhn_gc = self.d2_gc(fake_svhn_gc)
            reconst_mnist_gc = self.g21(fake_svhn_gc)

            gen_gc_loss_A = torch.mean((out_svhn_gc-1)**2)
            g_loss += gen_gc_loss_A
            
            if self.use_reconst_loss:
                reconst_gc_loss_A = torch.mean((mnist_gc - reconst_mnist_gc) ** 2)
                g_loss += reconst_gc_loss_A
            #if self.use_distance_loss:
            #    dist_gc_A = self.get_distance_losses(mnist_gc, fake_svhn_gc, A_to_AB=True) * self.lambda_distance_A
            #    g_loss += dist_gc_A
            #elif self.use_self_distance:
            #    dist_gc_A = self.get_self_distances(mnist_gc, fake_svhn_gc, A_to_AB=True) * self.lambda_distance_A
            #    g_loss += dist_gc_A

           
            gc_loss_A = self.get_gc_loss(fake_svhn, fake_svhn_gc)
            g_loss += gc_loss_A


            g_loss.backward()
            self.g_optimizer.step()

            # train svhn-mnist-svhn cycle
            self.reset_grad()
            fake_mnist  = self.g21(svhn)
            out_mnist = self.d1(fake_mnist)
            reconst_svhn = self.g12(fake_mnist)

            gen_loss_B = torch.mean((out_mnist - 1) ** 2)
            g_loss = gen_loss_B

            if self.use_reconst_loss:
                reconst_loss_B = torch.mean((svhn - reconst_svhn) ** 2)
                g_loss += reconst_loss_B

            if self.use_distance_loss:
                dist_B = self.get_distance_losses(svhn, fake_mnist, A_to_AB=False) * self.lambda_distance_B
                g_loss += dist_B
            elif self.use_self_distance:
                dist_B = self.get_self_distances(svhn, fake_mnist, A_to_AB=False) * self.lambda_distance_B
                g_loss += dist_B


            fake_mnist_gc  = self.g21(svhn_gc)
            out_mnist_gc = self.d1_gc(fake_mnist_gc)
            reconst_svhn_gc = self.g12(fake_mnist_gc)

            gen_gc_loss_B = torch.mean((out_mnist_gc - 1) ** 2)
            g_loss += gen_gc_loss_B

            if self.use_reconst_loss:
                reconst_gc_loss_B = torch.mean((svhn_gc - reconst_svhn_gc) ** 2)
                g_loss += reconst_gc_loss_B

            #if self.use_distance_loss:
            #    dist_gc_B = self.get_distance_losses(svhn_gc, fake_mnist_gc, A_to_AB=False) * self.lambda_distance_B
            #    g_loss += dist_gc_B
            #elif self.use_self_distance:
            #    dist_gc_B = self.get_self_distances(svhn_gc, fake_mnist_gc, A_to_AB=False) * self.lambda_distance_B

            gc_loss_B = self.get_gc_loss(fake_mnist, fake_mnist_gc)
            g_loss += gc_loss_B

            g_loss.backward()
            self.g_optimizer.step()

            # print the log info
            if (step+1) % self.log_step == 0:

                print('Step [%d/%d], d_real_loss: %.4f, d_mnist_loss: %.4f, d_svhn_loss: %.4f, '
                    'd_fake_loss: %.4f, gen_loss_A: %.4f, gen_loss_B: %.4f, gc_loss_A: %.4f, gc_loss_B: %.4f'
                      %(step+1, self.train_iters, d_real_loss.item(), d_mnist_loss.item(),
                        d_svhn_loss.item(), d_fake_loss.item(), gen_loss_A.item(),gen_loss_B.item(), gc_loss_A.item(), gc_loss_B.item()))

                if self.use_reconst_loss:
                    print ('reconst_loss_A: %.4f, recons_loss_B: %.4f, ' %
                           (reconst_loss_A.item(), reconst_loss_B.item()))
                if self.use_distance_loss or self.use_self_distance:
                    print  ('dist_loss_A: %.4f, dist_loss_B: %.4f, ' %
                            (dist_A.item(), dist_B.item()))


            # save the sampled images
            if (step+1) % self.sample_step == 0:
                fake_svhn = self.g12(fixed_mnist)
                fake_mnist = self.g21(fixed_svhn)
                
                mnist, fake_mnist = self.to_data(fixed_mnist), self.to_data(fake_mnist)
                svhn , fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)
                
                merged = self.merge_images(mnist, fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(svhn, fake_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-s-m.png' %(step+1))
                scipy.misc.imsave(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % 5000 == 0:
                # save the model parameters for each epoch
                g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(step+1))
                g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, 'd2-%d.pkl' %(step+1))
                d1_gc_path = os.path.join(self.model_path, 'd1_gc-%d.pkl' %(step+1))
                d2_gc_path = os.path.join(self.model_path, 'd2_gc-%d.pkl' %(step+1))
                torch.save(self.g12.state_dict(), g12_path)
                torch.save(self.g21.state_dict(), g21_path)
                torch.save(self.g21_gc.state_dict(), g21_gc_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)
                torch.save(self.d1_gc.state_dict(), d1_gc_path)
                torch.save(self.d2_gc.state_dict(), d2_gc_path)

