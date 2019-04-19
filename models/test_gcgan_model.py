import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import random
import math
import sys
import pdb

class TestGcGANModel(BaseModel):
    def name(self):
        return 'TestGcGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.netG_AB = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG_AB, 'G_AB', which_epoch)
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_AB)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_image_paths(self):
        return self.image_paths

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.fineSize
        inv_idx = torch.arange(size-1, -1, -1).long().cuda()
        if direction == 0:
          tensor = torch.index_select(tensor, 3, inv_idx)
        else:
          tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        #real_gc_A = util.tensor2im(self.real_gc_A.data)
        #real_gc_B = util.tensor2im(self.real_gc_B.data)

        fake_B = util.tensor2im(self.fake_B)
        fake_gc_B = util.tensor2im(self.fake_gc_B)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('fake_gc_B', fake_gc_B)])
        return ret_visuals

    def test(self):
        #self.netG_AB.eval()
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        size = self.opt.fineSize

        if self.opt.geometry == 'rot':
          self.real_gc_A = self.rot90(input_A, 0)
          self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
          inv_idx = torch.arange(size-1, -1, -1).long().cuda()
          self.real_gc_A = Variable(torch.index_select(input_A, 2, inv_idx))
          self.real_gc_B = Variable(torch.index_select(input_B, 2, inv_idx))
        else:
          raise ValueError("Geometry transformation function [%s] not recognized." % opt.geometry)

        self.fake_B = self.netG_AB.forward(self.real_A).data
        self.fake_gc_B = self.netG_AB.forward(self.real_gc_A).data
