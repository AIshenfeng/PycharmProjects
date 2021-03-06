#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
from add import add
import numpy as np
import math
import chainer.computational_graph as c

class ResidualB(chainer.Chain):
    def __init__(self, in_size, out_size):
        super(ResidualB, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, out_size/2, 1, 1, 0),
            bn2=L.BatchNormalization(out_size/2),
            conv2=L.Convolution2D(out_size/2, out_size/2, 3, 1, 1),
            bn3=L.BatchNormalization(out_size/2),
            conv3=L.Convolution2D(out_size/2, out_size, 1, 1, 0),

            conv4=L.Convolution2D(in_size, out_size, 1, 1, 0),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))
        h2 = self.conv4(x)

        return h1 + h2

class ResidualA(chainer.Chain):
    def __init__(self, in_size, out_size):
        super(ResidualA, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, out_size/2, 1, 1, 0),
            bn2=L.BatchNormalization(out_size/2),
            conv2=L.Convolution2D(out_size/2, out_size/2, 3, 1, 1),
            bn3=L.BatchNormalization(out_size/2),
            conv3=L.Convolution2D(out_size/2, out_size, 1, 1, 0),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x)))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))

        return h1 + x

class Hg(chainer.Chain):
    def __init__(self, train):
        self.train = train
        super(Hg, self).__init__(
            up1=ResidualA(256,256),
            low1=ResidualA(256,256),
            up2=ResidualA(256,256),
            low2=ResidualA(256,256),
            up3=ResidualA(256,256),
            low3=ResidualA(256,256),
            up4=ResidualA(256,256),
            low4=ResidualA(256,256),
            mid=ResidualA(256,256),
            low5=ResidualA(256,256),
            low6=ResidualA(256,256),
            low7=ResidualA(256,256),
            low8=ResidualA(256,256),
        )

    def __call__(self, x, train):
        h1 = self.up1(x, self.train)
        h = F.max_pooling_2d(x, 2, stride=2)
        h = self.low1(h, self.train)
        h2 = self.up2(h, self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.low2(h, self.train)
        h3 = self.up3(h, self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.low3(h, self.train)
        h4 = self.up4(h, self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.low4(h, self.train)
        h = self.mid(h, self.train)
        h = self.low5(h, self.train)
        h = F.unpooling_2d(h, 2, outsize=(8,8))
        h = h + h4
        h = self.low6(h, self.train)
        h = F.unpooling_2d(h, 2, outsize=(16,16))
        h = h + h3
        h = self.low7(h, self.train)
        h = F.unpooling_2d(h, 2, outsize=(32,32))
        h = h + h2
        h = self.low8(h, self.train)
        h = F.unpooling_2d(h, 2, outsize=(64,64))
        h = h + h1

        return h

class Hourglass(chainer.Chain):

 
    def __init__(self, args):

        self.CLASSES = args.n_joints
        self.IN_SIZE = args.inputRes
        self.OUTPUT_RES = args.outputRes
        self.train = True

        super(Hourglass, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3),
            bn1=L.BatchNormalization(64),
            res2=ResidualB(64,128),
            res3=ResidualA(128,128),
            res4=ResidualB(128,256),
            hg1=Hg(self.train),
            res5=ResidualA(256,256),
            conv6=L.Convolution2D(256, 256, 1, 1, 0),
            bn6=L.BatchNormalization(256),
            conv7=L.Convolution2D(256, 256, 1, 1, 0),
            inter_conv1=L.Convolution2D(256, self.CLASSES, 1, 1, 0),
            inter_conv2=L.Convolution2D(self.CLASSES, 256, 1, 1, 0),
            hg2=Hg(self.train),
            res8=ResidualA(256,256),
            conv9=L.Convolution2D(256, 256, 1, 1, 0),
            bn9=L.BatchNormalization(256),
            final_conv1=L.Convolution2D(256, self.CLASSES, 1, 1, 0),
        )

    def clear(self):
        self.loss = None
        self.accuracy = None
 
    def __call__(self, x, t):

        self.clear()
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.res2(h, self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        inter = F.identity(h)
        h = self.hg1(h, self.train)

        # Residual layers at output resolution
        h = self.res5(h, self.train)
        h = F.relu(self.bn6(self.conv6(h)))
        ll_ = self.conv7(h) 

        # Predicted heatmaps
        tmpOut = self.inter_conv1(h)
        tmpOut_ = self.inter_conv2(tmpOut)
           
        h = add(ll_, tmpOut_, inter)
        h = self.hg2(h, self.train)
        h = self.res8(h, self.train)
        h = F.relu(self.bn9(self.conv9(h)))
        h = self.final_conv1(h)
        h = F.concat((tmpOut, h))

        t = F.concat((t, t))
         
        self.loss = F.mean_squared_error(h,t)
        if self.train:
            return self.loss
        else:
            self.pred = h
            return self.pred
