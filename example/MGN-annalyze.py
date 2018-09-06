# coding=utf-8

from __future__ import absolute_import
import sys
sys.path.insert(0,'.')
import torch
import argparse
import sys,os
from analysis.PytorchA import analyse
from analysis.utils import save_csv
from torch.autograd import Variable
import torch.nn as nn
import sys
sys.path.insert(0,'.')
from option import args as configs
from model import mgn
from torchvision.models import resnet
"""
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path class_name shape`
- The path is the python file path which contaning your class.
- The class_name is the class name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be: batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py tmp/pytorch_analysis_test.py ResNet218 1,3,224,224`
"""


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--path',help='python file location',default = "MGN_analysis_example.py",type=str)
    parser.add_argument('--name',help='the class name or instance name in your python file',default = "mgn",type=str)
    parser.add_argument('--shape',help='input shape of the network(split by comma `,`), image shape should be: batch,c,h,w',type=str)
    parser.add_argument('--out',help='path to save the csv file',default='pytorch_analyse.csv',type=str)
    parser.add_argument('--class_args',help='args to init the class in python file',default='net',type=str)

    args=parser.parse_args()
    # net = mgn.MGN(configs)
    # # print(net)
    # # net.load('/home/wdd/Work',)
    # # net = mgn.MGN()
 
    # # state_dict = torch.load('/home/wdd/Work/Pytorch/pytorch-caffe-darknet-convert-master/torch_model/model_160_max.pt')
    # checkpoint = torch.load('/home/shining/Projects/github-projects/pytorch-project/nn_tools/model_100.pt')
    # net.load_state_dict(checkpoint)
    # name = 'MGN'
    # # net = inception_v3(True, transform_input=False)
    # net.eval()
    name='resnet18'
    net=resnet.resnet18()
    net.eval()
    #input=Variable(torch.ones([1,3,224,224]))


    x = torch.rand( [1,3,224, 224])
    blob_dict, layers = analyse(net, x)
    save_csv(layers, args.out)


