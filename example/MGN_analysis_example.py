import sys
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from torchvision.models import resnet
import pytorch_analyser
from option import args
from model import mgn

if __name__=='__main__':
        # ckpt = torch.load(self.dir + '/map_log.pt')
    net = mgn.MGN(args)
    # print(net)
    # net.load('/home/wdd/Work',)
    # net = mgn.MGN()
 
    # state_dict = torch.load('/home/wdd/Work/Pytorch/pytorch-caffe-darknet-convert-master/torch_model/model_160_max.pt')
    checkpoint = torch.load('/home/shining/Projects/github-projects/pytorch-project/nn_tools/model_100.pt')
    net.load_state_dict(checkpoint)


    # net = Model(num_classes=2220)
    # # print ('person_ReID:', m)
    # state_dict = torch.load('/home/wdd/Work/Pytorch/pytorch-caffe-darknet-convert-master/torch_model/Ep600_ckpt.pth')
    # # print ("state_dict: ", state_dict)
    # state_dict = state_dict["state_dicts"][0]
    # net.load_state_dict(state_dict)

    name = 'MGN'
    # net = inception_v3(True, transform_input=False)
    net.eval()
    input_tensor=torch.ones(1,3,384,128)
    blob_dict, tracked_layers=pytorch_analyser.analyse(net,input_tensor)
    pytorch_analyser.save_csv(tracked_layers,'/tmp/analysis.csv')

