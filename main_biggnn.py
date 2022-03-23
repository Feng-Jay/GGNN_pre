import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.model import GGNN
from utils.model import BiGGNN
from utils.model import ContrastiveLoss

from utils.train_biggnn import train
from utils.test_biggnn import test
from utils.data.dataset import CrossLingualProgramData
from utils.data.dataloader import bAbIDataloader
from tensorboardX import SummaryWriter
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--state_dim', type=int, default=5, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=10, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=2, help='manual seed')
parser.add_argument('--left_directory', default="my_data/new/train/train_x.txt", help='left encoded program data')
parser.add_argument('--right_directory', default="my_data/old/train/train_x.txt", help='right encoded program data')
parser.add_argument('--model_path', default="model/model.ckpt", help='path to save the model')
parser.add_argument('--n_hidden', type=int, default=50, help='number of hidden layers')
parser.add_argument('--size_vocabulary', type=int, default=7200, help='maximum number of node types')
parser.add_argument('--is_training_ggnn', type=bool, default=False, help='Training GGNN or BiGGNN')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--loss', type=int, default=0 ,help='1 is contrastive loss, 0 is cross entropy loss')
parser.add_argument('--log_path', default="" ,help='log path for tensorboard')
parser.add_argument('--data_percentage', type=float, default=1.0, help='data percentage')
parser.add_argument('--epoch', type=int, default=0, help='epoch to test')



opt = parser.parse_args()
print(opt)
if opt.training and opt.log_path != "":
  previous_runs = os.listdir(opt.log_path)
  if len(previous_runs) == 0:
    run_number = 1
  else:
    run_number = max([int(s.split("run-")[1]) for s in previous_runs]) + 1
  writer = SummaryWriter("%s/run-%03d" % (opt.log_path, run_number))
else:
  writer = None

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.training:
  if opt.loss == 1:
    print("Training Bi-GGNN with contrastive loss.")
  if opt.loss == 0:
    print("Training Bi-GGNN with cross entropy loss.")

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# This part is the implementation to illustrate Graph-Level output from program data
def main(opt):
    opt.data_percentage = 1
    print("Loading data...............")
    if opt.training:
       train_dataset = CrossLingualProgramData(opt.size_vocabulary, opt.left_directory,opt.right_directory, True, opt.loss, opt.n_classes,opt.data_percentage)
    #    print(train_dataset)
       train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.train_batch_size, \
                                      shuffle=True, num_workers=1)


    test_dataset = CrossLingualProgramData(opt.size_vocabulary, 'my_data/new/test/test_x.txt','my_data/old/test/test_x.txt', False,opt.loss, opt.n_classes,opt.data_percentage)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.train_batch_size, \
                                      shuffle=True, num_workers=1)

    opt.annotation_dim = 1  # for bAbI
    if opt.training:
       opt.n_edge_types = train_dataset.n_edge_types
       opt.n_node = train_dataset.n_node
    else:
       opt.n_edge_types = test_dataset.n_edge_types
       opt.n_node = test_dataset.n_node
    # print("Max node : " + str(opt.n_node))

    # if opt.testing:
    #     filename = "{}.{}".format(opt.model_path, opt.epoch)
    epoch = opt.epoch
    # else:
    #     filename = opt.model_path
    #     epoch = -1
    # if os.path.exists(filename):
    #     if opt.testing:
    #        print("Using No. {} saved model....".format(opt.epoch))
    #     dirname = os.path.dirname(filename)
    #     basename = os.path.basename(filename)
    #     epochs = os.listdir(dirname)
    #     if len(epochs) > 0:
    #        for s in epochs:
    #           if s.startswith(basename) and basename != s:
    #              x = s.split(os.extsep)
    #              e = x[len(x) - 1]
    #              epoch = max(epoch, int(e))
    #        if epoch != -1:
    #           print("Using No. {} of the saved models...".format(epoch))
    #           filename = "{}.{}".format(opt.model_path, epoch)
    #     if epoch != -1:
    #        print("Using No. {} saved model....".format(epoch))
    #     else:
    #        print("Using saved model....")
    #     net = torch.load(filename)
    # else:
    net = BiGGNN(opt)
    net.double()

    if opt.loss == 1:
        criterion = ContrastiveLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    if opt.training:
        for epoch in range(epoch+1, epoch + opt.niter):
            # input("train!")
            train(epoch, train_dataloader, net, criterion, optimizer, opt, writer)
        # writer.close()
    print("test!")

    if opt.testing:
        # filename = "{}.{}".format(opt.model_path, epoch)
        # # if os.path.exists(filename):
        #     net = torch.load(filename)
        #     net.cuda()
        #     optimizer = optim.Adam(net.parameters(), lr=opt.lr)
        test(test_dataloader, net, criterion, optimizer, opt)
if __name__ == "__main__":
    main(opt)

