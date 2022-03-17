import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

import pickle

from utils.model import GGNN
from utils.model import ClassPrediction
from utils.train_ggnn import train
from utils.test_ggnn import test
from utils.data.dataset import MonoLanguageProgramData
from utils.data.dataloader import bAbIDataloader
from tensorboardX import SummaryWriter
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=5, help='input batch size')
parser.add_argument('--test_batch_size', type=int, default=5, help='input batch size')
parser.add_argument('--state_dim', type=int, default=30, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=10, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=10, help='manual seed')
parser.add_argument('--directory', default="program_data/github_java_sort_function_babi", help='program data')
parser.add_argument('--model_path', default="model/model.ckpt", help='path to save the model')
parser.add_argument('--n_hidden', type=int, default=50, help='number of hidden layers')
parser.add_argument('--size_vocabulary', type=int, default=59, help='maximum number of node types')
parser.add_argument('--is_training_ggnn', type=bool, default=True, help='Training GGNN or BiGGNN')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--training_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--log_path', default="logs/" ,help='log path for tensorboard')
parser.add_argument('--epoch', type=int, default=0, help='epoch to test')
parser.add_argument('--pretrained_embeddings', default="embedding/fast_pretrained_vectors.pkl", help='pretrained embeddings url, there are 2 objects in this file, the first object is the embedding matrix, the other is the lookup dictionary')


opt = parser.parse_args()
print(opt)
if opt.training and opt.log_path != "":
    previous_runs = os.listdir(opt.log_path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split("run-")[1]) for s in previous_runs]) + 1
    writer = SummaryWriter("%s/run-%03d" % (opt.log_path, run_number))
    # writer = SummaryWriter(opt.log_path)
else:
    writer = None

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# This part is the implementation to illustrate Graph-Level output from program data
def main(opt):
    with open(opt.pretrained_embeddings, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh,encoding='latin1')
        # print(len(embeddings))
        # print(embed_lookup.keys())

        opt.embeddings = embeddings
        opt.embed_lookup = embed_lookup


    if opt.training:
       train_dataset = MonoLanguageProgramData(opt.size_vocabulary, opt.embeddings, opt.directory, True, opt.n_classes, opt.training_percentage)
       train_dataloader = bAbIDataloader(train_dataset, batch_size=opt.train_batch_size, \
                                      shuffle=True, num_workers=2)

    test_dataset = MonoLanguageProgramData(opt.size_vocabulary, opt.embeddings, opt.directory, False, opt.n_classes)
    test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.test_batch_size, \
                                     shuffle=True, num_workers=2)

    opt.annotation_dim = 1  # for bAbI
    if opt.training:
       opt.n_edge_types = train_dataset.n_edge_types
       opt.n_node = train_dataset.n_node
    else:
       opt.n_edge_types = test_dataset.n_edge_types
       opt.n_node = test_dataset.n_node
    if opt.testing:
        filename = "{}.{}".format(opt.model_path, opt.epoch)
        epoch = opt.epoch
    else:
        filename = opt.model_path
        epoch = -1
    if os.path.exists(filename):
        if opt.testing:
           print("Using No. {} saved model....".format(opt.epoch))
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        epochs = os.listdir(dirname)
        if len(epochs) > 0:
           for s in epochs:
              if s.startswith(basename) and basename != s:
                 x = s.split(os.extsep)
                 e = x[len(x) - 1]
                 epoch = max(epoch, int(e))
           if epoch != -1:
              print("Using No. {} of the saved models...".format(epoch))
              filename = "{}.{}".format(opt.model_path, epoch)
        if epoch != -1:
           print("Using No. {} saved model....".format(epoch))
        else:
           print("Using saved model....")
        net = torch.load(filename)
    else:
        net = GGNN(opt)
        net.double()

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    # embedding_matrix = train_dataset.embedding_matrix
    if opt.training:
        for epoch in range(epoch+1, epoch + opt.niter):
            train(epoch, train_dataloader, net, criterion, optimizer, opt, writer)
        writer.close()

    if opt.testing:
        filename = "{}.{}".format(opt.model_path, epoch)
        if os.path.exists(filename):
                 net = torch.load(filename)
                 net.cuda()
                 optimizer = optim.Adam(net.parameters(), lr=opt.lr)
        test(test_dataloader, net, criterion, optimizer, opt)
if __name__ == "__main__":
    main(opt)

