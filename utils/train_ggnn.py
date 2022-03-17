import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from shutil import copyfile

def train(epoch, dataloader, net, criterion, optimizer, opt, writer):
    
    for i, (adj_matrix, embedding_matrix, target) in enumerate(dataloader, 0):
        # print("----------------")
        net.zero_grad()
        # print(embedding_matrix)
        # padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        # init_input = torch.cat((annotation, padding), 2)
        # init_input = torch.zeros(len(adj_matrix), opt.n_node, opt.state_dim).double()

        # init_input = torch.from_numpy(embedding_matrix).double()
        init_input = embedding_matrix
        # print(init_input.shape)
        # print(init_input)
        if opt.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            # annotation = annotation.cuda()
            target = target.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        # annotation = Variable(annotation)
        target = Variable(target)
        output = net(init_input, adj_matrix)
        # print(output.shape)
        # print(target.shape)
        # print(output)
        # print(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print(loss)
        # print(epoch)
        writer.add_scalar('loss', loss.data.item(), int(epoch))
        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    torch.save(net, opt.model_path)
    copyfile(opt.model_path, "{}.{}".format(opt.model_path, epoch))
