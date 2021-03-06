import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from shutil import copyfile

def train(epoch, dataloader, net, criterion, optimizer, opt, writer):
    # print(epoch)
    # print(dataloader)
    for j, (adj_matrices, target) in enumerate(dataloader, 0):
        # print(adj_matrices)
        # print("i = {i}".format(i=j))
        # print(len(adj_matrices[0]))
        # continue 
        net.zero_grad()
        # optimizer.zero_grad()
        # print("test")
        left_adj_matrix = adj_matrices[0]
        right_adj_matrix = adj_matrices[1]
    
        left_init_input = torch.zeros( opt.n_node, opt.state_dim).double()
        right_init_input = torch.zeros(opt.n_node, opt.state_dim).double()

        if opt.cuda:
            # print("Using cuda for training.......")
            left_init_input = left_init_input.cuda()
            right_init_input = right_init_input.cuda()
            left_adj_matrix = left_adj_matrix.cuda()
            right_adj_matrix = right_adj_matrix.cuda()
            target = target.cuda()

        left_init_input = Variable(left_init_input)
        right_init_input = Variable(right_init_input)

        left_adj_matrix = Variable(left_adj_matrix)
        right_adj_matrix = Variable(right_adj_matrix)

        target = Variable(target)
        # print(target)

        if opt.loss == 1:
            for i in range(len(left_adj_matrix)):
                left_output, right_output = net(left_init_input, left_adj_matrix, right_init_input, right_adj_matrix)
                loss = criterion(left_output,right_output, target) 
            if writer:
               writer.add_scalar('loss', loss.data.item(), int(epoch))
        else:
            for i in range(len(left_adj_matrix)):
                output = net(left_init_input, left_adj_matrix[i], right_init_input, right_adj_matrix[i])
                # print(output)
                # print(target[i])
                loss = abs(target[i] - output)
                # loss = criterion(output, target) 
            if writer:
               writer.add_scalar('loss', loss.data.item(), int(epoch))
           
        loss.backward()
        optimizer.step()

        # if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
        print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, j, len(dataloader), loss.item()))

    # torch.save(net, opt.model_path)
    # copyfile(opt.model_path, "{}.{}".format(opt.model_path, epoch))
