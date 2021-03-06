import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .performance import calculate_performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def test(dataloader, net, criterion, optimizer, opt):
    # net.train()
    test_loss = 0
    correct = 0
    net.eval()
    # print("Len test data : " + str(len(dataloader.dataset)))

    all_targets = []
    all_predicted = []
    for i, (adj_matrices, target) in enumerate(dataloader, 0):
        # print(target)
        net.zero_grad()
        # optimizer.zero_grad()
        # print("------------------")
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
            left_output, right_output = net(left_init_input, left_adj_matrix, right_init_input, right_adj_matrix)
            loss = criterion(left_output,right_output, target) 

            euclidean_distance = F.pairwise_distance(left_output, right_output)   
            # print(left_output)
            pred = output.data.max(1, keepdim=True)[1]
            # print(pred)
            # print(target.data.view_as(pred))
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    
        else:
            
            # all_targets = []
            for i in range(len(left_adj_matrix)):
                output = net(left_init_input, left_adj_matrix[i], right_init_input, right_adj_matrix[i])
                if output >= 0.5 :
                    all_predicted.append(1)
                else: 
                    all_predicted.append(0)
                # loss = target[i] - output
            # print("output")
            # print(output)
            # pred = output.data.max(1, keepdim=True)[1]
            # print(pred)
            # print(pred.data.view_as(target))
            # print(target.data.view_as(pred))
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # # print(correct)
            # all_predicted.extend(pred.data.view_as(target).cpu().numpy())
            # all_targets.extend(target.cpu().numpy())

        # test_loss += loss.item()
    print(all_predicted)
    print(target)
    TP, FP, TN, FN = calculate_performance(target, all_predicted)
    print(TP,FP,TN,FN)
    correct = TP + TN
    precision = 0
    if TP + FP > 0:
       precision = TP / (TP + FP)
    recall = 0
    if TP + FN > 0:
       recall = TP / (TP + FN)
    
    # test_loss /= len(dataloader.dataset)

    print('Accuracy: {}/{} ({:.0f}%), Precision: ({:.0f}%), Recall: ({:.0f}%)'.format(
        correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset), 100. * precision, 100. * recall))
