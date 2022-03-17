import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        # print(A.shape)
        # print(state_in.shape)
        A_in = A[:, :self.n_node]
        A_out = A[:, self.n_node:]

        a_in = torch.mm(A_in, state_in)
        a_out = torch.mm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        
        joined_input = torch.cat((a_in, a_out, r * state_cur), 1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output

class ClassPrediction(nn.Module):

    def __init__(self, opt):
        super(ClassPrediction, self).__init__()

        self.class_prediction = nn.Sequential(
            nn.Linear(opt.n_node, opt.n_hidden),
            nn.Tanh(),
            nn.Linear(opt.n_hidden, opt.n_classes),
            nn.Softmax(dim=1)    
        )

        self.criterion = nn.CrossEntropyLoss()
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, graph_representation, target):
        output = self.class_prediction(graph_representation)
        loss = self.criterion(output, target)
       
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1.0 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        # assert (opt.state_dim >= opt.annotation_dim, 'state_dim must be no less than annotation_dim')
        self.is_training_ggnn = opt.is_training_ggnn
        self.state_dim = opt.state_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.n_classes = opt.n_classes

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LeakyReLU(),
            nn.Linear(self.state_dim, 1),
            nn.Tanh(),   
        )

      
        self.soft_attention = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LeakyReLU(),
            nn.Linear(self.state_dim, 1),
            nn.Sigmoid(),
           
        )

        self.class_prediction = nn.Sequential(
            nn.Linear(opt.n_node, opt.n_hidden),
            nn.LeakyReLU(),
            nn.Linear(opt.n_hidden, opt.n_classes),
            nn.Softmax(dim=1)    
        )

        # self.class_prediction = nn.Sequential(
        #     nn.Linear(opt.n_node, opt.n_classes),
        #     nn.Softmax(dim=1)    
        # )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, prop_state, A):
        # print(prop_state.shape)
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            # print(prop_state.shape)
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            # print(in_states.shape)
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            # print(self.n_node)
            in_states = in_states.view(self.n_node, self.state_dim)

            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(self.n_node, self.state_dim)
            # print(in_states.shape)
            prop_state = self.propogator(in_states, out_states, prop_state, A)
       
        # print("Prop state : " + str(prop_state.shape))
        # output = self.out(prop_state)
        # print("Out : " + str(output.shape))
        # print("prop")
        # print(prop_state.shape)
        soft_attention_ouput = self.soft_attention(prop_state)
        # print(soft_attention_ouput)
        # print("Soft : " + str(soft_attention_ouput.shape))
        # Element wise hadamard product to get the graph representation, check Equation 7 in GGNN paper for more details
        output = torch.mul(prop_state,soft_attention_ouput)
        # print(output.shape)
        # print(output.shape)
        # print("Out : " + str(output.shape))
        output = output.sum(1)
        # print(output.shape)
        if self.is_training_ggnn == True:
            output = self.class_prediction(output)
        # print(output.shape)

        return output


class BiGGNN(nn.Module):
    def __init__(self, opt):
        super(BiGGNN, self).__init__()

        self.opt = opt
        self.ggnn = GGNN(opt)
  
        self.n_node = opt.n_node

        self.fc_output = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

        self.feature_extraction = nn.Sequential(
            nn.Linear(2000, 50),
            nn.ReLU(),
            nn.Linear(50, 20)  
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, left_prop_state, left_A, right_prop_state, right_A):
        
        # self.ggnn.zero_grad()
        # print(left_prop_state.shape)
        left_output = self.ggnn(left_prop_state, left_A)
        right_output = self.ggnn(right_prop_state, right_A)
        # print(self.n_node)
        # print(left_output.shape)

        left_output = self.feature_extraction(left_output)
        right_output = self.feature_extraction(right_output)
        # print(left_output)

        if self.opt.loss == 1:
            return left_output, right_output
        else:
            # concat_layer = torch.cat((left_output, right_output),0)
            # print(concat_layer.shape)
            concat_layer = left_output - right_output
            output = self.fc_output(concat_layer)
            # print(output)
            return output





# class BiGGNN(nn.Module):
#     def __init__(self, opt):
#         super(BiGGNN, self).__init__()

#         self.opt = opt
#         self.state_dim = opt.state_dim
#         self.annotation_dim = opt.annotation_dim
#         self.n_edge_types = opt.n_edge_types
#         self.n_node = opt.n_node
#         self.n_steps = opt.n_steps
#         self.n_classes = opt.n_classes

#         for i in range(self.n_edge_types):
#             # incoming and outgoing edge embedding
#             in_fc = nn.Linear(self.state_dim, self.state_dim)
#             out_fc = nn.Linear(self.state_dim, self.state_dim)
#             self.add_module("in_{}".format(i), in_fc)
#             self.add_module("out_{}".format(i), out_fc)

#         self.in_fcs = AttrProxy(self, "in_")
#         self.out_fcs = AttrProxy(self, "out_")

#         # Propogation Model
#         self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

#         # Output Model
#         self.out = nn.Sequential(
#             nn.Linear(self.state_dim, self.state_dim),
#             nn.Tanh(),
#             nn.Linear(self.state_dim, 5),
#             nn.Tanh(),   
#         )
      
#         self.soft_attention = nn.Sequential(
#             nn.Linear(self.state_dim, self.state_dim),
#             nn.Tanh(),
#             nn.Linear(self.state_dim, 5),
#             nn.Sigmoid(),
           
#         )

#         self.fc_output = nn.Sequential(
#             nn.Linear(10*2, 50),
#             nn.ReLU(),
#             nn.Linear(50, 2),
#             nn.Softmax(dim=1)
#         )

#         self.feature_representation = nn.Sequential(
#             nn.Linear(self.n_node, 50),
#             nn.ReLU(),
#             nn.Linear(50, 10)  
#         )

#         self._initialization()

#     def _initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.xavier_normal_(m.weight.data)
#                 if m.bias is not None:
#                     init.normal_(m.bias.data)

#     def side_forward(self, prop_state, annotation, A):
#         for i_step in range(self.n_steps):
#             in_states = []
#             out_states = []
#             for i in range(self.n_edge_types):
#                 in_states.append(self.in_fcs[i](prop_state))
#                 out_states.append(self.out_fcs[i](prop_state))
#             in_states = torch.stack(in_states).transpose(0, 1).contiguous()
#             in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
#             out_states = torch.stack(out_states).transpose(0, 1).contiguous()
#             out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

#             prop_state = self.propogator(in_states, out_states, prop_state, A)
  
#         output = self.out(prop_state)

#         soft_attention_ouput = self.soft_attention(prop_state)
#         output = torch.mul(output,soft_attention_ouput)
#         output = output.sum(2)
#         return output

#     def forward(self, left_prop_state, left_annotation, left_A, right_prop_state, right_annotation, right_A):
#         left_output = self.side_forward(left_prop_state, left_annotation, left_A)
#         right_output = self.side_forward(right_prop_state, right_annotation, right_A)

#         left_output = self.feature_representation(left_output)
#         right_output = self.feature_representation(right_output)
        
#         if self.opt.loss == 1:
#             return left_output, right_output
#         else:
#             concat_layer = torch.cat((left_output, right_output),1)
#             output = self.fc_output(concat_layer)
#             print(output)
#             return output


