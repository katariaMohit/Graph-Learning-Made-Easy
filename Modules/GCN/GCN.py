import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, task='node'):
        super(GCN, self).__init__()
        self.task = task

        # Input Layer
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim_list[0]))
        #-------------------



        # hidden layers/Convolutional or Message Passing Layers
        for i in range(len(hidden_dim_list) - 1):
            self.convs.append(self.build_conv_model(hidden_dim_list[i], hidden_dim_list[i+1]))
        # ------------------        



        # Layer Normalization before un-commenting first read more about it
        # self.lns = nn.ModuleList()
        # self.lns.append(nn.LayerNorm(hidden_dim))
        # self.lns.append(nn.LayerNorm(hidden_dim))
        # -------------------
        


        # Add 2 Linear Layers i.e layers used for classification once we have the embeddings
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim_list[len(hidden_dim_list)-1], hidden_dim_list[len(hidden_dim_list)-1]), nn.Dropout(0.25), 
            nn.Linear(hidden_dim_list[len(hidden_dim_list)-1], output_dim))
        # -------------------



        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Ohhhh..ho!!! Currently We only support node classification Level Tasks.')

        self.dropout = 0.25
        # number of hidden layers 
        self.num_layers = len(hidden_dim_list)

    
    # Helper Function
    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x

            # only Defining Relu for now modify here if you want to change the activation function
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            
            # Layer Normalization before un-commenting first read more about it (also see this in init function)
            # if not i == self.num_layers - 1:
            #     x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
        
        # Mapping of the embeddings is completed here 
        # emb is the embedding in the desired space i.e embedding which we got after message passing


        # Second part of GCN(Classification task on extracted embeddings)
        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)


    # define our loss function here
    def loss(self, pred, label):
        return F.nll_loss(pred, label)