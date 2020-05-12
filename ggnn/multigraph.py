from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
import torch
import random
import math
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from ggnn.rgcn import RGCNConv


class GCNSequential(nn.Sequential):
    """docstring for GCNSequential"""

    def __init__(self, *args, **kwargs):
        super(GCNSequential, self).__init__()
        self.args = args
        self.kwargs = kwargs

        super(GCNSequential, self).__init__(*args, **kwargs)

    def forward(self, input, edge_index):
        for module in self._modules.values():
            input = module(input, edge_index)
        return input


def zoneout(prev_h, next_h, rate, training=True):
    """TODO: Docstring for zoneout.

    :prev_h: TODO
    :next_h: TODO

    :p: when p = 1, all new elements should be droped
        when p = 0, all new elements should be maintained

    :returns: TODO

    """
    from torch.nn.functional import dropout
    if training:
        # bernoulli: draw a value 1.
        # p = 1 -> d = 1 -> return prev_h
        # p = 0 -> d = 0 -> return next_h
        # d = torch.zeros_like(next_h).bernoulli_(p)
        # return (1 - d) * next_h + d * prev_h
        next_h = (1 - rate) * dropout(next_h - prev_h, rate) + prev_h
    else:
        next_h = rate * prev_h + (1 - rate) * next_h

    return next_h


class KStepRGCN(nn.Module):
    """docstring for KStepRGCN"""

    def __init__(
            self,
            in_channels,
            out_channels,
            num_relations,
            num_bases,
            K,
            bias,
    ):
        super(KStepRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.K = K
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(in_channels,
                     out_channels,
                     num_relations,
                     num_bases,
                     bias)
        ] + [
            RGCNConv(out_channels,
                     out_channels,
                     num_relations,
                     num_bases,
                     bias) for _ in range(self.K - 1)
        ])

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.K):
            x = self.rgcn_layers[i](x=x,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    edge_norm=None)
            # not final layer, add relu
            if i != self.K - 1:
                x = torch.relu(x)
        return x


class GGRUCell(nn.Module):
    """Docstring for GGRUCell. """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_type=None,
                 dropout_prob=0.0,
                 num_relations=3,
                 num_bases=3,
                 K=1,
                 num_nodes=80,
                 global_fusion=False):
        """TODO: to be defined1. """
        super(GGRUCell, self).__init__()
        self.num_chunks = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion
        self.cheb_i = KStepRGCN(in_channels,
                                out_channels * self.num_chunks,
                                num_relations=num_relations,
                                num_bases=num_bases,
                                K=K,
                                bias=False)
        self.cheb_h = KStepRGCN(out_channels,
                                out_channels * self.num_chunks,
                                num_relations=num_relations,
                                num_bases=num_bases,
                                K=K,
                                bias=False)

        self.bias_i = Parameter(torch.Tensor(self.out_channels))
        self.bias_r = Parameter(torch.Tensor(self.out_channels))
        self.bias_n = Parameter(torch.Tensor(self.out_channels))
        self.dropout_prob = dropout_prob
        self.dropout_type = dropout_type
        if global_fusion is True:
            # self.convi11 = nn.Conv1d(self.num_nodes, 1, 1)
            # self.convh11 = nn.Conv1d(self.num_nodes, 1, 1)
            self.mlpi = nn.Linear(self.num_nodes * self.in_channels,
                                  self.out_channels)
            self.mlph = nn.Linear(self.num_nodes * self.out_channels,
                                  self.out_channels)
            self.bias_i_g = Parameter(torch.Tensor(self.out_channels))
            self.bias_r_g = Parameter(torch.Tensor(self.out_channels))
            self.bias_n_g = Parameter(torch.Tensor(self.out_channels))
            # self.conv_comb =
            # nn.Conv1d(self.out_channels * 2, self.out_channels, 1)
            # self.mlp_comb = nn.Linear()
            self.global_i = nn.Linear(out_channels,
                                      out_channels * self.num_chunks)
            self.global_h = nn.Linear(out_channels,
                                      out_channels * self.num_chunks)
            self.mlpatt = nn.Linear(self.out_channels * 2, self.out_channels)
            self.ln = nn.LayerNorm([self.out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.bias_i)
        init.ones_(self.bias_r)
        init.ones_(self.bias_n)

        if self.global_fusion is True:
            init.ones_(self.bias_i_g)
            init.ones_(self.bias_r_g)
            init.ones_(self.bias_n_g)

    def forward(self, inputs, edge_index, edge_attr, hidden=None):
        """TODO: Docstring for forward.

        :inputs: TODO
        :hidden: TODO
        :returns: TODO

        """
        if hidden is None:
            hidden = torch.zeros(inputs.size(0),
                                 self.out_channels,
                                 dtype=inputs.dtype,
                                 device=inputs.device)
        gi = self.cheb_i(inputs, edge_index=edge_index, edge_attr=edge_attr)
        gh = self.cheb_h(hidden, edge_index=edge_index, edge_attr=edge_attr)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r + self.bias_r)
        inputgate = torch.sigmoid(i_i + h_i + self.bias_i)
        newgate = torch.tanh(i_n + resetgate * h_n + self.bias_n)
        next_hidden = (1 - inputgate) * newgate + inputgate * hidden

        if self.global_fusion is True:
            global_input = self.mlpi(
                inputs.view(-1,
                            self.num_nodes * self.in_channels))
            global_hidden = self.mlph(
                hidden.view(-1,
                            self.num_nodes * self.out_channels))
            i_r_g, i_i_g, i_n_g = self.global_i(global_input).chunk(3, 1)
            h_r_g, h_i_g, h_n_g = self.global_h(global_hidden).chunk(3, 1)
            r_g = torch.sigmoid(i_r_g + h_r_g + self.bias_r_g)
            i_g = torch.sigmoid(i_i_g + h_i_g + self.bias_i_g)
            n_g = torch.tanh(i_n_g + r_g * h_n_g + self.bias_i_g)
            o_g = (1 - i_g) * n_g + i_g * global_hidden
            o_g = o_g.unsqueeze(1).repeat(1, self.num_nodes, 1)
            # residual_x = next_hidden

            next_hidden = next_hidden.view(-1,
                                           self.num_nodes,
                                           self.out_channels)
            combine_hidden = torch.cat([next_hidden, o_g], dim=-1)
            combine_hidden = combine_hidden.view(-1, 2 * self.out_channels)
            next_hidden = next_hidden.view(-1,
                                           self.out_channels) * torch.tanh(
                                               self.mlpatt(combine_hidden))
            # [batch, num_nodes, 2*dim] -> [batch, 2*dim, num_nodes]
            # combine_hidden = combine_hidden.transpose(1, 2)
            # [batch, dim, num_nodes] -> [batch, num_nodes, dim]
            # combine_hidden = combine_hidden.transpose(1, 2)
            next_hidden = self.ln(next_hidden).reshape(-1, self.out_channels)

        output = next_hidden

        if self.dropout_type == 'zoneout':
            next_hidden = zoneout(prev_h=hidden,
                                  next_h=next_hidden,
                                  rate=self.dropout_prob,
                                  training=self.training)

        elif self.dropout_type == 'dropout':
            next_hidden = F.dropout(next_hidden,
                                    self.dropout_prob,
                                    self.training)

        return output, next_hidden


class Net(torch.nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()
        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_input_dim = cfg['model']['input_dim']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']
        self.cfg = cfg
        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.use_curriculum_learning = self.cfg['model'][
            'use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(
            data=[self.cfg['model']['cl_decay_steps']])
        self.use_go = self.cfg['model'].get('use_go', True)
        self.fusion = self.cfg['model'].get('fusion', 'concat')
        self.dropout_type = cfg['model'].get('dropout_type', None)
        self.dropout_prob = cfg['model'].get('dropout_prob', 0.0)
        self.ar_alpha = cfg['model'].get('ar_alpha', 0)
        self.tar_beta = cfg['model'].get('tar_beta', 0)
        self.use_input = cfg['model'].get('use_input', True)
        self.num_relations = cfg['model'].get('num_relations', 3)
        self.K = cfg['model'].get('K', 3)
        self.num_bases = cfg['model'].get('num_bases', 3)
        act = cfg['model'].get('activation', 'relu')
        act_dict = {
            'relu': F.relu,
            'selu': F.selu,
            'relu6': F.relu6,
            'elu': F.elu,
            'celu': F.celu,
            'leaky_relu': F.leaky_relu,
        }
        self.mediate_activation = act_dict[act]
        self.global_fusion = cfg['model'].get('global_fusion', False)

        self.encoder_cells = nn.ModuleList([
            GGRUCell(self.num_input_dim,
                     self.num_units,
                     self.dropout_type,
                     self.dropout_prob,
                     self.num_relations,
                     num_bases=self.num_bases,
                     K=self.K,
                     num_nodes=self.num_nodes,
                     global_fusion=self.global_fusion),
        ] + [
            GGRUCell(self.num_units,
                     self.num_units,
                     self.dropout_type,
                     self.dropout_prob,
                     self.num_relations,
                     num_bases=self.num_bases,
                     K=self.K,
                     num_nodes=self.num_nodes,
                     global_fusion=self.global_fusion)
            for _ in range(self.num_rnn_layers - 1)
        ])

        self.decoder_cells = nn.ModuleList([
            GGRUCell(self.num_input_dim,
                     self.num_units,
                     self.dropout_type,
                     self.dropout_prob,
                     self.num_relations,
                     num_bases=self.num_bases,
                     K=self.K,
                     num_nodes=self.num_nodes,
                     global_fusion=self.global_fusion),
        ] + [
            GGRUCell(self.num_units,
                     self.num_units,
                     self.dropout_type,
                     self.dropout_prob,
                     self.num_relations,
                     self.K,
                     num_nodes=self.num_nodes,
                     global_fusion=self.global_fusion)
            for _ in range(self.num_rnn_layers - 1)
        ])
        # self.fusion = cfg['model'].get('fusion')
        self.output_type = cfg['model'].get('output_type', 'fc')
        if not self.fusion == 'concat':
            raise NotImplementedError(self.fusion)

        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)
        self.global_step = 0

    @staticmethod
    def _compute_sampling_threshold(step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse
        sigmoid.

        :step: TODO
        :k: TODO
        :returns: TODO

        """
        return k / (k + math.exp(step / k))

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        """TODO: Docstring for linear_scheduler_sampling.
        :returns: TODO

        """
        return k / (k + math.exp(step / k))

    def encode(self, sequences, edge_index, edge_attr=None):
        """
        Encodes input into hidden state on one branch for T steps.

        Return: hidden state on one branch.
        """
        hidden_states = [None] * len(self.encoder_cells)
        outputs = []
        for t, batch in enumerate(sequences):
            cur_input = batch.x
            for i, rnn_cell in enumerate(self.encoder_cells):
                cur_h = hidden_states[i]
                cur_out, cur_h = rnn_cell(inputs=cur_input,
                                          edge_index=edge_index,
                                          edge_attr=edge_attr,
                                          hidden=cur_h)

                # the hidden/output of previous layer is be fed to
                # the input of next layer
                hidden_states[i] = cur_h
                cur_input = self.mediate_activation(cur_out)
            outputs.append(cur_out)

        return outputs, hidden_states

    def forward(self, sequences):
        # encoder
        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()
        outputs, encoder_hiddens =\
            self.encode(sequences, edge_index=edge_index,
                        edge_attr=edge_attr)

        # decoder
        predictions = []
        decoder_hiddens = encoder_hiddens  # copy states
        GO = torch.zeros(decoder_hiddens[0].size()[0],
                         self.num_output_dim,
                         dtype=encoder_hiddens[0].dtype,
                         device=encoder_hiddens[0].device)
        decoder_input = GO

        for t in range(self.horizon):
            for i, rnn_cell in enumerate(self.decoder_cells):
                cur_h = decoder_hiddens[i]
                cur_out, cur_h = rnn_cell(inputs=decoder_input,
                                          edge_index=edge_index,
                                          edge_attr=edge_attr,
                                          hidden=cur_h)

                decoder_hiddens[i] = cur_h
                decoder_input = self.mediate_activation(cur_out)

            out = cur_out.reshape(-1, self.num_units)
            out = self.output_layer(out).view(-1,
                                              self.num_nodes,
                                              self.num_output_dim)
            predictions.append(out)
            if self.training and self.use_curriculum_learning:
                c = random.uniform(0, 1)
                T = self.inverse_sigmoid_scheduler_sampling(
                    self.global_step,
                    self.cl_decay_steps)
                use_truth_sequence = True if c < T else False
            else:
                use_truth_sequence = False

            if use_truth_sequence:
                # Feed the prev label as the next input
                decoder_input = sequences[t].y
            else:
                # detach from history as input
                decoder_input = out.detach().view(-1, self.num_output_dim)
            if not self.use_input:
                decoder_input = GO.detach()

        if self.training:
            self.global_step += 1

        return torch.stack(predictions).transpose(0, 1)
