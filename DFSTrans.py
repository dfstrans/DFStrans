from positional_encodings import *
import torch
import os
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import *
import h5py
import numpy as np
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torch.multiprocessing as mp
import time
import math
import random
from sklearn.model_selection import train_test_split
from torch.nn import Module
import csv
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MultiHead1DCNN(nn.Module):

    def __init__(self,conv_filters = 20, time_steps = 80):
        super(MultiHead1DCNN, self).__init__()

        self.conv_filters = conv_filters
        self.time_steps = time_steps

        self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=self.conv_filters, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(self.conv_filters,track_running_stats=False)
        self.conv1d2 = nn.Conv1d(in_channels=self.conv_filters, out_channels=self.conv_filters, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(self.conv_filters,track_running_stats=False)
        self.conv1d3 = nn.Conv1d(in_channels=self.conv_filters, out_channels=self.conv_filters, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(self.conv_filters,track_running_stats=False)
        self.maxpool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.conv1d1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.bn1(x)
        X = x.view(-1, x.size()[0] // self.time_steps, self.time_steps, self.conv_filters)
        X = x.view(-1, *(x.size()[2:]))
        x = self.conv1d2(x)

        x = F.relu(x)
        x = self.maxpool(x)

        x = self.bn2(x)
        X = x.view(-1, x.size()[0] // self.time_steps, self.time_steps, self.conv_filters)
        X = x.view(-1, *(x.size()[2:]))
        x = self.conv1d3(x)

        x = F.relu(x)
        x = self.maxpool(x)

        x = self.bn3(x)
        x = x.view(x.size()[0], x.size()[1], -1)
        return x

class TemporalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TemporalEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        return src, weights


class SpatialEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(SpatialEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SpatialEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        return src, weights

class TransTS(nn.Module):
    def __init__(self, feature_size=240, num_layers=1, dropout=0.1):
        super(TransTS, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.encoder_layer = TemporalEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        output, weights = self.transformer_encoder(src, self.src_mask)

        output = output.permute(1, 0, 2)

        return output, weights

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float(0.0))
        return mask


class TransSensor(nn.Module):
    def __init__(self, feature_size=240, num_layers=1, dropout=0.1):
        super(TransSensor, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.encoder_layer = SpatialEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):

        bs = src.size()[1]

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        output, weights = self.transformer_encoder(src, self.src_mask)

        output = output.permute(1, 0, 2)

        return output, weights

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float(0.0))
        return mask

class DFSTrans_model(nn.Module):

    def __init__(self, activation="relu", d_model=240, dim_feedforward=2048,
                 dropout=0.1,n_channels = 20,n_time_steps = 80,output_dim = 4800, n_units_l1 = 512):
        super(DFSTrans_model, self).__init__()
        self.conv_cell = nn.ModuleList([MultiHead1DCNN(time_steps=time_steps,n_channels=n_channels) for i in range(n_channels)])
        self.TimeDistributed_flatten = nn.ModuleList([TimeDistributed(Flatten) for i in range(n_channels)])
        self.trace = []
        self.TransformerTS_list = nn.ModuleList([TransTS() for i in range(n_channels)])
        self.TransformerS_list = nn.ModuleList([TransSensor() for i in range(n_time_steps)])
        self.d_model = d_model

        self.pos_encoder = DFTEncoding(self.d_model)

        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.dense1 = nn.Linear(output_dim, n_units_l1)
        self.dropout_out1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(n_units_l1, 1)

        self.n_time_steps = n_time_steps
        self.n_channels = n_channels

    def forward(self, input_x):

        trace = []
        for sensor_n in range(20):
            input_layer = input_x[sensor_n]
            input_layer_reshape = input_layer.view(-1, *(input_layer.size()[2:]))
            x = self.conv_cell[sensor_n](input_layer_reshape)
            x = x.view(x.size()[0] // self.n_time_steps, self.n_time_steps, -1)
            trace.append(x)

        x = torch.stack(trace)
        x = x.permute(2, 1, 3, 0)
        x = self.pos_encoder(x)

        input_ts = torch.clone(x)
        input_sensor = torch.clone(x)
        input_ts = input_ts.permute(3, 0, 1, 2)
        input_sensor = input_sensor.permute(0, 3, 1, 2)

        output_ts_list = []
        ts_weights_list = []

        for i in range(self.n_channels):
            channel = input_ts[i, :, :, :]
            trans_ts = self.TransformerTS_list[i]

            output_ts, weights_ts = trans_ts(channel)
            output_ts_list.append(output_ts)
            ts_weights_list.append(weights_ts)

        output_s_list = []
        s_weights_list = []

        for i in range(self.n_time_steps):
            ts = input_sensor[i, :, :, :]
            trans_s = self.TransformerS_list[i]

            output_s, weights_s = trans_s(ts)
            output_s_list.append(output_s)
            s_weights_list.append(weights_s)

        output_ts = torch.stack(output_ts_list)
        output_ts = output_ts.permute(1, 2, 3, 0)
        output_sensor = torch.stack(output_s_list)
        output_sensor = output_sensor.permute(1, 0, 3, 2)

        output_ts_sensor = output_ts + output_sensor

        bs = output_ts_sensor.size()[0]

        output_ts_sensor = output_ts_sensor.view(-1, output_ts_sensor.size()[2], output_ts_sensor.size()[3])
        output_ts_sensor = output_ts_sensor.permute(0, 2, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(output_ts_sensor))))

        output_ts_sensor = output_ts_sensor + self.dropout2(src2)
        src = self.norm2(output_ts_sensor)
        src = src.permute(0, 2, 1)
        src = src.reshape(bs, self.n_time_steps, src.size()[1] * src.size()[2])
        src = src.mean(1)

        x = self.dense1(src)
        x = F.relu(x)
        x = self.dropout_out1(x)
        output = self.dense2(x)

        return output

