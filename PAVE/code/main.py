
import os
import sys
import time
import json
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm
from tools import parse, py_op
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import loss
import models
import function
import loaddata
from loaddata import dataloader
from models import attention

arg_list = parse.args
arg_list.hard_mining = 0
if torch.cuda.is_available():
    arg_list.gpu = 1
else:
    arg_list.gpu = 0

if arg_list.model == 'attention':
    arg_list.epochs = max(30, arg_list.epochs)

arg_list.use_trend = max(arg_list.use_trend, arg_list.use_value)
arg_list.use_value = max(arg_list.use_trend, arg_list.use_value)
arg_list.rnn_size = arg_list.embed_size
arg_list.hidden_size = arg_list.embed_size
if arg_list.num_layers > 1 or arg_list.model != 'attention':
    arg_list.compute_weight = 0
arg_list.compute_weight = 0


def train_eval_func(data_dict, phase_type='train'):
    epoch_data = data_dict['epoch']
    model_data = data_dict['model']
    loss_data = data_dict['loss']
    if phase_type != 'train':
        data_loader = data_dict['val_loader']
        model_data.eval()
    else:
        data_loader = data_dict['train_loader']
        optimizer = data_dict['optimizer']
        model_data.train()

    temp_data_dic = dict()

    for item_data in tqdm(data_loader):
        if arg_list.use_visit:
            item_data = item_data[:-1]
            if arg_list.gpu:
                item_data = [Variable(x.cuda()) for x in item_data]
            item_visits, item_values, item_mask, item_master, item_labels, item_times, item_trends = item_data
            output_list = model_data(
                item_visits, item_master, item_mask, item_times, phase_type, item_values, item_trends)
            output = output_list[0]

        classification_loss_output = loss_data(
            output, item_labels, arg_list.hard_mining)
        loss_gradient = classification_loss_output[0]
        function.compute_metric_func(
            output, item_labels, time, classification_loss_output, temp_data_dic, phase_type)

        if phase_type == 'train':
            optimizer.zero_grad()
            loss_gradient.backward()
            optimizer.step()

    print('\nEpoch: {:d} \t phase: {:s} \n'.format(epoch_data, phase_type))
    metric = function.print_metric_func('classification', temp_data_dic, phase_type)
    if arg_list.phase != 'train':
        print('metric = ', metric)
    if phase_type == 'val':
        if metric > data_dict['best_metric'][0]:
            data_dict['best_metric'] = [metric, epoch_data]
            function.save_models_func(data_dict)

        print('valid: metric: {:3.4f}\t epoch: {:d}\n'.format(
            metric, epoch_data))
        print('\t\t\t valid: best_metric: {:3.4f}\t epoch: {:d}\n'.format(
            data_dict['best_metric'][0], data_dict['best_metric'][1]))
    else:
        print('train: metric: {:3.4f}\t epoch: {:d}\n'.format(
            metric, epoch_data))


def main():
    data_dict = dict()
    data_dict['args'] = arg_list
    arg_list.split_nn = 3 * 5
    arg_list.vocab_size = arg_list.split_nn * 145 + 2
    if arg_list.task == 'mortality':
        pat_time_record_data = py_op.myreadjson(os.path.join(
            arg_list.result_dir, 'patient_time_record_dict.json'))
        pat_master_data = py_op.myreadjson(os.path.join(
            arg_list.result_dir, 'patient_master_dict.json'))
        pat_label_data = py_op.myreadjson(os.path.join(
            arg_list.result_dir, 'patient_label_dict.json'))

        if os.path.exists(os.path.join(arg_list.result_dir, 'train.json')):
            pat_train = list(
                json.load(open(os.path.join(arg_list.result_dir, 'train.json'))))
            pat_valid = list(
                json.load(open(os.path.join(arg_list.result_dir, 'valid.json'))))
            pat_test = list(
                json.load(open(os.path.join(arg_list.result_dir, 'test.json'))))
        else:
            pats = sorted(set(pat_label_data.keys()) & set(
                pat_time_record_data) & set(pat_master_data))
            num_train = int(0.7 * len(pats))
            num_valid = int(0.2 * len(pats))
            pat_train = pats[:num_train]
            pat_valid = pats[num_train:num_train+num_valid]
            pat_test = pats[num_train+num_valid:]

        arg_list.master_size = len(pat_master_data[pats[0]])

    train_load, val_load, test_load = pat_set_func(pat_time_record_data, pat_master_data, pat_label_data, pat_train, pat_valid, pat_test)

    data_dict['train_loader'] = train_load
    if arg_list.phase != 'train':
        data_dict['val_loader'] = test_load
    else:
        data_dict['val_loader'] = val_load

    cudnn.benchmark = True
    net = attention.MyAttention(arg_list)
    if arg_list.gpu:
        net = net.cuda()
        data_dict['loss'] = loss.Loss().cuda()
    else:
        data_dict['loss'] = loss.Loss()

    parameters = []
    for p_item in net.parameters():
        parameters.append(p_item)
    optimizer = torch.optim.Adam(parameters, lr=arg_list.lr)
    data_dict['optimizer'] = optimizer
    data_dict['model'] = net

    data_dict['epoch'] = 0
    data_dict['best_metric'] = [0, 0]

    if os.path.exists(arg_list.resume):
        function.load_models_func(data_dict, arg_list.resume)

    if arg_list.phase != 'train':
        train_eval_func(data_dict, 'test')
    else:
        for epoch in range(data_dict['epoch'] + 1, arg_list.epochs):
            data_dict['epoch'] = epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = arg_list.lr
            train_eval_func(data_dict, 'train')
            train_eval_func(data_dict, 'val')
        log_info = '# task : {:s}; model: {:s} ; last_time: {:d} ; auc: {:3.4f} \n'.format(
            arg_list.task, arg_list.model, arg_list.last_time, data_dict['best_metric'][0])
        with open('../result/log.txt', 'a') as f:
            f.write(log_info)

def pat_set_func(pat_time_record_data, pat_master_data, pat_label_data, pat_train, pat_valid, pat_test):
    train_data_set = dataloader.MyDataSet(
        pat_train,
        pat_time_record_data,
        pat_label_data,
        pat_master_data,
        arg_list=arg_list,
        phase='train')
    train_load = DataLoader(
        dataset=train_data_set,
        batch_size=arg_list.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
    val_data_set = dataloader.MyDataSet(
        pat_valid,
        pat_time_record_data,
        pat_label_data,
        pat_master_data,
        arg_list=arg_list,
        phase='val')
    val_load = DataLoader(
        dataset=val_data_set,
        batch_size=arg_list.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
    test_data_set = dataloader.MyDataSet(
        pat_test,
        pat_time_record_data,
        pat_label_data,
        pat_master_data,
        arg_list=arg_list,
        phase='val')
    test_load = DataLoader(
        dataset=test_data_set,
        batch_size=arg_list.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)
        
    return train_load,val_load,test_load


if __name__ == '__main__':
    main()
