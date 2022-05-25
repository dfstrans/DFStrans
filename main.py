from DFSTrans import *
from utils_DFSTrans import *
from positional_encodings import *
import argparse
import time
import os
import numpy as np
import math
import random
import pickle
from sklearn.model_selection import StratifiedShuffleSplit


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sequence_length', required = True, type=int, help='sequence length',defauñt = 8000)
    parser.add_argument(
        '--n_channels', required = True, type=int, help='number of channels',defauñt = 20)
    parser.add_argument(
        '--batch_size', required = True, type=int, help='batch size',defauñt = 16)
    parser.add_argument(
        '--conv_filters', required = True, type=int, help='number of filters on each convolutional layer'
        ,defauñt = 20)
    parser.add_argument(
        '--window_length', required = True, type=int, help='window length',defauñt = 100)
    parser.add_argument(
        '--window_step', required = True, type=int, help='window step',defauñt = 100)
    parser.add_argument(
        '--time_steps', required = True, type=int, help='number of windows in which the time series are divided',
        defauñt = 80)
    parser.add_argument(
        '--learning_rate', required = True, type=float, help='sequence length',defauñt = 0.00001)
    parser.add_argument(
        '--n_epochs', required = True, type=int, help='sequence length',defauñt = 200)

    parser.add_argument(
        '-patience', '--patience', required=True, type=int,defauñt = 40)
    parser.add_argument(
        '--anomalous_percent', required=True, type=float,defauñt = 3)

    parser.add_argument( '--data_path', required=True, type=str,defauñt = 'simulations_data.h5')

    parser.add_argument( '--path_results_excell', required=True, type=str,defauñt = 'Results.xlsx')

    parser.add_argument( '--workbook_name', required=True, type=str,defauñt = 'DFStrans')


    parser.add_argument('--eval', action='store_true')

    parser.add_argument('--n_cv_folds', required = True, type=int, help='number cv folds',
        defauñt = 5)

    args = parser.parse_args()


def main(args):
    # set a random seed to reproduce the results over different executions
    random.seed(7)

    for iteration in range(args.n_cv_folds):

        with h5py.File(args.data_path, 'r') as hf:
            labels = hf['dataset'][:, -1]

        # get an ordered an ascending index of labels to be able to match each instance with its corresponding label
        index = np.array(range(len(labels)))
        data_index_list = np.array(range(len(labels)))
        normal_data_index = [index for index in data_index_list if labels[index] == 0]
        anomalous_data_index = [index for index in data_index_list if labels[index] == 1]

        np.random.shuffle(anomalous_data_index)
        n_anomalies_to_get = int(args.anomalous_percent * len(labels) / 100)
        reduced_anomalous_data_index = anomalous_data_index[:n_anomalies_to_get]
        reduced_data_index = np.hstack((normal_data_index, reduced_anomalous_data_index))
        reduced_labels = labels[reduced_data_index]

        # set a random seed to reproduce the results over different executions
        random.seed(7)

        with h5py.File(data_path, 'r') as hf:
            labels = hf['dataset'][:, -1]

        # get an ordered an ascending index of labels to be able to match each instance with its corresponding label
        index = np.array(range(len(labels)))

        data_index_list = np.array(range(len(labels)))
        normal_data_index = [index for index in data_index_list if labels[index] == 0]
        anomalous_data_index = [index for index in data_index_list if labels[index] == 1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, train_size=0.7, random_state=42)
        for train_index, val_test_index in sss.split(reduced_data_index, reduced_labels):
            # train_index and test_index are not the real indexes but new indexes made from reduced_data_index
            real_train_index, real_val_test_index = reduced_data_index[train_index], reduced_data_index[val_test_index]
            y_train, y_val_test = labels[real_train_index], labels[real_val_test_index]

            real_val_index, real_test_index, y_val, y_test = train_test_split(real_val_test_index, y_val_test,
                                                                              test_size=0.5, random_state=42)

            trainset = CustomDataGenerator(args.data_path, real_train_index, args.sequence_length, args.time_steps,
                                           args.window_length, args.window_step, args.n_channels)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=6)
            valset = CustomDataGenerator(args.data_path, real_val_index, args.sequence_length, args.time_steps,
                                         args.window_length, args.window_step, args.n_channels)
            valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=6)

            testset = CustomDataGenerator(args.data_path, real_test_index, args.sequence_length, args.time_steps,
                                          args.window_length, args.window_step, args.n_channels)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=6)

            trainset_minmax = CustomDataGenerator_minmax(args.data_path, np.array(real_train_index),
                                                         args.sequence_length,args.time_steps, args.window_length,
                                                         args.window_step, args.n_channels)
            trainloader_minmax = torch.utils.data.DataLoader(trainset_minmax, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=6)
            minmax_dict = MinMax_total(trainloader_minmax, args.n_channels)

            net = CNN_TimeSensorTransformer()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
            cudnn.benchmark = True
            net = net.cuda()
            f1_best = 0
            recall_best = 0
            precision_best = 0

            mean_execution_time = 0

            early_stop = 0
            loss_best = 1000

            for epoch in range(args.n_epochs):  # loop over the dataset multiple times
                if early_stop > args.patience:
                    break
                running_loss = 0.0
                steps_per_epoch = 0
                start_time = time.time()
                n_samples = 0

                tp_all = 0
                tn_all = 0
                fp_all = 0
                fn_all = 0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = torch.cat(inputs).view(args.n_channels, inputs[0].size()[0],
                                                    args.time-steps, 1, args.window_length)
                    for sensor in range(args.n_channels):
                        channel_values = inputs[sensor].view(inputs[sensor].shape[0], -1)
                        min_val, max_val = minmax_dict['channel_{}'.format(sensor)]
                        scaled_data = custom_MINMAX(channel_values, min_val, max_val)
                        inputs[sensor] = torch.Tensor(scaled_data.float()).view(inputs[0].size()[0],
                                                                                args.time-steps, 1, args.window_length)

                    inputs, labels = Variable(inputs.cuda().type(torch.cuda.FloatTensor)), Variable(labels.cuda())
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    output = net(inputs)

                    loss = criterion(output.float(), labels.float())
                    output = (torch.sigmoid(output) > 0.5).int()
                    tp = (labels * output).sum().to(torch.float32)
                    tn = ((1 - labels) * (1 - output)).sum().to(torch.float32)
                    fp = ((1 - labels) * output).sum().to(torch.float32)
                    fn = (labels * (1 - output)).sum().to(torch.float32)
                    tp_all += tp
                    tn_all += tn
                    fp_all += fp
                    fn_all += fn

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    steps_per_epoch += 1

                precision = tp_all / (tp_all + fp_all)
                recall = tp_all / (tp_all + fn_all)

                f1 = 2 * (precision * recall) / (precision + recall)

                print('Epoch %d loss: %.5f ' % (epoch + 1, running_loss / steps_per_epoch))
                print('Precision train %.5f , Recall train %.5f , F1 train %.5f' % (precision, recall, f1))
                # measure execution time in seconds
                execution_time = int(round((time.time() - start_time)))
                mean_execution_time += execution_time

                with torch.no_grad():
                    net.eval()
                    tp_all = 0
                    tn_all = 0
                    fp_all = 0
                    fn_all = 0
                    val_loss = 0.0
                    steps_per_epoch = 0
                    start_time = time.time()
                    correct = 0
                    for i, data in enumerate(valloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = torch.cat(inputs).view(20, inputs[0].size()[0], 80, 1, 100)
                        for sensor in range(args.n_channels):
                            channel_values = inputs[sensor].view(inputs[sensor].shape[0], -1)
                            min_val, max_val = minmax_dict['channel_{}'.format(sensor)]
                            scaled_data = custom_MINMAX(channel_values, min_val, max_val)
                            inputs[sensor] = torch.Tensor(scaled_data.float()).view(inputs[0].size()[0],
                                                                                    args.time - steps, 1,
                                                                                    args.window_length)

                        inputs, labels = Variable(inputs.cuda().type(torch.cuda.FloatTensor)), Variable(labels.cuda())

                        # forward + backward + optimize
                        output = net(inputs)

                        output = output.flatten()
                        labels = labels.flatten()
                        output = output.view(output.size()[0], 1)
                        labels = labels.view(labels.size()[0], 1)

                        loss = criterion(output.float(), labels.float())
                        output = (torch.sigmoid(output) > 0.5).int()

                        tp = (labels * output).sum().to(torch.float32)
                        tn = ((1 - labels) * (1 - output)).sum().to(torch.float32)
                        fp = ((1 - labels) * output).sum().to(torch.float32)
                        fn = (labels * (1 - output)).sum().to(torch.float32)

                        tp_all += tp
                        tn_all += tn
                        fp_all += fp
                        fn_all += fn

                        val_loss += loss.item()
                        steps_per_epoch += 1

                    precision = tp_all / (tp_all + fp_all)
                    recall = tp_all / (tp_all + fn_all)

                    f1 = 2 * (precision * recall) / (precision + recall)
                    g_mean = (precision * recall) ** 0.5
                    acc = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)

                    print('test loss: %.5f ' % (val_loss / steps_per_epoch))
                    print('Precision test %.5f , Recall test %.5f , F1 test %.5f' % (precision, recall, f1))

                    if val_loss / steps_per_epoch < loss_best:
                        early_stop = 0
                        loss_best = val_loss / steps_per_epoch
                        checkpoint = {'epoch': epoch + 1,
                                      'state_dict': net.state_dict(),
                                      'optimizer': optimizer.state_dict()}
                        torch.save(checkpoint,
                                   'DFStrans_it_{}.pt'.format(
                                       iteration))
                    else:
                        early_stop += 1
                    net.train()


        mean_execution_time = mean_execution_time / epoch
        with torch.no_grad():
            net = CNN_TimeSensorTransformer()
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
            net = net.cuda()
            checkpoint = torch.load('DFStrans_it_{}.pt'.format(iteration))

            net.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            net.eval()

            tp_all = 0
            tn_all = 0
            fp_all = 0
            fn_all = 0
            val_loss = 0.0
            steps_per_epoch = 0
            start_time = time.time()
            correct = 0
            for i, data in enumerate(testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = torch.cat(inputs).view(20, inputs[0].size()[0], 80, 1, 100)
                for sensor in range(args.n_channels):
                    channel_values = inputs[sensor].view(inputs[sensor].shape[0], -1)
                    min_val, max_val = minmax_dict['channel_{}'.format(sensor)]
                    scaled_data = custom_MINMAX(channel_values, min_val, max_val)
                    inputs[sensor] = torch.Tensor(scaled_data.float()).view(inputs[0].size()[0],
                                                                            args.time - steps, 1, args.window_length)

                #         #         torch.cat(inputs).view(32,20,80,100,1)
                inputs, labels = Variable(inputs.cuda().type(torch.cuda.FloatTensor)), Variable(labels.cuda())

                # forward + backward + optimize
                output = net(inputs)

                output = output.flatten()
                labels = labels.flatten()
                output = output.view(output.size()[0], 1)
                labels = labels.view(labels.size()[0], 1)

                loss = criterion(output.float(), labels.float())
                output = (torch.sigmoid(output) > 0.5).int()

                tp = (labels * output).sum().to(torch.float32)
                tn = ((1 - labels) * (1 - output)).sum().to(torch.float32)
                fp = ((1 - labels) * output).sum().to(torch.float32)
                fn = (labels * (1 - output)).sum().to(torch.float32)

                tp_all += tp
                tn_all += tn
                fp_all += fp
                fn_all += fn

                val_loss += loss.item()
                steps_per_epoch += 1

            precision = tp_all / (tp_all + fp_all)
            recall = tp_all / (tp_all + fn_all)

            f1 = 2 * (precision * recall) / (precision + recall)
            g_mean = (precision * recall) ** 0.5
            acc = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)

            save_results_excel(args.path_results_excel, args.workbook_name,
                               iteration, acc, precision, recall, f1,
                               mean_execution_time, g_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DFStrans training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
