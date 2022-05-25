import numpy as np
from torch.utils.data.dataset import Dataset
import torch

# custom class to generate batch data on the fly
class CustomDataGenerator(Dataset):
    def __init__(self, path, data_index, sequence_length, time_steps, window_length,
                 window_step, n_channels):
        """ Method called at the initialization of the class (Constructor).

            Args:
                path (string): path where the dataset is located
                data_index (array): index list of the dataset
                batch_size (int): the size of the batch
                sequence_length (int): the length of the time series
                time_steps (int): number of windows in which the time series are divided
                window_length (int): the length (data point) of the window
                window_step (int): indicates how much to slice de window over the time series
                n_channels (int): number of sensors

        """
        self.path = path
        self.data_index = data_index
        self.sequence_length = sequence_length
        self.time_steps = time_steps
        self.window_length = window_length
        self.window_step = window_step
        self.n_channels = n_channels
        self.len = data_index.shape[0]

    def __len__(self):
        """ Method called at the time of requiring the number of batches per epoch

                Returns:
                    int: number of batches per epoch

            """
        return self.len

    def __getitem__(self, idx):
        index = self.data_index[idx]

        col_size = self.sequence_length * self.n_channels
        with h5py.File(self.path, 'r') as hf:
            x = hf['dataset'][index, :col_size]
            y = hf['dataset'][index, -1:]

        x = reshape_data(x, self.sequence_length, self.time_steps, self.window_length, self.window_step,
                         self.n_channels)

        input_data_x_list = []
        for channel in range(self.n_channels):
            input_data_x_list.append(x[:, :, channel:channel + 1])

        return input_data_x_list, y


def reshape_data(data, sequence_length, time_steps, window_length, window_step, n_channels):
    x_reshaped = np.zeros((time_steps, window_length, n_channels))

    current_simulation_values = np.zeros((time_steps, window_length, n_channels))
    for channel in range(n_channels):
        channel_start_point = channel * sequence_length
        channel_end_point = channel_start_point + sequence_length
        param_values = data[channel_start_point:channel_end_point]

        for step in range(time_steps):
            time_step_start_point = step * window_step
            time_step_stop_point = time_step_start_point + window_length
            time_step_values = param_values[time_step_start_point:time_step_stop_point]
            current_simulation_values[step, :, channel] = time_step_values

        x_reshaped[:, :, :] = current_simulation_values

    return x_reshaped

def custom_MINMAX(batch_x, min_val, max_val):
    '''Scales data between values (0,1)

    Args:

        batch_x (1D mumpy array): features data of the current batch
        a (int): min value of transformed data
        b (int): max value of transformed data

    Returns:

        1D numpy array: Transformed data
    '''
    return (batch_x - min_val) / (max_val - min_val)


def save_results_excel(path_results_excel, workbook_name,
                       iteration, accuracy_score, precision_score, recall_score, f1_score,
                       execution_time, g_mean):
    workbook = load_workbook(path_results_excel)
    worksheet = workbook[workbook_name]

    initial_cell = 'B,2'
    initial_col = str.encode(initial_cell.split(',')[0])
    initial_row = int(initial_cell.split(',')[1])

    # accuracy
    cell = bytes([initial_col[0]]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(accuracy_score.item(),4)

    # precision
    cell = bytes([initial_col[0] + 1]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(precision_score.item(),4)

    # recall
    cell = bytes([initial_col[0] + 2]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(recall_score.item(),4)

    # f1_measure
    cell = bytes([initial_col[0] + 3]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(g_mean.item(),4)

    # execution_time
    cell = bytes([initial_col[0] + 4]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(f1_score.item(),4)

    # g_mean
    cell = bytes([initial_col[0] + 5]).decode('utf-8') + str(initial_row + iteration)
    worksheet[cell] = round(execution_time,4)

    workbook.save(path_results_excel)

# custom class to generate batch data on the fly
class CustomDataGenerator_minmax(Dataset):
    def __init__(self, path, data_index, sequence_length, time_steps, window_length,
                 window_step, n_channels):
        """ Method called at the initialization of the class (Constructor).

            Args:
                path (string): path where the dataset is located
                data_index (array): index list of the dataset
                batch_size (int): the size of the batch
                sequence_length (int): the length of the time series
                time_steps (int): number of windows in which the time series are divided
                window_length (int): the length (data point) of the window
                window_step (int): indicates how much to slice de window over the time series
                n_channels (int): number of sensors
                isTrainig (bool): whether the generator is for training or test

        """
        self.path = path
        self.data_index = data_index
        self.sequence_length = sequence_length
        self.time_steps = time_steps
        self.window_length = window_length
        self.window_step = window_step
        self.n_channels = n_channels
        self.len = data_index.shape[0]

    def __len__(self):
        """ Method called at the time of requiring the number of batches per epoch

                Returns:
                    int: number of batches per epoch

            """
        return self.len

    def __getitem__(self, idx):
        index = self.data_index[idx]

        col_size = self.sequence_length * self.n_channels
        with h5py.File(self.path, 'r') as hf:
            x = hf['dataset'][index, :col_size]
            y = hf['dataset'][index, -1:]

        # reshape data to 3D (time_steps* window_length, channels)
        x = reshape_data_minmax(x, self.sequence_length, self.time_steps, self.window_length, self.window_step,
                         self.n_channels)
        return x

def reshape_data_minmax(data, sequence_length, time_steps, window_length, window_step, n_channels):
    x_reshaped = np.zeros((time_steps* window_length, n_channels))
    for channel in range(n_channels):
        channel_start_point = channel * sequence_length
        channel_end_point = channel_start_point + sequence_length
        param_values = data[channel_start_point:channel_end_point]
        x_reshaped[:,channel] = param_values

    return x_reshaped

def MinMax_total(trainloader,n_channels):
    global_min = 1000000000000
    global_max = -1000000000000
    minmax_dict = {'channel_{}'.format(i): [global_min,global_max] for i in range(n_channels)}
    for i, data in enumerate(trainloader, 0):
        batch_x = data
#                 print(batch_x.shape)
        for channel in range(n_channels):
            channel_values = batch_x[:,:,channel]
            min_batch = torch.amin(channel_values)
            max_batch = torch.amax(channel_values)
            if min_batch<minmax_dict['channel_{}'.format(channel)][0]:
                minmax_dict['channel_{}'.format(channel)][0] = min_batch
            if  max_batch>minmax_dict['channel_{}'.format(channel)][1]:
                minmax_dict['channel_{}'.format(channel)][1] = max_batch
    return minmax_dict