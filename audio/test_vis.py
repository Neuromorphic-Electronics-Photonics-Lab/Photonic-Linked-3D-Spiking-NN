import sys
import torch
sys.path.append("src")
from params import *
from dataloaders import create_dataloaders_GSC
from model_for_fintune import WaveSenseNet_Hardware
import tqdm
import numpy as np
from sklearn.metrics import f1_score
import warnings
from torch.nn import CrossEntropyLoss
warnings.filterwarnings("ignore")
import json
import random
from utils import *

import os
import pandas as pd
import matplotlib.pyplot as plt


file_dir = os.path.dirname(__file__)
os.environ['BASE_DIR'] = file_dir

def read_neurons_data(neuron_num, batch_size, time_steps):
    # initialize arrays
    spk2_input_all = np.zeros((batch_size, time_steps, neuron_num))
    spk2_output_all = np.zeros((batch_size, time_steps, neuron_num))
    spkout_input_all = np.zeros((batch_size, time_steps, neuron_num))
    spkout_output_all = np.zeros((batch_size, time_steps, neuron_num))
    
    # read data
    for neuron_idx in range(neuron_num):
        filename = f'./outputs/csvs/neuron_{neuron_idx}.csv'
        df = pd.read_csv(filename)
        
        # restore the data to its original format.
        spk2_input_all[:, :, neuron_idx] = df['spk2_input'].values.reshape(batch_size, time_steps)
        spk2_output_all[:, :, neuron_idx] = df['spk2_output'].values.reshape(batch_size, time_steps)
        spkout_input_all[:, :, neuron_idx] = df['spkout_input'].values.reshape(batch_size, time_steps)
        spkout_output_all[:, :, neuron_idx] = df['spkout_output'].values.reshape(batch_size, time_steps)
        
        print(f"Data for neuron {neuron_idx} loaded from {filename}")
    
    return spk2_input_all, spk2_output_all, spkout_input_all, spkout_output_all
    
model = WaveSenseNet_Hardware(**load_model_params)
model = change_hp(model)
model = model
#print(model)


# Load Checkpoints
checkpoint = torch.load('./checkpoint/audio_checkpoint.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']
#print(checkpoint['f1_score'])
model.load_state_dict(state_dict, strict=True)

train_dataloader, val_dataloader, test_dataloader = create_dataloaders_GSC(filterbank_params,
                                           spike_conversion_params,
                                           load_model_params,
                                           load_dataloader_params)

################################################
# Test and Vis Example
#################################################

# Specify the data to be used for test and vis
start_ind = 130
end_ind = 140
batch_id = 3

# Fetch data
dataiter = iter(val_dataloader)
for i in range(batch_id):
    batch, target = next(dataiter)
    print(f"loading data {i}")
data = batch[start_ind: end_ind]
target = target[start_ind: end_ind]

# Run Inference
model.reset_state()
out, _, rec = model(data.clip(0, 15), record=True)
out = rec['spk_out']['spikes'].squeeze()


# Get Weight
# print("quant_levels:",model.readout.weight_quant.grids)
# print("float_weight:", model.readout.weight.T)
# quant_weight = model.readout.weight_quant(model.readout.weight.T)
# print("quant_weight", quant_weight)


neuron_spk2_input = rec["hidden_output"].detach().cpu().numpy()
neuron_spk2_output = rec['spk2']['spikes'].detach().cpu().numpy()
neuron_spkout_input = rec['readout_output'].detach().cpu().numpy()
neuron_spkout_output = rec['spk_out']['spikes'].detach().cpu().numpy()

# print(neuron_spk2_input.shape)
# print(neuron_spk2_output.shape)
# print(neuron_spkout_input.shape)
# print(neuron_spkout_output.shape)

B, T, neuron_num = neuron_spk2_input.shape
time_steps = range(T)
bar_width = 0.1 

# saving to files
for neuron_idx in range(neuron_num):
    data = {
        'spk2_input': neuron_spk2_input[:, :, neuron_idx].flatten(),
        'spk2_output': neuron_spk2_output[:, :, neuron_idx].flatten(),
        'spkout_input': neuron_spkout_input[:, :, neuron_idx].flatten(),
        'spkout_output': neuron_spkout_output[:, :, neuron_idx].flatten(),
    }
    df = pd.DataFrame(data)

    filename = f'./outputs/csvs/batch{batch_id}_data_{start_ind}-{end_ind}_neuron_{neuron_idx}.csv'
    df.to_csv(filename, index=False)

    print(f"Data for neuron {neuron_idx} saved to {filename}")

# Plot Data
for i in range(B):
    for neuron_idx in range(neuron_num):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Batch {start_ind + i} Neuron {neuron_idx} Activity')
        # plot spk2_input
        axs[0].bar(time_steps, neuron_spk2_input[i, :, neuron_idx], label='spk2_input', width=bar_width)
        axs[0].set_ylabel('spk2_input')
        axs[0].legend()

        # plot spk2_output
        axs[1].bar(time_steps, neuron_spk2_output[i, :, neuron_idx], label='spk2_output', width=bar_width)
        axs[1].set_ylabel('spk2_output')
        axs[1].legend()

        # plot spkout_input
        axs[2].bar(time_steps, neuron_spkout_input[i, :, neuron_idx], label='spkout_input', width=bar_width)
        axs[2].set_ylabel('spkout_input')
        axs[2].legend()

        # plot spkout_output
        axs[3].bar(time_steps, neuron_spkout_output[i, :, neuron_idx], label='spkout_output')
        axs[3].set_ylabel('spkout_output')
        axs[3].legend()

        # set x-axis label
        axs[3].set_xlabel('Time Steps')

        # save fig
        plt.tight_layout()
        plt.savefig(f'./outputs/figures/batch{batch_id}_data_{start_ind + i}_neu_{neuron_idx}_label{target[i]}_pred{pred[i]}.png')