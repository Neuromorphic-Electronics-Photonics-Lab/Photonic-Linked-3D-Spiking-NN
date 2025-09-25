from mlflow import log_metric, log_param, log_artifacts
import torch
from rockpool.nn.modules import LIFTorch, LIFBitshiftTorch
from rockpool.nn.modules import LIFSlayer
from rockpool.parameters import Constant

# filter bank parameter
filterbank_params = {"sampling_freq": 16000., 
                     "order": 2,
                     "low_freq": 100.,
                     "num_filters": 32,
                     "rectify": True}

# spike conv params
spike_conversion_params = {"shape": (filterbank_params['num_filters'], filterbank_params['num_filters']),
                           "tau_mem": 0.02,
                           "tau_syn": 0.002,
                           "bias": 0.0,
                           "has_rec": False,
                           "threshold": 0.4,
                           "dt": 1e-3, #realted to T
                           "noise_std": 0.0}

# simulation parameter
dataloader_params = {'batch_size': 256,
                     'num_workers': 8,
                     'shuffle_data': True,
                     'test_shuffle': False,
                     'cache_prefix': '.'}

load_dataloader_params = {'batch_size': 256,
                     'num_workers': 8,
                     'shuffle_data': True,
                     'test_shuffle': False,
                     'cache_prefix': '.'}


training_params = {"lr": 1.5e-3,
                   "num_epochs": 150, #20
                   "num_e_1": 20,
                   "quant_bit": 5,
                   "finetune_epochs": 50,
                   "device": 'cuda:0' if torch.cuda.is_available() else 'cpu'}

'''
# model params --heysnips
# model_params = {"dilations": [2, 4, 8, 16, 2, 4, 8, 16],
#                 "n_classes": 4,
#                 "n_channels_in": filterbank_params['num_filters'],
#                 "n_channels_res": 16,
#                 "n_channels_skip": 32,
#                 "n_hidden": 32,
#                 "kernel_size": 2,
#                 "bias": Constant(0.0),
#                 "smooth_output": False,
#                 "tau_mem": Constant(0.02),
#                 "base_tau_syn": Constant(0.02),
#                 "tau_lp": Constant(0.02),
#                 "threshold": Constant(1.0),
#                 "neuron_model": LIFSlayer,
#                 "dt": 0.01}'''

# # model params --GSC
model_params = {#"dilations": [2, 4, 8, 16],
                "dilations": [2, 4, 8, 16, 2, 4, 8, 16],
                "n_classes": 4,
                "n_channels_in": filterbank_params['num_filters'],
                "n_channels_res": 32,
                "n_channels_skip": 4,
                "n_hidden": 4,
                "kernel_size": 2,
                "bias": Constant(0.0),
                "smooth_output": False,
                "tau_mem": Constant(0.02),
                "base_tau_syn": Constant(0.02),
                "tau_lp": Constant(0.02),
                "threshold": Constant(1.0),
                "neuron_model": LIFTorch,
                "dt": 0.005} # T = 1/dt

load_model_params = {#"dilations": [2, 4, 8, 16],
                "dilations": [2, 4, 8, 16, 2, 4, 8, 16],
                "n_classes": 4,
                "n_channels_in": filterbank_params['num_filters'],
                "n_channels_res": 32,
                "n_channels_skip": 4,
                "n_hidden": 4,
                "kernel_size": 2,
                "bias": Constant(0.0),
                "smooth_output": False,
                "tau_mem": Constant(0.02),
                "base_tau_syn": Constant(0.02),
                "tau_lp": Constant(0.02),
                "threshold": Constant(1.0),
                "neuron_model": LIFTorch,
                "dt": 0.005} # T = 1/dt


log_param("filterbank_params", filterbank_params)
log_param("spike_conversion_params", spike_conversion_params)
log_param("dataloader_params", dataloader_params)
log_param("training_params", training_params)
log_param("model_params", model_params)
