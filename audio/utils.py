from rockpool.nn.modules.sinabs.lif_exodus import LIFSlayer
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import math
import csv
import torch


#### change spike layer to avoid overfitting
class PeriodicalExponential:
    window: float = 0.5

    def __call__(self, v_mem, spike_threshold):
        vmem_shifted = v_mem - spike_threshold / 2
        vmem_periodic = vmem_shifted - torch.div(
            vmem_shifted, spike_threshold, rounding_mode="floor"
        )
        vmem_below = vmem_shifted * (v_mem < spike_threshold)
        vmem_above = vmem_periodic * (v_mem >= spike_threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
                torch.exp(-torch.abs(vmem_new - spike_threshold / 2) / self.window)
                / spike_threshold
        )
        return spikePdf


def change_hp(model):
    for n, m in model.named_modules():
        if isinstance(m, LIFSlayer):
            # m.learning_window = torch.tensor(0.3)
            m.surrogate_grad_fn = PeriodicalExponential()
    return model

def get_spike_count(rec, count=0):
    if 'spikes' in rec.keys():
        # xylo supports up to 15 spikes per timestep. Here we sum up all spikes exeeding that limit
        avg_spikes = torch.nn.functional.relu(rec['spikes'] - 5).mean()
        return count + avg_spikes
    for name, val in rec.items():
        if "output" in name:
            continue
        count = get_spike_count(val, count)

    return count



def write_data_to_csv(data, filename):
    """
    Write a list of data to a CSV file.

    :param data: List of items to be written to the CSV file.
    :param filename: The name of the CSV file to write to.
    """
    # Open the file in append mode ('a') so that data is added at the end.
    # If the file does not exist, it will be created.
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data in a single row.
        writer.writerow(data)
        
class CosineAnnealingLRWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=1.0e-5, last_epoch=-1, verbose=False,
                 warmup_steps=2, warmup_start_lr=1.0e-5):
        super(CosineAnnealingLRWarmup, self).__init__(optimizer,T_max=T_max,
                                                      eta_min=eta_min,
                                                      last_epoch=last_epoch)
        self.warmup_steps=warmup_steps
        self.warmup_start_lr = warmup_start_lr
        if warmup_steps>0:
            self.base_warup_factors = [
                (base_lr/warmup_start_lr)**(1.0/self.warmup_steps)
                for base_lr in self.base_lrs
            ]
 
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()
 
    def _get_closed_form_lr(self):
        if hasattr(self,'warmup_steps'):
            if self.last_epoch<self.warmup_steps:
                return [self.warmup_start_lr*(warmup_factor**self.last_epoch)
                        for warmup_factor in self.base_warup_factors]
            else:
                return [self.eta_min + (base_lr - self.eta_min) *
                        (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)))*0.5
                        for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
 
