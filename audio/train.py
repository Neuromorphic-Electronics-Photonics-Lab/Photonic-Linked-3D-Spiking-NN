import sys
import torch
sys.path.append("src")
from params import *
from dataloaders import create_dataloaders_GSC
import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import warnings
from torch.nn import CrossEntropyLoss
warnings.filterwarnings("ignore")
import random
from torch.optim import AdamW
import os

from model import WaveSenseNet_Hardware
from utils import *

#GSC 2 classes
train_dataloader, val_dataloader, test_dataloader = create_dataloaders_GSC(filterbank_params,
                                           spike_conversion_params,
                                           model_params,
                                           dataloader_params)

savename = "audio_checkpoint.pth" 
lr = training_params['lr']
epoch_num = training_params['num_epochs']
finetune_epochs = training_params['finetune_epochs']
quant_bit = training_params["quant_bit"]
device = training_params['device']
print('save: '+savename, 'epoch: '+ str(epoch_num))
csv_filename = f'train_{lr}_log.csv'



# Fix Seeds

seed = 42
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###########Start Training#####
best_vala_acc=0
loss_data=[]
tf1_data=[]
vf1_data=[]

best_vala_acc = float('-inf')

model = WaveSenseNet_Hardware(**model_params)
model= change_hp(model)
model.to(training_params['device'])
criterion = CrossEntropyLoss()
opt = AdamW(model.parameters().astorch(), lr=lr)



scheduler = CosineAnnealingLRWarmup(opt,
                                    T_max=epoch_num,
                                    eta_min=5.0e-4,
                                    last_epoch=-1,
                                    warmup_steps=0,
                                    warmup_start_lr=1.0e-5)


for epoch in range(epoch_num):
    train_preds = []
    train_targets = []
    sum_loss = 0.0
    time = 0

    # Loop over training batches
    for batch, target in tqdm.tqdm(train_dataloader):
        batch = batch.to(training_params['device'])
        target = target.to(training_params['device'])

        # Reset model state and gradients
        model.reset_state()
        opt.zero_grad()

        # Forward
        out, _, rec = model(batch.clip(0, 15), record=True)
        # We use the synaptic currents of the output layer for training
        out = rec['spk_out']['spikes'].squeeze()
        ans=[]

        peaks = out.mean(1)

        # Regularizer based on activity level in the network
        reg = 1e-3 * (get_spike_count(rec) ** 2)
        
        # Loss = CE + Reg
        loss = criterion(peaks, target) + reg
        
        # Calculate gradients
        loss.backward()

        # Gradient step
        opt.step()

        # Store targets and predictions for later use
        with torch.no_grad():
            pred = peaks.argmax(1).detach()
            train_preds += pred.detach().cpu().numpy().tolist()
            train_targets += target.detach().cpu().numpy().tolist()
            sum_loss += loss.item()

    # Calculate F1 score
    f1 = f1_score(train_targets,
                train_preds,
                average="macro")

    scheduler.step()

    print(f"TRAIN Epoch {epoch} Loss {sum_loss} F1 Score {f1}")
    
    loss_data.append(sum_loss)
    tf1_data.append(f1)
    
    val_preds = []
    val_targets = []
    # Loop over validation batches
    for batch, target in tqdm.tqdm(val_dataloader):

        with torch.no_grad():
            time += 1
            batch = batch.to('cuda:0')
            target = target.to('cuda:0')

            # Reset model
            model.reset_state()

            # Forward
            out, _, rec = model(batch.clip(0, 15), record=True)
            out = rec['spk_out']['spikes'].squeeze()
            
            peaks = out.mean(1)

            pred = peaks.argmax(1).detach()
            # Save predictions and targets for later use
            val_preds += pred.detach().cpu().numpy().tolist()
            val_targets += target.detach().cpu().numpy().tolist()

    acc = accuracy_score(val_targets,
                val_preds
                )
    
    vf1_data.append(acc)
    write_data_to_csv([sum_loss/time, acc], csv_filename)
    # Save best model so far
    if acc > best_vala_acc:
        best_vala_acc = acc
    
    if acc > 0.9:
        model.save(savename)
        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': sum_loss,
            'f1_score': acc
        }, f"checkpoints/Epoch_{epoch}_{savename}")
        print(f"Checkpoint saved for epoch {epoch} with F1 Score: {acc} (Epoch_{epoch}_{savename})")

    print(f"VAL Epoch {epoch} ACC {acc}")
