from model import *
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from spikingjelly.clock_driven import functional

# Load Dataset
train_dataset = MNIST(root='./dataset', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='./dataset', train=False, transform=ToTensor())

# Create Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)

# Create model, criterion and optimizer
model = ConvNet().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLRWarmup(optimizer,
                                        T_max=num_epochs,
                                        eta_min=1e-5,
                                        last_epoch=-1,
                                        warmup_steps=0,
                                        warmup_start_lr=1.0e-5)


# Training
def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.cuda())
            loss = criterion(output, target.cuda())
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item()}')
    
    print('Training complete.')

# Test
def test_model():
    global best_acc
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            functional.reset_net(model)
    
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * acc:.0f}%)')
    
    if acc > best_acc:
        print(f'Saving checkpoint with accuracy {acc:.2f}%...')
        best_acc = acc
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }
        torch.save(checkpoint, 'mnist_checkpoint.pt')

import os
import numpy as np
import random

seed = 42
best_acc = 0.0
num_epochs = 100

# Fix random seed
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Start training
for i in range(num_epochs):
    print(f"----------------Epoch {i}----------------")
    # Train and Test the model
    train_model(1)
    scheduler.step()
    test_model()