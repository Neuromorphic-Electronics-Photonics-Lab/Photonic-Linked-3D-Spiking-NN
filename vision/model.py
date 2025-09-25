from quantization import *
import torch.nn as nn


from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
)


backend = 'torch'

class ConvNet(nn.Module):
    def __init__(self):
        tau_define = 29.744 # device parameter
        super(ConvNet, self).__init__()

        self.conv1_pos = QuantConv2d_5bit(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_neg = QuantConv2d_5bit(1, 64, kernel_size=3, stride=2, padding=1)
        self.lif1_pos = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.lif1_neg = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.relu1 = nn.ReLU()

        self.conv2_pos = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_neg = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lif2_pos = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.lif2_neg = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.relu2 = nn.ReLU()

        self.conv3_pos = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_neg = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lif3_pos = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.lif3_neg = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.relu3 = nn.ReLU()

        self.conv4_pos = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4_neg = QuantConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.lif4_pos = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.lif4_neg = MultiStepLIFNode(tau=tau_define, detach_reset=True, backend=backend, decay_input=True)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)
        self.T = 10 #timestep

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1) #T, B, C, H, W
        else:
            x = x.transpose(0, 1).contiguous() #

        T, B, _, _, _ = x.shape

        
        #encoding
        x_pos = self.conv1_pos(x.flatten(0,1))
        x_neg = self.conv1_neg(x.flatten(0,1))
        _, C, H, W = x_pos.shape
        x_pos = self.lif1_pos(x_pos.reshape(T, B, C, H, W).contiguous())
        x_neg = self.lif1_neg(x_neg.reshape(T, B, C, H, W).contiguous())
        x = (x_pos - x_neg) # Differential Neuron Bank
        x = self.relu1(x) # Equivalent forms of the Heaviside function
        
        
        x = x.flatten(0,1)
        x_pos = self.conv2_pos(x)
        x_neg = self.conv2_neg(x)
        _, C, H, W = x_pos.shape
        x_pos = self.lif2_pos(x_pos.reshape(T, B, C, H, W).contiguous())
        x_neg = self.lif2_neg(x_neg.reshape(T, B, C, H, W).contiguous())
        x = x_pos - x_neg # Differential Neuron Bank
        x = self.relu2(x) # Equivalent forms of the Heaviside function

        x = x.flatten(0,1)
        x_pos = self.conv3_pos(x)
        x_neg = self.conv3_neg(x)
        _, C, H, W = x_pos.shape
        x_pos = self.lif3_pos(x_pos.reshape(T, B, C, H, W).contiguous())
        x_neg = self.lif3_neg(x_neg.reshape(T, B, C, H, W).contiguous())
        x = x_pos - x_neg # Differential Neuron Bank
        x = self.relu3(x) # Equivalent forms of the Heaviside function


        x = x.flatten(0,1)
        x_pos = self.conv4_pos(x)
        x_neg = self.conv4_neg(x)
        _, C, H, W = x_pos.shape
        x_pos = self.lif4_pos(x_pos.reshape(T, B, C, H, W).contiguous())
        x_neg = self.lif4_neg(x_neg.reshape(T, B, C, H, W).contiguous())
        x = x_pos - x_neg # Differential Neuron Bank
        x = self.relu4(x) # Equivalent forms of the Heaviside function
        
        x = x.mean(0) # post processing
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc(x)
        x = self.fc2(x)
        return x
