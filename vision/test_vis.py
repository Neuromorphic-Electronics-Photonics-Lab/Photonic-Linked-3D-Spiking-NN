from model import *
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Any

class HookManager:
    def __init__(
        self,
        model: nn.Module,
        keep_history: bool = False,
        to_cpu: bool = True,
        detach: bool = True,
        clone: bool = False,
        reduce_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ):
        """
            model: the model to attach hooks to
            keep_history: if True, each forward pass output for that layer is appended to a list; if False, only the most recent output is stored
            to_cpu: move outputs to CPU (saves GPU memory and eases post-processing)
            detach: call detach() to break the computation graph (useful for analysis/visualization)
            clone: call clone() to avoid cached tensors being affected by in-place modifications
            reduce_fn: optional tensor post-processing function, e.g., lambda t: t.float().mean(dim=0)
        """
        self.model = model
        self.keep_history = keep_history
        self.to_cpu = to_cpu
        self.detach = detach
        self.clone = clone
        self.reduce_fn = reduce_fn

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache: Dict[str, Any] = defaultdict(list) if keep_history else {}
        self._name_to_module: Dict[str, nn.Module] = {}

    def _wrap_tensor(self, out: Any) -> Any:
        def process_tensor(t: torch.Tensor) -> torch.Tensor:
            if self.detach:
                t = t.detach()
            if self.clone:
                t = t.clone()
            if self.to_cpu:
                t = t.cpu()
            return t

        if isinstance(out, torch.Tensor):
            t = process_tensor(out)
            return self.reduce_fn(t) if self.reduce_fn else t

        if isinstance(out, (list, tuple)):
            proc = [self._wrap_tensor(x) for x in out]
            return type(out)(proc)

        if isinstance(out, dict):
            return {k: self._wrap_tensor(v) for k, v in out.items()}

        return out

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            val = self._wrap_tensor(output)
            if self.keep_history:
                self._cache[name].append(val)
            else:
                self._cache[name] = val
        return hook

    def register(self, mapping: Dict[str, nn.Module]) -> None:
        """
        mapping: { "key": obj }
        """
        # Avoid duplicate registration
        self.remove()
        self._name_to_module = dict(mapping)

        for name, m in self._name_to_module.items():
            handle = m.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def get(self, name: str) -> Any:
        if name not in self._cache:
            raise KeyError(f"No cached output for key '{name}'. Did you run a forward pass and register the hook?")
        return self._cache[name]

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def keys(self):
        return list(self._name_to_module.keys())

    def clear(self) -> None:
        if self.keep_history:
            for k in list(self._cache.keys()):
                self._cache[k].clear()
        else:
            self._cache.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "keys": self.keys(),
            "keep_history": self.keep_history,
            "to_cpu": self.to_cpu,
            "detach": self.detach,
            "clone": self.clone,
            "has_cache": {k: (len(v) if isinstance(v, list) else 1) for k, v in self._cache.items()},
        }

    def register_default(self) -> None:
        
        m = self.model
        mapping = {}
        # conv{1..4}_{pos,neg}
        for i in range(1, 5): # Range can be modified
            # Comment to ignore hooking
            mapping[f"conv{i}_pos"] = getattr(m, f"conv{i}_pos") # input of pos bank
            mapping[f"conv{i}_neg"] = getattr(m, f"conv{i}_neg") # input of neg bank
            
        # lif{1..4}_{pos,neg} and relu{1..4}
        for i in range(1, 5): # Range can be modified
            # Comment to ignore hooking
            mapping[f"lif{i}_pos"] = getattr(m, f"lif{i}_pos") # expected output of pos bank
            mapping[f"lif{i}_neg"] = getattr(m, f"lif{i}_neg") # expected output of neg bank
            mapping[f"relu{i}"] = getattr(m, f"relu{i}") # output of differential neuron bank

        self.register(mapping)
        

def visualize_feature_maps(feature_maps, id, vis_channel_idx =0, average = False):
    '''
        feature_maps: Tensor/array of feature maps (with time and spatial dims) to visualize.
        id: Identifier appended to the output PNG filenames.
        vis_channel_idx: Channel index to visualize from the feature maps.
        average: If True, average over timesteps and save one image; if False, save one image per timestep.
    '''
    kernel_size = feature_maps.shape[-1]
    act_shape = int(kernel_size)
    
    if average == True: # Average over all timesteps
        feature_maps = feature_maps.mean(0).squeeze()
        fig, ax = plt.subplots(figsize=(act_shape, act_shape), gridspec_kw=dict(wspace=0.1, hspace=0.1))
        ax.set_facecolor('white')
        im = ax.imshow(feature_maps[vis_channel_idx].squeeze().detach().cpu().numpy(), cmap='viridis')

        ax.grid(which="minor", color='w', linestyle='-', linewidth=2)
        ax.set_xticks(np.arange(-0.5, kernel_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, kernel_size, 1), minor=True)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                text = ax.text(j, i, f"{feature_maps[vis_channel_idx][i, j]:.2f}", ha="center", va="center", color="white", fontsize=12)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig(f'Average_feature_map_{id}.png')  # save img
        plt.clf()
    else: # visualization for each time step
        feature_maps = feature_maps.squeeze().detach().cpu().numpy()
        feature_maps = feature_maps[:,vis_channel_idx,:,:]
        min = feature_maps.min()
        max = feature_maps.max()
        if max != min:
            feature_maps = (feature_maps - min) / (max - min)
        else:
            feature_maps = np.zeros_like(feature_maps)
        
        for timestep in range(feature_maps.shape[0]):
            _feature_maps = feature_maps[timestep]
            fig, ax = plt.subplots(figsize=(act_shape, act_shape), gridspec_kw=dict(wspace=0.1, hspace=0.1))
            ax.set_facecolor('white')
            im = ax.imshow(_feature_maps, cmap='viridis')

            ax.grid(which="minor", color='w', linestyle='-', linewidth=2)
            ax.set_xticks(np.arange(-0.5, kernel_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, kernel_size, 1), minor=True)
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    text = ax.text(j, i, f"{_feature_maps[i, j]:.2f}", ha="center", va="center", color="white", fontsize=12)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.savefig(f'Per_Time_feature_map_T{timestep}_{id}.png')  # save img

            plt.clf()
            
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
    
    
# Create Model
model = ConvNet().cuda()

# Load Checkpoints
checkpoint_path = "./checkpoint/mnist_checkpoint.pt"
checkpoint = torch.load(checkpoint_path)
print(checkpoint['best_acc']) # the recorded best acc
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

# Fix Random Seeds
import os
import numpy as np
import random
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

# Add Hooks
hooks = HookManager(model, keep_history=True, to_cpu=True, detach=True) 
hooks.register_default()
# Check the hook keys
print(hooks.keys())

available_keys = ['conv1_pos', 'conv1_neg', 'conv2_pos', 'conv2_neg', 
                  'conv3_pos', 'conv3_neg', 'conv4_pos', 'conv4_neg', 
                  'lif1_pos', 'lif1_neg', 'relu1', 'lif2_pos', 
                  'lif2_neg', 'relu2', 'lif3_pos', 'lif3_neg',
                  'relu3', 'lif4_pos', 'lif4_neg', 'relu4']

#################################################################################
# Visualizaion Example
#################################################################################

# Specify the conv channel for visualization
vis_channel = 10 
    
# Run inference
image_id = 1000
test_dataset = MNIST(root='./dataset', train=False, transform=ToTensor())
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
iteration = iter(test_loader)

for i in range(image_id):
    image, label = next(iteration)
    
plt.imshow(image.squeeze(), cmap='viridis')
plt.axis('off') 
plt.savefig(f'orig_{image_id}.png') # save raw input

_ = model(image.cuda())
T = model.T 

featuremaps = torch.cat(hooks[available_keys[0]]) 
if len(featuremaps.shape) == 4:
    featuremaps = featuremaps.reshape(T, -1, *featuremaps.shape[1:])
    
hooks.clear() # clear memory
hooks.remove() # clear hook

#for i, feature in enumerate(featuremaps):
visualize_feature_maps(feature, image_id , vis_channel_idx, average = False)
    
