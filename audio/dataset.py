import pathlib
import json
import sys

import numpy as np
from typing import Tuple, Optional, Callable
import librosa
import os
DATA_PATH = pathlib.Path(__file__).absolute().parent / "dataset/speech_commands_v2"

print(DATA_PATH)
test_list=[]
val_list=[]
with open(os.path.join(DATA_PATH,'testing_list.txt') ,'r') as f:
    data=f.readlines()
    for i in data:
        test_list.append(i.strip())
        # sys.exit()
with open(os.path.join(DATA_PATH,'validation_list.txt') ,'r') as f:
    data=f.readlines()
    for i in data:
        val_list.append(i.strip())

unread=['_background_noise_','LICENSE','README.md','testing_list.txt','validation_list.txt','.DS_Store']

data = 0
# print(test_list)
class GSC:
    mapping = {'cat':1, 'dog':2, 'house':3}#,'two':2,'zero':3,'down':4,'off':5}
    for category in os.listdir(DATA_PATH):
        if category not in unread and category not in mapping.keys():
            mapping.setdefault(category,0)
    print(mapping)
    def __init__(self, partition: str = "train",transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):

        self.sampling_freq = 16000.0
        self.transform = transform
        self.target_transform = target_transform

        self.files = []
        self.labels = []
        self.path=DATA_PATH
        for category in os.listdir(self.path):
            if category not in unread:
                for wav in os.listdir(self.path / category):
                    # print(os.path.join(category , wav),test_list[0])
                    # sys.exit()
                    if (partition == "train") and (os.path.join(category , wav) not in test_list) and (os.path.join(category , wav) not in val_list):

                        self.add_sample_target(self.path / category / wav, category)

                    elif (partition == "test") and os.path.join(category , wav) in test_list:

                        self.add_sample_target(self.path / category / wav, category)

                    elif (partition == "dev") and os.path.join(category , wav) in val_list:

                        self.add_sample_target(self.path / category / wav, category)
        print(len(self.files),len(self.labels))

        # sys.exit()

    def add_sample_target(self, file, strlabel):
        self.files.append(file)
        self.labels.append(self.mapping.get(strlabel))

    def __getitem__(self, item) -> Tuple[np.ndarray, int]:
        signal, fs = librosa.load(self.files[item], sr=16000)
        target = self.labels[item]

        if signal.ndim == 1:
            signal = signal[None, ...]  # Introduce a channel dimension
        if self.transform:
            signal = self.transform(signal)
        if self.target_transform:
            target = self.target_transform(target)
        return signal, target


    def __len__(self) -> int:
            return len(self.files)
            
    