#from heysnips import HeySnips
import tonic
import tonic.transforms as transforms
import librosa
from tonic import audio_transforms
from tonic import CachedDataset
from rockpool.nn.modules import LIFTorch, LIFBitshiftTorch
from dataset import GSC
import torch
from torch.utils.data import DataLoader

def create_dataloaders_GSC(filterbank_params,
                       spike_conversion_params,
                       model_params,
                       dataloader_params):
    # Functions for spike conversion
    def _spike_conversion(sample):
        sample = torch.from_numpy(sample.T).to('cpu').unsqueeze(0)
        sample, _, _ = spike_conversion.evolve(sample)
        return sample[0].detach().numpy()

    def _resample(sample):
        return librosa.resample(sample, orig_sr=filterbank_params['sampling_freq'],
                                target_sr=1 / spike_conversion_params['dt'])

    spike_conversion = LIFTorch(**spike_conversion_params)

    preprocess = transforms.Compose(
        [audio_transforms.FixLength(length=int(filterbank_params['sampling_freq']), axis=1),
         audio_transforms.normalize,
         audio_transforms.MelButterFilterBank(**filterbank_params),
         _resample,
         _spike_conversion,
         audio_transforms.Bin(1 / spike_conversion_params['dt'], 1 / model_params['dt'], axis=0), #to model process dt
         ])

    print("Train Dataset Created")
    train_set = CachedDataset(GSC(partition="train",transform=preprocess),
                              cache_path=f"{dataloader_params['cache_prefix']}/train1_cache_{filterbank_params['num_filters']}_{filterbank_params['sampling_freq']}_{spike_conversion_params['dt']}_{model_params['dt']}")
    print("Train Dataloader Created")
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=dataloader_params['batch_size'],
                                  num_workers=dataloader_params['num_workers'],
                                  shuffle=dataloader_params['shuffle_data'])
    print("Val Dataset Created")
    val_set = CachedDataset(GSC(partition="dev",transform=preprocess),
                            cache_path=f"{dataloader_params['cache_prefix']}/val1_cache_{filterbank_params['num_filters']}_{filterbank_params['sampling_freq']}_{spike_conversion_params['dt']}_{model_params['dt']}")
    print("Val Loader Created")
    val_dataloader = DataLoader(dataset=val_set,
                                batch_size=dataloader_params['batch_size'],
                                num_workers=dataloader_params['num_workers'],
                                shuffle=dataloader_params['test_shuffle'])

    test_set = CachedDataset(GSC(partition="test",transform=preprocess),
                             cache_path=f"{dataloader_params['cache_prefix']}/test1_cache_{filterbank_params['num_filters']}_{filterbank_params['sampling_freq']}_{spike_conversion_params['dt']}_{model_params['dt']}")

    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=dataloader_params['batch_size'],
                                 num_workers=dataloader_params['num_workers'],
                                 shuffle=dataloader_params['test_shuffle'])

    return train_dataloader, val_dataloader, test_dataloader