import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import random
from utils import handle_scp
import numpy as np
import soundfile as sf
import torchaudio


def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L 
                     L is the number of audio frames 
                     C is the number of channels. 
                sr: sample rate
    '''
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs, chunk_size=chunk_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=is_train,
                      drop_last=True)


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, mix_scp=None, ref_scp=None, sr=8000, chunk_size=32000):
        super(Datasets, self).__init__()
        self.mix_audio = handle_scp(mix_scp)
        self.ref_audio = [handle_scp(r) for r in ref_scp]
        self.sr = sr
        self.chunk_size = chunk_size
        self.test = self.chunk_size is None
        self.key_tmp = list(self.mix_audio.keys())
        self.key = self.key_tmp.copy()
        for k in self.key_tmp:
            mix = read_wav(self.mix_audio[k])
            if mix.shape[-1] < 32000:
                self.key.remove(k)

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        index = self.key[index]
        mix = read_wav(self.mix_audio[index])
        ref = [read_wav(r[index]) for r in self.ref_audio]
        if mix.shape[-1] == self.chunk_size or self.test:
            rand_start = 0
        else:
            if mix.shape[-1] - self.chunk_size < 0:
                print(index)
            rand_start = np.random.randint(0, mix.shape[-1] - self.chunk_size)
        if self.test:
            stop = None
        else:
            stop = rand_start + self.chunk_size
        mix, _ = sf.read(self.mix_audio[index], start=rand_start,
                         stop=stop, dtype='float32')
        for i in range(len(ref)):
            ref[i], _ = sf.read(self.ref_audio[i][index], start=rand_start,
                                stop=stop, dtype='float32')
        return {
            'mix': torch.from_numpy(mix),
            'ref': [torch.from_numpy(r) for r in ref]
        }


if __name__ == "__main__":
    datasets = Datasets('/home/likai/data1/create_scp/cv_mix.scp',
                        ['/home/likai/data1/create_scp/cv_s1.scp', '/home/likai/data1/create_scp/cv_s2.scp'])
    print(datasets.key.index('012c020o_1.2887_409o0319_-1.2887.wav'))
