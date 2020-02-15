import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from AudioReader import AudioReader
import torch.nn.functional as F
import random


def make_dataloader(is_train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Datasets(**data_kwargs)
    return DataLoaders(dataset,
                      is_train=is_train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
    '''

    def __init__(self, mix_scp=None, ref_scp=None, sr=8000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(mix_scp, sample_rate=sr)
        self.ref_audio = [AudioReader(r, sample_rate=sr) for r in ref_scp]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        key = self.mix_audio.keys[index]
        mix = self.mix_audio[key]
        ref = [r[key] for r in self.ref_audio]
        return {
            'mix': mix,
            'ref': ref
        }


class Spliter():
    '''
       Split the audio. All audio is divided 
       into 4s according to the requirements in the paper.
       input:
             chunk_size: split size
             least: Less than this value will not be read
    '''

    def __init__(self, chunk_size=32000, is_train=True, least=16000):
        super(Spliter, self).__init__()
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.least = least

    def chunk_audio(self, sample, start):
        '''
           Make a chunk audio
           sample: a audio sample
           start: split start time
        '''
        chunk = dict()
        chunk['mix'] = sample['mix'][start:start+self.chunk_size]
        chunk['ref'] = [r[start:start+self.chunk_size] for r in sample['ref']]
        return chunk

    def splits(self, sample):
        '''
           Split a audio sample
        '''
        length = sample['mix'].shape[0]
        if length < self.least:
            return []
        audio_lists = []
        if length < self.chunk_size:
            gap = self.chunk_size-length
            sample['mix'] = F.pad(sample['mix'], (0, gap), mode='constant')
            sample['ref'] = [F.pad(r, (0, gap), mode='constant')
                             for r in sample['ref']]
            audio_lists.append(sample)
        else:
            random_start = random.randint(
                0, length % self.least) if self.is_train else 0
            while True:
                if random_start+self.chunk_size > length:
                    break
                audio_lists.append(self.chunk_audio(sample, random_start))
                random_start += self.least
        return audio_lists


class DataLoaders():
    '''
        Custom dataloader method
        input:
              dataset (Dataset): dataset from which to load the data.
              num_workers (int, optional): how many subprocesses to use for data (default: 4)
              chunk_size (int, optional): split audio size (default: 32000(4 s))
              batch_size (int, optional): how many samples per batch to load
              is_train: if this dataloader for training
    '''

    def __init__(self, dataset, num_workers=4, chunk_size=32000, batch_size=1, is_train=True):
        super(DataLoaders, self).__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_train = is_train
        self.data_loader = DataLoader(self.dataset,
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size // 2,
                                      shuffle=self.is_train,
                                      collate_fn=self._collate)
        self.spliter = Spliter(
            chunk_size=self.chunk_size, is_train=self.is_train, least=self.chunk_size // 2)

    def _collate(self, batch):
        '''
            merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        '''
        batch_audio = []
        for b in batch:
            batch_audio += self.spliter.splits(b)
        return batch_audio

    def __iter__(self):
        mini_batch = []
        for batch in self.data_loader:
            mini_batch += batch
            length = len(mini_batch)
            if self.is_train:
                random.shuffle(mini_batch)
            collate_chunk = []
            for start in range(0, length-self.batch_size+1, self.batch_size):
                b = default_collate(
                    mini_batch[start:start+self.batch_size])
                collate_chunk.append(b)
            idx = length % self.batch_size
            mini_batch = mini_batch[-idx:] if idx else []
            for m_batch in collate_chunk:
                yield m_batch # batch of datasets
                '''
                   mini_batch like this
                   'mix': batch x L
                   'ref': [bathc x L, bathc x L]
                '''


if __name__ == "__main__":
    datasets = Datasets('/home/likai/data1/create_scp/cv_mix.scp',
                        ['/home/likai/data1/create_scp/cv_s1.scp', '/home/likai/data1/create_scp/cv_s2.scp'])
    dataloaders = DataLoaders(datasets, num_workers=0,
                              batch_size=10, is_train=False)
    for eg in dataloaders:
        print(eg)
        import pdb
        pdb.set_trace()
