import random
import torch
from torch.utils.data import Dataset

import torchaudio

from transform import TacotronSTFT


class VCDataset(Dataset):
    def __init__(self, x_files, y_files, x_stats, y_stats, segment_length):
        self.x_files = x_files
        self.y_files = y_files

        self.x_length = len(x_files)
        self.y_length = len(y_files)

        self.x_stats = x_stats
        self.y_stats = y_stats

        self.segment_length = segment_length

        self.to_mel = TacotronSTFT()

        self.cnt = 0

    def __len__(self):
        return self.x_length if self.x_length > self.y_length else self.y_length

    def rand_slice(self, length):
        b = random.randint(0, length - self.segment_length)
        e = b + self.segment_length
        return b, e

    def load_mel(self, wav_path, stats):
        wav, sr = torchaudio.load(wav_path)
        wav = wav + torch.rand_like(wav) / 32768.0
        mel = self.to_mel(wav).squeeze()
        mel = (mel - stats['mean']) / stats['std']
        b, e = self.rand_slice(mel.size(-1))
        return mel[:, b:e]

    def __getitem__(self, idx):
        if self.cnt == 0:
            random.shuffle(self.y_files)
        self.cnt = (self.cnt + 1) % self.y_length
        if self.x_length > self.y_length:
            x_path = self.x_files[idx]
            y_path = self.y_files[idx % self.y_length]
        else:
            x_path = self.x_files[idx % self.x_length]
            y_path = self.y_files[idx]
        x = self.load_mel(x_path, self.x_stats)
        y = self.load_mel(y_path, self.y_stats)
        return x, y
