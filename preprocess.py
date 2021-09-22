import torch
import torchaudio

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler

from transform import TacotronSTFT


class PreProcessor:
    def __init__(self, config):
        self.src_dir = Path(config.src_dir)
        self.tgt_dir = Path(config.tgt_dir)

        self.src_train_length = config.src_train_length
        self.tgt_train_length = config.tgt_train_length

        self.to_mel = TacotronSTFT()

    def process_dir(self, speaker_dir: Path, train_length: int):
        stats_dir = speaker_dir.parent / 'mel_stats'
        stats_dir.mkdir(exist_ok=True)
        wav_files = list(sorted(speaker_dir.glob('*.wav')))
        scaler = StandardScaler()

        for i, fp in tqdm(enumerate(wav_files), total=len(wav_files)):
            wav, _ = torchaudio.load(fp)
            mel = self.to_mel(wav).squeeze()
            if i < train_length:
                scaler.partial_fit(mel.view(-1, 1).numpy())

        stats = {
            'mean': scaler.mean_[0],
            'std': scaler.scale_[0]
        }

        torch.save(stats, stats_dir / 'stats.pt')

    def run(self):
        self.process_dir(self.src_dir, self.src_train_length)
        self.process_dir(self.tgt_dir, self.tgt_train_length)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/preprocess.yaml')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    PreProcessor(config).run()
