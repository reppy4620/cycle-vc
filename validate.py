import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from models import Generator
from hifi_gan import load_hifi_gan
from transform import TacotronSTFT

SR = 24000


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--hifi_gan', type=str, required=True)
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--tgt_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    config = OmegaConf.load(f'{args.model_dir}/config.yaml')

    src_dir = Path(args.src_dir)
    tgt_dir = Path(args.tgt_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f'{args.model_dir}/latest.ckpt', map_location=device)
    g_xy = Generator(n_mel=config.data.n_mel, **config.model.generator)
    g_xy.load_state_dict(checkpoint['g_xy'])
    print(f'Loaded {checkpoint["iteration"]} Iteration Model')
    hifi_gan = load_hifi_gan(args.hifi_gan)
    g_xy, hifi_gan = g_xy.eval().to(device), hifi_gan.eval().to(device)

    def infer(x):
        x = x.to(device)
        with torch.no_grad():
            y = g_xy(x)
            y_wav = hifi_gan(y)
            y, y_wav = y.cpu(), y_wav.squeeze(1).cpu()
        return y, y_wav

    def save_wav(wav, path):
        torchaudio.save(
            str(path),
            wav,
            24000,
            encoding='PCM_S',
            bits_per_sample=16
        )

    def save_mel_three_attn(src, tgt, gen, path):
        plt.figure(figsize=(20, 7))
        plt.subplot(311)
        plt.gca().title.set_text('MSK')
        plt.imshow(src, aspect='auto', origin='lower')
        plt.subplot(312)
        plt.gca().title.set_text('JSUT')
        plt.imshow(tgt, aspect='auto', origin='lower')
        plt.subplot(313)
        plt.gca().title.set_text('GEN')
        plt.imshow(gen, aspect='auto', origin='lower')
        plt.savefig(path)
        plt.close()

    src_files = list(sorted(src_dir.glob('*.wav')))[config.data.x_train_length:]
    tgt_files = list(sorted(tgt_dir.glob('*.wav')))[config.data.y_train_length:]

    to_mel = TacotronSTFT()

    for i in tqdm(range(len(src_files))):
        src_wav, _ = torchaudio.load(src_files[i])
        tgt_wav, _ = torchaudio.load(tgt_files[i])
        src_mel = to_mel(src_wav)
        tgt_mel = to_mel(tgt_wav)
        mel_gen, wav_gen = infer(src_mel)

        d = output_dir / os.path.splitext(src_files[i].name)[0]
        d.mkdir(exist_ok=True)

        save_wav(src_wav, d / 'src.wav')
        save_wav(tgt_wav, d / 'tgt.wav')
        save_wav(wav_gen, d / 'gen.wav')

        save_mel_three_attn(
            src_mel.squeeze(),
            tgt_mel.squeeze(),
            mel_gen.squeeze(),
            d / 'comp.png'
        )


if __name__ == '__main__':
    main()
