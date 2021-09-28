import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data import VCDataset
from .generator import Generator
from .discriminator import Discriminator
from .loss import d_loss, g_loss, feature_map_loss
from .lr_scheduler import NoamLR
from utils import seed_everything, Tracker


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        config = OmegaConf.load(self.config_path)

        accelerator = Accelerator(fp16=config.train.fp16)

        seed_everything(config.seed)

        output_dir = Path(config.model_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(config, output_dir / 'config.yaml')

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=f'{str(output_dir)}/logs')
        else:
            writer = None

        train_data, valid_data = self.prepare_data(config.data)
        train_dataset = VCDataset(*train_data, segment_length=config.data.segment_length)
        valid_dataset = VCDataset(*valid_data, segment_length=config.data.segment_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=8
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.train.batch_size,
            num_workers=8
        )

        g_xy = Generator(n_mel=config.data.n_mel, **config.model.generator)
        g_yx = Generator(n_mel=config.data.n_mel, **config.model.generator)
        d_x = Discriminator(**config.model.discriminator)
        d_y = Discriminator(**config.model.discriminator)
        g_params = list(g_xy.parameters()) + list(g_yx.parameters())
        d_params = list(d_x.parameters()) + list(d_y.parameters())

        optimizer_g = optim.AdamW(g_params, eps=1e-9, **config.optimizer)
        optimizer_d = optim.AdamW(d_params, eps=1e-9, **config.optimizer)

        epochs = self.load(config, g_xy, g_yx, d_x, d_y, optimizer_g, optimizer_d)

        g_xy, g_yx, d_x, d_y, optimizer_g, optimizer_d, train_loader, valid_loader = accelerator.prepare(
            g_xy, g_yx, d_x, d_y, optimizer_g, optimizer_d, train_loader, valid_loader
        )

        scheduler_g = NoamLR(optimizer_g, channels=config.model.generator.base_channels, last_epoch=epochs * len(train_loader) - 1)
        scheduler_d = NoamLR(optimizer_d, channels=config.model.discriminator.base_channels, last_epoch=epochs * len(train_loader) - 1)

        for epoch in range(epochs, config.train.num_epochs):
            self.train_step(
                config,
                epoch,
                [g_xy, g_yx, d_x, d_y],
                [optimizer_g, optimizer_d],
                [scheduler_g, scheduler_d],
                train_loader,
                writer,
                accelerator
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.valid_step(config, epoch, [g_xy, g_yx, d_x, d_y], valid_loader, writer)
                if (epoch + 1) % config.train.save_interval == 0:
                    self.save(
                        output_dir / 'latest.ckpt',
                        epoch,
                        (epoch+1)*len(train_loader),
                        accelerator.unwrap_model(g_xy),
                        accelerator.unwrap_model(g_yx),
                        accelerator.unwrap_model(d_x),
                        accelerator.unwrap_model(d_y),
                        optimizer_g,
                        optimizer_d
                    )
        if accelerator.is_main_process:
            writer.close()

    def train_step(self, config, epoch, models, optimizers, schedulers, loader, writer, accelerator):
        tracker = Tracker()
        bar = tqdm(desc=f'Epoch: {epoch + 1}', total=len(loader), disable=not accelerator.is_main_process)
        for i, batch in enumerate(loader):
            self._handle_batch(config, tracker, batch, models, optimizers, schedulers, accelerator)
            bar.update()
            self.set_loss(bar, tracker)
        self.set_loss(bar, tracker)
        if accelerator.is_main_process:
            self.write_losses(epoch, writer, tracker, mode='train')
        bar.close()

    @torch.no_grad()
    def valid_step(self, config, epoch, models, loader, writer):
        tracker = Tracker()
        for i, batch in enumerate(loader):
            self._handle_batch(config, tracker, batch, models)
        self.write_losses(epoch, writer, tracker, mode='valid')

    def _handle_batch(self, config, tracker, batch, models, optimizers=None, schedulers=None, accelerator=None):
        g_xy, g_yx, d_x, d_y = models
        if optimizers is not None:
            g_xy.train(); g_yx.train(); d_x.train(); d_y.train()
            optimizer_g, optimizer_d = optimizers
            scheduler_g, scheduler_d = schedulers
        else:
            g_xy.eval(); g_yx.eval(); d_x.eval(); d_y.eval()

        x, y = batch
        x_fake = g_yx(y)
        y_fake = g_xy(x)

        cycle_x = g_yx(y_fake)
        cycle_y = g_xy(x_fake)

        id_x = g_yx(x)
        id_y = g_xy(y)

        # D
        pred_x_real, _, recon_x = d_x(x, need_recon=True)
        pred_x_fake, *_ = d_x(x_fake.detach(), need_recon=False)
        pred_y_real, _, recon_y = d_y(y, need_recon=True)
        pred_y_fake, *_ = d_y(y_fake.detach(), need_recon=False)

        loss_d_x = d_loss(pred_x_real, pred_x_fake)
        loss_d_y = d_loss(pred_y_real, pred_y_fake)
        loss_recon = F.l1_loss(x, recon_x) + F.l1_loss(y, recon_y)
        loss_d = loss_d_x + loss_d_y + loss_recon
        if optimizers is not None:
            optimizer_d.zero_grad()
            accelerator.backward(loss_d)
            accelerator.clip_grad_orm_(d_x.parameters(), 5)
            accelerator.clip_grad_norm_(d_y.parameters(), 5)
            optimizer_d.step()
            scheduler_d.step()

        # G
        _, fm_x_real, _ = d_x(x, need_recon=False)
        pred_x_fake, fm_x_fake, _ = d_x(x_fake, need_recon=False)
        _, fm_y_real, _ = d_y(y, need_recon=False)
        pred_y_fake, fm_y_fake, _ = d_y(y_fake, need_recon=False)
        loss_g_x_gan = g_loss(pred_x_fake)
        loss_g_y_gan = g_loss(pred_y_fake)
        loss_g_gan = loss_g_x_gan + loss_g_y_gan
        loss_fm_x = feature_map_loss(fm_x_real, fm_x_fake)
        loss_fm_y = feature_map_loss(fm_y_real, fm_y_fake)
        loss_fm = loss_fm_x + loss_fm_y
        loss_cycle = F.l1_loss(cycle_x, x) + F.l1_loss(cycle_y, y)
        loss_id = F.l1_loss(id_x, x) + F.l1_loss(id_y, y)
        loss_g = loss_g_gan + loss_fm + config.train.lambda_cycle * loss_cycle + config.train.lambda_id * loss_id
        if optimizers is not None:
            optimizer_g.zero_grad()
            accelerator.backward(loss_g)
            accelerator.clip_grad_norm_(g_xy.parameters(), 5)
            accelerator.clip_grad_norm_(g_yx.parameters(), 5)
            optimizer_g.step()
            scheduler_g.step()

        tracker.update(
            loss_d=loss_d.item(),
            loss_recon=loss_recon.item(),
            loss_g=loss_g.item(),
            loss_g_gan=loss_g_gan.item(),
            loss_fm=loss_fm.item(),
            loss_cycle=loss_cycle.item(),
            loss_id=loss_id.item()
        )

    def prepare_data(self, data_config):
        x_dir = Path(data_config.x_dir)
        y_dir = Path(data_config.y_dir)
        assert x_dir.exists() and y_dir.exists()

        x_files = list(sorted(x_dir.glob('*.wav')))
        y_files = list(sorted(y_dir.glob('*.wav')))

        x_train, x_valid = x_files[:data_config.x_train_length], x_files[data_config.x_train_length:]
        y_train, y_valid = y_files[:data_config.y_train_length], y_files[data_config.y_train_length:]

        x_stats = torch.load(data_config.x_stats)
        y_stats = torch.load(data_config.y_stats)
        return (x_train, y_train, x_stats, y_stats), (x_valid, y_valid, x_stats, y_stats)

    def load(self, config, g_xy, g_yx, d_x, d_y, optimizer_g, optimizer_d):
        if config.resume_checkpoint:
            checkpoint = torch.load(f'{config.model_dir}/latest.ckpt')
            epochs = checkpoint['epoch']
            iteration = checkpoint['iteration']
            g_xy.load_state_dict(checkpoint['g_xy'])
            g_yx.load_state_dict(checkpoint['g_yx'])
            d_x.load_state_dict(checkpoint['d_x'])
            d_y.load_state_dict(checkpoint['d_y'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f'Loaded {iteration}iter model and optimizer.')
            return epochs + 1
        else:
            return 0

    def save(self, save_path, epoch, iteration, g_xy, g_yx, d_x, d_y, optimizer_g, optimizer_d):
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'g_xy': g_xy.state_dict(),
            'g_yx': g_yx.state_dict(),
            'd_x': d_x.state_dict(),
            'd_y': d_y.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict()
        }, save_path)

    def write_losses(self, epoch, writer, tracker, mode='train'):
        for k, v in tracker.items():
            writer.add_scalar(f'{mode}/{k}', v.mean(), epoch)

    def set_loss(self, bar, tracker):
        bar.set_postfix_str(', '.join([f'{k}: {v.mean():.6f}' for k, v in tracker.items()]))
