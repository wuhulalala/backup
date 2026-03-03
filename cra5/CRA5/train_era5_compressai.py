import os
import sys
import math
import json
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from cra5.models.compressai.models.google import (
    FactorizedPrior,
    FactorizedPriorReLU,
    ScaleHyperprior,
    MeanScaleHyperprior,
    JointAutoregressiveHierarchicalPriors,
)
from cra5.models.compressai.models.waseda import Cheng2020Anchor, Cheng2020Attention
from cra5.models.compressai.models.tcm2023 import TCM2023
from cra5.models.compressai.models.elic2022 import ELIC2022
from cra5.models.compressai.models.stf2022 import SymmetricalTransFormer2022

VNAMES = dict(
    pressure=['z', 'q', 'u', 'v', 't', 'r', 'w'],
    single=['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'tp', 'msl'],
)

PRESSURE_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]

IN_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])

TOTAL_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]


def get_mean_std():
    api_dir = os.path.join(os.path.dirname(__file__), 'cra5', 'api')
    with open(os.path.join(api_dir, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    with open(os.path.join(api_dir, 'mean_std_single.json'), 'r') as f:
        mean_std_single = json.load(f)

    level_mapping = [TOTAL_LEVELS.index(val) for val in PRESSURE_LEVELS if val in TOTAL_LEVELS]

    mean_list, std_list = [], []
    for vname in VNAMES['pressure']:
        mean_list += [mean_std['mean'][vname][idx] for idx in level_mapping]
        std_list += [mean_std['std'][vname][idx] for idx in level_mapping]
    for vname in VNAMES['single']:
        mean_list.append(mean_std_single['mean'][vname])
        std_list.append(mean_std_single['std'][vname])

    return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)


def read_nc(pressure_file, single_file):
    one_step = []
    pressure_data = xr.open_dataset(pressure_file, engine='netcdf4')
    single_data = xr.open_dataset(single_file, engine='netcdf4')

    pha_levels = list(pressure_data.pressure_level.data)
    level_mapping = [pha_levels.index(val) for val in PRESSURE_LEVELS if val in pha_levels]

    for vname in VNAMES['pressure']:
        D = pressure_data[vname].data
        for level in level_mapping:
            one_step.append(D[0][level][None])

    for vname in VNAMES['single']:
        D = single_data[vname].data
        if vname == 'tp':
            D = D * 1000
        one_step.append(D)

    pressure_data.close()
    single_data.close()
    one_step = np.concatenate(one_step, 0).astype(np.float32)
    return one_step


class ERA5CompressDataset(Dataset):
    def __init__(self, data_root, crop_size=256, cache_data=True):
        self.data_root = data_root
        self.crop_size = crop_size
        self.cache_data = cache_data
        self.mean, self.std = get_mean_std()
        self.data_cache = {}

        self.file_pairs = []
        for root, dirs, files in os.walk(data_root):
            for f in sorted(files):
                if f.endswith('_pressure.nc'):
                    ts = f.replace('_pressure.nc', '')
                    single_f = os.path.join(root, f'{ts}_single.nc')
                    pressure_f = os.path.join(root, f)
                    if os.path.exists(single_f):
                        self.file_pairs.append((pressure_f, single_f, ts))

        self.file_pairs.sort(key=lambda x: x[2])
        if len(self.file_pairs) == 0:
            raise RuntimeError(f"No ERA5 .nc file pairs found in {data_root}")
        logging.info(f"Found {len(self.file_pairs)} ERA5 timestamps")

    def __len__(self):
        return max(len(self.file_pairs), 100)

    def _load(self, idx):
        real_idx = idx % len(self.file_pairs)
        if real_idx in self.data_cache:
            return self.data_cache[real_idx]
        pressure_f, single_f, ts = self.file_pairs[real_idx]
        data = read_nc(pressure_f, single_f)
        data = (data - self.mean[:, None, None]) / self.std[:, None, None]
        if self.cache_data:
            self.data_cache[real_idx] = data
        return data

    def __getitem__(self, idx):
        data = self._load(idx)
        C, H, W = data.shape
        cs = self.crop_size

        if H <= cs:
            y0 = 0
            crop_h = H
        else:
            y0 = random.randint(0, H - cs)
            crop_h = cs

        if W <= cs:
            x0 = 0
            crop_w = W
        else:
            x0 = random.randint(0, W - cs)
            crop_w = cs

        patch = data[:, y0:y0 + crop_h, x0:x0 + crop_w]
        return torch.from_numpy(patch.copy())


class ERA5FullResDataset(Dataset):
    def __init__(self, data_root, divisor=64):
        self.data_root = data_root
        self.divisor = divisor
        self.mean, self.std = get_mean_std()

        self.file_pairs = []
        for root, dirs, files in os.walk(data_root):
            for f in sorted(files):
                if f.endswith('_pressure.nc'):
                    ts = f.replace('_pressure.nc', '')
                    single_f = os.path.join(root, f'{ts}_single.nc')
                    pressure_f = os.path.join(root, f)
                    if os.path.exists(single_f):
                        self.file_pairs.append((pressure_f, single_f, ts))
        self.file_pairs.sort(key=lambda x: x[2])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pressure_f, single_f, ts = self.file_pairs[idx]
        data = read_nc(pressure_f, single_f)
        data = (data - self.mean[:, None, None]) / self.std[:, None, None]

        C, H, W = data.shape
        pad_h = (self.divisor - H % self.divisor) % self.divisor
        pad_w = (self.divisor - W % self.divisor) % self.divisor
        if pad_h > 0 or pad_w > 0:
            data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        return torch.from_numpy(data.copy()), H, W, ts


def build_model(arch, N=192, M=320):
    common = dict(in_channel=IN_CHANNELS)

    if arch == 'factorized':
        return FactorizedPrior(N=N, M=M, **common)
    elif arch == 'factorized-relu':
        return FactorizedPriorReLU(N=N, M=M, **common)
    elif arch == 'hyperprior':
        return ScaleHyperprior(N=N, M=M, **common)
    elif arch == 'mean-hyperprior':
        return MeanScaleHyperprior(N=N, M=M, **common)
    elif arch == 'joint':
        return JointAutoregressiveHierarchicalPriors(N=N, M=M, **common)
    elif arch == 'cheng2020-anchor':
        return Cheng2020Anchor(N=N, **common)
    elif arch == 'cheng2020-attn':
        return Cheng2020Attention(N=N, **common)
    elif arch == 'tcm2023':
        return TCM2023(N=N, M=M, in_channel=IN_CHANNELS)
    elif arch == 'elic2022':
        return ELIC2022(N=N, M=M, in_chans=IN_CHANNELS)
    elif arch == 'stf2022':
        return SymmetricalTransFormer2022(in_chans=IN_CHANNELS)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def compute_rd_loss(out_net, x, lmbda):
    mse = nn.functional.mse_loss(out_net['x_hat'], x)
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    bpp = 0.0
    for key, likelihoods in out_net['likelihoods'].items():
        bpp += -torch.log2(likelihoods).sum() / num_pixels
    loss = lmbda * mse + bpp
    return loss, mse.item(), bpp.item()


def train_one_epoch(model, dataloader, optimizer, optimizer_aux, device, epoch, lmbda, clip_max_norm=1.0, use_wandb=False):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_bpp = 0.0
    n = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss, mse_val, bpp_val = compute_rd_loss(out, x, lmbda)

        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if optimizer_aux is not None:
            optimizer_aux.zero_grad()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            optimizer_aux.step()

        total_loss += loss.item()
        total_mse += mse_val
        total_bpp += bpp_val
        n += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse_val:.6f}',
            'bpp': f'{bpp_val:.4f}',
        })

        if use_wandb:
            step = epoch * len(dataloader) + batch_idx
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_mse': mse_val,
                'train/batch_bpp': bpp_val,
            }, step=step)

    return total_loss / n, total_mse / n, total_bpp / n


@torch.no_grad()
def validate(model, dataloader, device, lmbda):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_bpp = 0.0
    n = 0

    for x in tqdm(dataloader, desc='Validating'):
        x = x.to(device)
        out = model(x)
        loss, mse_val, bpp_val = compute_rd_loss(out, x, lmbda)
        total_loss += loss.item()
        total_mse += mse_val
        total_bpp += bpp_val
        n += 1

    if n == 0:
        return 0, 0, 0
    return total_loss / n, total_mse / n, total_bpp / n


@torch.no_grad()
def validate_full_res(model, dataset, device, lmbda):
    model.eval()
    total_mse = 0.0
    total_bpp = 0.0
    n = 0

    for i in range(len(dataset)):
        data, H, W, ts = dataset[i]
        x = data.unsqueeze(0).to(device)
        out = model(x)
        x_hat = out['x_hat'][:, :, :H, :W]
        x_orig = x[:, :, :H, :W]

        mse_val = nn.functional.mse_loss(x_hat, x_orig).item()
        num_pixels = H * W
        bpp_val = 0.0
        for key, likelihoods in out['likelihoods'].items():
            bpp_val += -torch.log2(likelihoods).sum().item() / num_pixels

        total_mse += mse_val
        total_bpp += bpp_val
        n += 1
        logging.info(f"  [{ts}] MSE={mse_val:.6f} BPP={bpp_val:.4f}")

    if n == 0:
        return 0, 0
    return total_mse / n, total_bpp / n


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, required=True,
                        help='ERA5 data directory containing .nc files')
    parser.add_argument('--val_data_root', type=str, default=None,
                        help='Validation data directory (defaults to data_root)')

    parser.add_argument('--arch', type=str, default='mean-hyperprior',
                        choices=['factorized', 'factorized-relu', 'hyperprior',
                                 'mean-hyperprior', 'joint',
                                 'cheng2020-anchor', 'cheng2020-attn',
                                 'tcm2023', 'elic2022', 'stf2022'])
    parser.add_argument('--N', type=int, default=192)
    parser.add_argument('--M', type=int, default=320)
    parser.add_argument('--lmbda', type=float, default=0.01,
                        help='Rate-distortion tradeoff: L = lmbda * MSE + BPP')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_aux', type=float, default=1e-3)
    parser.add_argument('--clip_max_norm', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--output_dir', type=str,
                        default='/root/work/cra5/results/era5_compressai')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='cra5-compressai')
    parser.add_argument('--wandb_entity', type=str, default=None)

    parser.add_argument('--gpu', type=str, default='0')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / args.arch / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / f'train_{timestamp}.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Input channels: {IN_CHANNELS}")
    logging.info(f"Device: {device}")

    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.arch}_{timestamp}",
            config=vars(args),
            tags=[args.arch, 'era5'],
        )

    train_dataset = ERA5CompressDataset(
        data_root=args.data_root,
        crop_size=args.crop_size,
        cache_data=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_data_root = args.val_data_root or args.data_root
    val_dataset = ERA5CompressDataset(
        data_root=val_data_root,
        crop_size=args.crop_size,
        cache_data=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.arch, N=args.N, M=args.M).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {args.arch}, N={args.N}, M={args.M}")
    logging.info(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
    )
    aux_params = [p for n, p in model.named_parameters() if 'quantiles' in n]
    optimizer_aux = optim.Adam(aux_params, lr=args.lr_aux) if aux_params else None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 1
    best_loss = float('inf')

    if args.checkpoint is not None:
        logging.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        logging.info(f"Resuming from epoch {start_epoch}")
        model.update()

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    logging.info(f"Lambda={args.lmbda}, Loss = lmbda * MSE + BPP")
    logging.info("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mse, train_bpp = train_one_epoch(
            model, train_loader, optimizer, optimizer_aux,
            device, epoch, args.lmbda,
            clip_max_norm=args.clip_max_norm,
            use_wandb=args.use_wandb and HAS_WANDB,
        )

        val_loss, val_mse, val_bpp = validate(
            model, val_loader, device, args.lmbda,
        )

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch {epoch}/{args.epochs}:")
        logging.info(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, BPP: {train_bpp:.4f}")
        logging.info(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, BPP: {val_bpp:.4f}")
        logging.info(f"  LR: {new_lr:.2e}")

        if args.use_wandb and HAS_WANDB:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/mse': train_mse,
                'train/bpp': train_bpp,
                'val/loss': val_loss,
                'val/mse': val_mse,
                'val/bpp': val_bpp,
                'lr': new_lr,
            })

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        model.update()

        ckpt_data = {
            'epoch': epoch,
            'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'args': vars(args),
        }

        if is_best:
            torch.save(ckpt_data, output_dir / 'best_model.pth')
            logging.info(f"  -> Saved best model (loss={val_loss:.6f})")

        if epoch % args.save_interval == 0:
            torch.save(ckpt_data, output_dir / f'checkpoint_epoch_{epoch}.pth')
            logging.info(f"  -> Saved checkpoint epoch {epoch}")

        logging.info("-" * 80)

    logging.info("Training complete!")
    logging.info(f"Best validation loss: {best_loss:.6f}")
    logging.info(f"Checkpoints saved in: {output_dir}")

    if args.use_wandb and HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
