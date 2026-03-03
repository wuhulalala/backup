import os
import sys
import json
import time
import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from cra5.models.compressai.zoo.image import _load_model

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

TOTAL_LEVELS = [
    1000., 975., 950., 925., 900., 875., 850., 825., 800.,
    775., 750., 700., 650., 600., 550., 500., 450., 400.,
    350., 300., 250., 225., 200., 175., 150., 125., 100.,
    70., 50., 30., 20., 10., 7., 5., 3., 2., 1.,
]

IN_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])
GROUP_SIZE = 3

ARCH_ZOO_MAP = {
    'factorized': 'bmshj2018-factorized',
    'hyperprior': 'bmshj2018-hyperprior',
    'mean-hyperprior': 'mbt2018-mean',
    'joint': 'mbt2018',
    'cheng2020-anchor': 'cheng2020-anchor',
    'cheng2020-attn': 'cheng2020-attn',
}


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


def get_channel_names():
    names = []
    for vname in VNAMES['pressure']:
        for level in PRESSURE_LEVELS:
            names.append(f"{vname}_{int(level)}")
    for vname in VNAMES['single']:
        names.append(vname)
    return names


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


def find_nc_pairs(data_root):
    pairs = []
    for root, dirs, files in os.walk(data_root):
        for f in sorted(files):
            if f.endswith('_pressure.nc'):
                ts = f.replace('_pressure.nc', '')
                single_f = os.path.join(root, f'{ts}_single.nc')
                pressure_f = os.path.join(root, f)
                if os.path.exists(single_f):
                    pairs.append((pressure_f, single_f, ts))
    pairs.sort(key=lambda x: x[2])
    return pairs


def pad_to_divisor(data, divisor=64):
    C, H, W = data.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    return data, H, W


def compute_bitstream_size(strings):
    total = 0
    for s_list in strings:
        for s in s_list:
            total += len(s)
    return total


def calculate_psnr(original, reconstructed, data_range=None):
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse < 1e-10:
        return float('inf'), mse
    if data_range is None:
        data_range = float(original.max() - original.min())
        if data_range < 1e-6:
            data_range = 1.0
    return float(10 * np.log10(data_range**2 / mse)), mse


def denormalize(data_np, mean, std):
    return data_np * std[:, None, None] + mean[:, None, None]


def split_channel_groups(x, group_size=GROUP_SIZE):
    B, C, H, W = x.shape
    groups = []
    for i in range(0, C, group_size):
        chunk = x[:, i:i + group_size]
        actual_c = chunk.shape[1]
        if actual_c < group_size:
            pad = chunk[:, -1:].expand(B, group_size - actual_c, H, W)
            chunk = torch.cat([chunk, pad], dim=1)
        groups.append((chunk, actual_c))
    return groups


def minmax_normalize(chunk):
    B, C, H, W = chunk.shape
    cmin = chunk.reshape(B, C, -1).min(dim=-1, keepdim=True).values.unsqueeze(-1)
    cmax = chunk.reshape(B, C, -1).max(dim=-1, keepdim=True).values.unsqueeze(-1)
    scale = cmax - cmin
    scale = scale.clamp(min=1e-8)
    normed = (chunk - cmin) / scale
    return normed, cmin, scale


def minmax_denormalize(chunk, cmin, scale):
    return chunk * scale + cmin


@torch.no_grad()
def test_forward_grouped(model, x, orig_H, orig_W):
    groups = split_channel_groups(x)
    num_groups = len(groups)

    x_hat_parts = []
    total_bpp_bits = 0.0
    total_time = 0.0

    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        out = model(normed)
        torch.cuda.synchronize() if x.is_cuda else None
        t1 = time.time()
        total_time += t1 - t0

        x_hat_chunk = minmax_denormalize(out['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])

        for key, likelihoods in out['likelihoods'].items():
            lk = likelihoods[:, :, :orig_H, :orig_W] if likelihoods.shape[-2] >= orig_H else likelihoods
            total_bpp_bits += -torch.log2(lk.clamp(min=1e-10)).sum().item()

    x_hat = torch.cat(x_hat_parts, dim=1)
    num_pixels = orig_H * orig_W
    bpp = total_bpp_bits / num_pixels

    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'time': total_time,
        'num_groups': num_groups,
    }


@torch.no_grad()
def test_compress_decompress_grouped(model, x, orig_H, orig_W):
    groups = split_channel_groups(x)
    num_groups = len(groups)

    x_hat_parts = []
    total_bitstream_bytes = 0
    total_encode_time = 0.0
    total_decode_time = 0.0

    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        compressed = model.compress(normed)
        torch.cuda.synchronize() if x.is_cuda else None
        t1 = time.time()

        decompressed = model.decompress(compressed['strings'], compressed['shape'])
        torch.cuda.synchronize() if x.is_cuda else None
        t2 = time.time()

        total_encode_time += t1 - t0
        total_decode_time += t2 - t1
        total_bitstream_bytes += compute_bitstream_size(compressed['strings'])
        x_hat_chunk = minmax_denormalize(decompressed['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])

    x_hat = torch.cat(x_hat_parts, dim=1)
    num_pixels = orig_H * orig_W
    bpp = total_bitstream_bytes * 8.0 / num_pixels

    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'bitstream_bytes': total_bitstream_bytes,
        'encode_time': total_encode_time,
        'decode_time': total_decode_time,
        'num_groups': num_groups,
    }


def compute_per_variable_metrics(x_orig, x_hat, mean, std, channel_names):
    x_denorm = x_orig * std[:, None, None] + mean[:, None, None]
    x_hat_denorm = x_hat * std[:, None, None] + mean[:, None, None]

    metrics = {}
    C = x_orig.shape[0]
    for c in range(C):
        rmse = np.sqrt(np.mean((x_denorm[c] - x_hat_denorm[c]) ** 2))
        metrics[channel_names[c]] = rmse

    group_metrics = {}
    idx = 0
    for vname in VNAMES['pressure']:
        rmse_list = []
        for _ in PRESSURE_LEVELS:
            rmse_list.append(metrics[channel_names[idx]])
            idx += 1
        group_metrics[vname] = {
            'mean_rmse': np.mean(rmse_list),
            'max_rmse': np.max(rmse_list),
        }
    for vname in VNAMES['single']:
        group_metrics[vname] = {
            'rmse': metrics[channel_names[idx]],
        }
        idx += 1

    return metrics, group_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/work/cra5/data/ERA5/2024')
    parser.add_argument('--arch', type=str, default='mean-hyperprior',
                        choices=list(ARCH_ZOO_MAP.keys()))
    parser.add_argument('--quality', type=int, default=4)
    parser.add_argument('--metric', type=str, default='mse', choices=['mse', 'ms-ssim'])
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/root/work/cra5/results/test_compressai')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    zoo_name = ARCH_ZOO_MAP[args.arch]
    output_dir = Path(args.output_dir) / f"{args.arch}_q{args.quality}_{args.metric}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'test.log', mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Args: {vars(args)}")
    logging.info(f"Device: {device}")
    logging.info(f"Input channels: {IN_CHANNELS}, group_size: {GROUP_SIZE}, "
                 f"num_groups: {math.ceil(IN_CHANNELS / GROUP_SIZE)}")

    mean, std = get_mean_std()
    channel_names = get_channel_names()

    file_pairs = find_nc_pairs(args.data_root)
    if len(file_pairs) == 0:
        logging.error(f"No ERA5 .nc file pairs found in {args.data_root}")
        return
    logging.info(f"Found {len(file_pairs)} ERA5 timestamps")

    logging.info(f"Loading pretrained model: {zoo_name}, quality={args.quality}, metric={args.metric}")
    model = _load_model(zoo_name, args.metric, args.quality, pretrained=True, progress=True)

    model = model.to(device)
    model.eval()
    model.update()

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {zoo_name}, params={total_params:,}")

    all_results = []

    for pressure_f, single_f, ts in file_pairs:
        logging.info(f"Processing: {ts}")

        raw_data = read_nc(pressure_f, single_f)
        norm_data = (raw_data - mean[:, None, None]) / std[:, None, None]
        padded, orig_H, orig_W = pad_to_divisor(norm_data, divisor=64)
        x = torch.from_numpy(padded[None]).to(device)

        logging.info(f"  Shape: raw={raw_data.shape}, padded={padded.shape}")

        result = {'timestamp': ts}

        logging.info("  Forward pass (channel-grouped)...")
        fwd = test_forward_grouped(model, x, orig_H, orig_W)

        x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
        raw_crop = raw_data[:, :orig_H, :orig_W]
        x_hat_denorm = denormalize(x_hat_np, mean, std)
        psnr, mse = calculate_psnr(raw_crop, x_hat_denorm)
        result['forward'] = {
            'mse': mse,
            'psnr': psnr,
            'bpp': fwd['bpp'],
            'time': fwd['time'],
        }
        logging.info(f"  Forward: MSE={mse:.6f}, PSNR={psnr:.2f}dB, "
                     f"BPP={fwd['bpp']:.4f}, Time={fwd['time']:.2f}s")

        per_var, group_var = compute_per_variable_metrics(
            norm_data[:, :orig_H, :orig_W], x_hat_np, mean, std, channel_names
        )
        result['per_variable_rmse'] = {k: float(v) for k, v in per_var.items()}
        result['group_rmse'] = {}
        for k, v in group_var.items():
            result['group_rmse'][k] = {kk: float(vv) for kk, vv in v.items()}

        logging.info("  Per-variable group RMSE (denormalized):")
        for vname, vmetrics in group_var.items():
            if 'mean_rmse' in vmetrics:
                logging.info(f"    {vname}: mean={vmetrics['mean_rmse']:.4f}, max={vmetrics['max_rmse']:.4f}")
            else:
                logging.info(f"    {vname}: rmse={vmetrics['rmse']:.4f}")

        if not args.no_compress:
            try:
                logging.info("  Compress/decompress (channel-grouped)...")
                comp = test_compress_decompress_grouped(model, x, orig_H, orig_W)

                comp_hat_np = comp['x_hat'].squeeze(0).cpu().numpy()
                comp_hat_denorm = denormalize(comp_hat_np, mean, std)
                comp_psnr, comp_mse = calculate_psnr(raw_crop, comp_hat_denorm)

                original_bytes = raw_data.nbytes
                compressed_bytes = comp['bitstream_bytes']
                ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0

                result['compress'] = {
                    'mse': comp_mse,
                    'psnr': comp_psnr,
                    'bpp': comp['bpp'],
                    'bitstream_bytes': compressed_bytes,
                    'original_bytes': original_bytes,
                    'compression_ratio': ratio,
                    'encode_time': comp['encode_time'],
                    'decode_time': comp['decode_time'],
                }

                logging.info(f"  Compress: MSE={comp_mse:.6f}, PSNR={comp_psnr:.2f}dB, "
                             f"BPP={comp['bpp']:.4f}, Size={compressed_bytes:,}B, "
                             f"Ratio={ratio:.2f}:1, "
                             f"Enc={comp['encode_time']:.2f}s, Dec={comp['decode_time']:.2f}s")
            except Exception as e:
                logging.warning(f"  Compress/decompress failed: {e}")
                import traceback
                traceback.print_exc()
                result['compress'] = {'error': str(e)}

        all_results.append(result)
        logging.info("-" * 60)

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {results_file}")

    if len(all_results) > 0:
        avg_mse = np.mean([r['forward']['mse'] for r in all_results])
        avg_bpp = np.mean([r['forward']['bpp'] for r in all_results])
        avg_psnr = np.mean([r['forward']['psnr'] for r in all_results])
        logging.info(f"Average Forward: MSE={avg_mse:.6f}, PSNR={avg_psnr:.2f}dB, BPP={avg_bpp:.4f}")


if __name__ == '__main__':
    main()
