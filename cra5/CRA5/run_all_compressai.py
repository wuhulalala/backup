import os
import sys
import json
import time
import argparse
import gc
import math
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from cra5.models.compressai.zoo.image import _load_model, model_urls

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

TOTAL_LEVELS = list(PRESSURE_LEVELS)

IN_CHANNELS = len(VNAMES['pressure']) * len(PRESSURE_LEVELS) + len(VNAMES['single'])
GROUP_SIZE = 3


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


def get_sample_channel_indices():
    p_count = len(PRESSURE_LEVELS)
    base_single = len(VNAMES['pressure']) * p_count
    return {
        'z_500': VNAMES['pressure'].index('z') * p_count + PRESSURE_LEVELS.index(500.),
        't_850': VNAMES['pressure'].index('t') * p_count + PRESSURE_LEVELS.index(850.),
        't2m': base_single + VNAMES['single'].index('t2m'),
        'u10': base_single + VNAMES['single'].index('u10'),
        'sp': base_single + VNAMES['single'].index('sp'),
    }


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
    return np.concatenate(one_step, 0).astype(np.float32)


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
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-10:
        return float('inf'), mse
    if data_range is None:
        data_range = float(orig64.max() - orig64.min())
        if data_range < 1e-6:
            data_range = 1.0
    return float(10 * np.log10(data_range ** 2 / mse)), mse


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
    x_hat_parts = []
    total_bpp_bits = 0.0
    total_time = 0.0
    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        out = model(normed)
        if x.is_cuda:
            torch.cuda.synchronize()
        total_time += time.time() - t0
        x_hat_chunk = minmax_denormalize(out['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])
        for key, likelihoods in out['likelihoods'].items():
            lk = likelihoods[:, :, :orig_H, :orig_W] if likelihoods.shape[-2] >= orig_H else likelihoods
            total_bpp_bits += -torch.log2(lk.clamp(min=1e-10)).sum().item()
    x_hat = torch.cat(x_hat_parts, dim=1)
    bpp = total_bpp_bits / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'compression_ratio': (IN_CHANNELS * 32.0) / bpp if bpp > 0 else float('inf'),
        'time': total_time,
    }


@torch.no_grad()
def test_compress_decompress_grouped(model, x, orig_H, orig_W):
    groups = split_channel_groups(x)
    x_hat_parts = []
    total_bitstream_bytes = 0
    total_encode_time = 0.0
    total_decode_time = 0.0
    for chunk, actual_c in groups:
        normed, cmin, scale = minmax_normalize(chunk)
        t0 = time.time()
        compressed = model.compress(normed)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        decompressed = model.decompress(compressed['strings'], compressed['shape'])
        if x.is_cuda:
            torch.cuda.synchronize()
        t2 = time.time()
        total_encode_time += t1 - t0
        total_decode_time += t2 - t1
        total_bitstream_bytes += compute_bitstream_size(compressed['strings'])
        x_hat_chunk = minmax_denormalize(decompressed['x_hat'], cmin, scale)
        x_hat_parts.append(x_hat_chunk[:, :actual_c, :orig_H, :orig_W])
    x_hat = torch.cat(x_hat_parts, dim=1)
    original_bytes = IN_CHANNELS * orig_H * orig_W * 4
    bpp = total_bitstream_bytes * 8.0 / (orig_H * orig_W)
    return {
        'x_hat': x_hat,
        'bpp': bpp,
        'bitstream_bytes': total_bitstream_bytes,
        'original_bytes': original_bytes,
        'compression_ratio': original_bytes / total_bitstream_bytes if total_bitstream_bytes > 0 else float('inf'),
        'encode_time': total_encode_time,
        'decode_time': total_decode_time,
    }


def compute_group_rmse(x_orig_norm, x_hat_norm, mean, std):
    x_denorm = x_orig_norm * std[:, None, None] + mean[:, None, None]
    x_hat_denorm = x_hat_norm * std[:, None, None] + mean[:, None, None]
    group_metrics = {}
    idx = 0
    for vname in VNAMES['pressure']:
        rmse_list = []
        for _ in PRESSURE_LEVELS:
            rmse_list.append(float(np.sqrt(np.mean((x_denorm[idx] - x_hat_denorm[idx]) ** 2))))
            idx += 1
        group_metrics[vname] = {
            'mean_rmse': float(np.mean(rmse_list)),
            'max_rmse': float(np.max(rmse_list)),
        }
    for vname in VNAMES['single']:
        group_metrics[vname] = {
            'rmse': float(np.sqrt(np.mean((x_denorm[idx] - x_hat_denorm[idx]) ** 2)))
        }
        idx += 1
    return group_metrics


def get_all_configs(metrics_filter=None, archs_filter=None):
    configs = []
    for arch, metric_dict in model_urls.items():
        if arch == 'vaeformer-pretrained':
            continue
        if archs_filter and arch not in archs_filter:
            continue
        for metric, quality_dict in metric_dict.items():
            if metrics_filter and metric not in metrics_filter:
                continue
            for quality, url in quality_dict.items():
                if isinstance(url, str) and url.startswith('http'):
                    configs.append({'arch': arch, 'metric': metric, 'quality': quality})
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/work/cra5/data/ERA5/2024')
    parser.add_argument('--output_dir', type=str, default='/root/work/cra5/results/all_compressai')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--max_samples', type=int, default=1)
    parser.add_argument('--metrics', nargs='+', default=None)
    parser.add_argument('--archs', nargs='+', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    mean, std = get_mean_std()
    channel_names = get_channel_names()
    sample_indices = get_sample_channel_indices()

    file_pairs = find_nc_pairs(args.data_root)
    if not file_pairs:
        print(f"No ERA5 file pairs found in {args.data_root}")
        return

    if args.max_samples > 0:
        file_pairs = file_pairs[:args.max_samples]
    print(f"Using {len(file_pairs)} ERA5 timestamps")

    all_data = []
    for pressure_f, single_f, ts in file_pairs:
        raw = read_nc(pressure_f, single_f)
        norm = (raw - mean[:, None, None]) / std[:, None, None]
        padded, orig_H, orig_W = pad_to_divisor(norm, divisor=64)
        all_data.append({
            'timestamp': ts,
            'raw': raw,
            'norm': norm,
            'padded': padded,
            'orig_H': orig_H,
            'orig_W': orig_W,
        })

    for data in all_data:
        raw_crop = data['raw'][:, :data['orig_H'], :data['orig_W']]
        orig_samples = {name: raw_crop[idx] for name, idx in sample_indices.items()}
        np.savez(samples_dir / f"original_{data['timestamp']}.npz", **orig_samples)

    configs = get_all_configs(metrics_filter=args.metrics, archs_filter=args.archs)
    print(f"Total models to test: {len(configs)}")

    summary_file = output_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        tested = set(r['model_id'] for r in summary if 'error' not in r)
    else:
        summary = []
        tested = set()

    start_time = time.time()
    completed = 0

    for i, config in enumerate(configs):
        model_id = f"{config['arch']}_{config['metric']}_q{config['quality']}"
        if model_id in tested:
            print(f"[{i+1}/{len(configs)}] Skip: {model_id}")
            completed += 1
            continue

        print(f"[{i+1}/{len(configs)}] Testing: {model_id}")

        model = None
        try:
            model = _load_model(config['arch'], config['metric'], config['quality'],
                                pretrained=True, progress=True)
            model = model.to(device)
            model.eval()
            model.update()
            total_params = sum(p.numel() for p in model.parameters())

            for data in all_data:
                x = torch.from_numpy(data['padded'][None]).to(device)
                orig_H, orig_W = data['orig_H'], data['orig_W']
                raw_crop = data['raw'][:, :orig_H, :orig_W]

                fwd = test_forward_grouped(model, x, orig_H, orig_W)
                x_hat_np = fwd['x_hat'].squeeze(0).cpu().numpy()
                x_hat_denorm = denormalize(x_hat_np, mean, std)
                psnr, mse = calculate_psnr(raw_crop, x_hat_denorm)
                group_rmse = compute_group_rmse(
                    data['norm'][:, :orig_H, :orig_W], x_hat_np, mean, std
                )

                result = {
                    'model_id': model_id,
                    'arch': config['arch'],
                    'metric': config['metric'],
                    'quality': config['quality'],
                    'params': total_params,
                    'timestamp': data['timestamp'],
                    'forward': {
                        'mse': mse,
                        'psnr': psnr,
                        'bpp': fwd['bpp'],
                        'compression_ratio': fwd['compression_ratio'],
                        'time': fwd['time'],
                    },
                    'group_rmse': group_rmse,
                }

                recon_samples = {name: x_hat_denorm[idx] for name, idx in sample_indices.items()}
                np.savez(samples_dir / f"{model_id}_{data['timestamp']}.npz", **recon_samples)

                if args.compress:
                    try:
                        comp = test_compress_decompress_grouped(model, x, orig_H, orig_W)
                        comp_hat_np = comp['x_hat'].squeeze(0).cpu().numpy()
                        comp_hat_denorm = denormalize(comp_hat_np, mean, std)
                        comp_psnr, comp_mse = calculate_psnr(raw_crop, comp_hat_denorm)
                        result['compress'] = {
                            'mse': comp_mse,
                            'psnr': comp_psnr,
                            'bpp': comp['bpp'],
                            'bitstream_bytes': comp['bitstream_bytes'],
                            'original_bytes': comp['original_bytes'],
                            'compression_ratio': comp['compression_ratio'],
                            'encode_time': comp['encode_time'],
                            'decode_time': comp['decode_time'],
                        }
                    except Exception as e:
                        result['compress'] = {'error': str(e)}

                summary.append(result)
                print(f"  {data['timestamp']}: PSNR={psnr:.2f}dB, BPP={fwd['bpp']:.4f}, CR={fwd['compression_ratio']:.2f}x, Time={fwd['time']:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            summary.append({
                'model_id': model_id,
                'arch': config['arch'],
                'metric': config['metric'],
                'quality': config['quality'],
                'error': str(e),
            })

        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        completed += 1
        elapsed = time.time() - start_time
        avg_per = elapsed / completed
        remaining = avg_per * (len(configs) - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min")

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone. Total time: {(time.time()-start_time)/60:.1f}min")
    print(f"Results: {summary_file}")
    print(f"Samples: {samples_dir}")


if __name__ == '__main__':
    main()
