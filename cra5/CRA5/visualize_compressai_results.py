import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ARCH_DISPLAY = {
    'bmshj2018-factorized': 'Factorized',
    'bmshj2018-hyperprior': 'Hyperprior',
    'mbt2018-mean': 'Mean-Hyperprior',
    'mbt2018': 'Joint-AR',
    'cheng2020-anchor': 'Cheng2020-Anchor',
    'cheng2020-attn': 'Cheng2020-Attn',
}

ARCH_COLORS = {
    'bmshj2018-factorized': '#1f77b4',
    'bmshj2018-hyperprior': '#ff7f0e',
    'mbt2018-mean': '#2ca02c',
    'mbt2018': '#d62728',
    'cheng2020-anchor': '#9467bd',
    'cheng2020-attn': '#8c564b',
}

ARCH_MARKERS = {
    'bmshj2018-factorized': 'o',
    'bmshj2018-hyperprior': 's',
    'mbt2018-mean': '^',
    'mbt2018': 'D',
    'cheng2020-anchor': 'v',
    'cheng2020-attn': 'P',
}

CHANNEL_DISPLAY = {
    'z_500': 'Geopotential (500 hPa)',
    't_850': 'Temperature (850 hPa)',
    't2m': '2m Temperature',
    'u10': '10m U-Wind',
    'sp': 'Surface Pressure',
}

CHANNEL_CMAP = {
    'z_500': 'viridis',
    't_850': 'RdYlBu_r',
    't2m': 'RdYlBu_r',
    'u10': 'RdBu',
    'sp': 'viridis',
}

CHANNEL_UNITS = {
    'z_500': 'm²/s²',
    't_850': 'K',
    't2m': 'K',
    'u10': 'm/s',
    'sp': 'Pa',
}


def load_summary(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [r for r in data if 'error' not in r]


def group_by_arch_metric(results):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r['metric']][r['arch']].append(r)
    for metric in grouped:
        for arch in grouped[metric]:
            grouped[metric][arch].sort(key=lambda x: x['quality'])
    return grouped


def plot_rate_distortion(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for arch in sorted(grouped[metric_name].keys()):
        results = grouped[metric_name][arch]
        bpps = [r['forward']['bpp'] for r in results]
        psnrs = [r['forward']['psnr'] for r in results]
        qualities = [r['quality'] for r in results]
        display_name = ARCH_DISPLAY.get(arch, arch)
        color = ARCH_COLORS.get(arch, 'gray')
        marker = ARCH_MARKERS.get(arch, 'o')
        ax.plot(bpps, psnrs, '-', color=color, marker=marker, markersize=8,
                label=display_name, linewidth=2)
        for bpp, psnr, q in zip(bpps, psnrs, qualities):
            ax.annotate(f'q{q}', (bpp, psnr), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)
    ax.set_xlabel('BPP (bits per pixel)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Rate-Distortion on ERA5 ({metric_name.upper()} optimized)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_psnr_heatmap(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    archs = sorted(grouped[metric_name].keys())
    all_qualities = set()
    for arch in archs:
        for r in grouped[metric_name][arch]:
            all_qualities.add(r['quality'])
    qualities = sorted(all_qualities)
    data = np.full((len(archs), len(qualities)), np.nan)
    for i, arch in enumerate(archs):
        for r in grouped[metric_name][arch]:
            j = qualities.index(r['quality'])
            data[i, j] = r['forward']['psnr']
    fig, ax = plt.subplots(figsize=(max(10, len(qualities) * 1.5), max(4, len(archs) * 0.8)))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(len(qualities)))
    ax.set_xticklabels([f'q{q}' for q in qualities])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([ARCH_DISPLAY.get(a, a) for a in archs])
    median_val = np.nanmedian(data)
    for i in range(len(archs)):
        for j in range(len(qualities)):
            if not np.isnan(data[i, j]):
                color = 'black' if data[i, j] > median_val else 'white'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, label='PSNR (dB)')
    ax.set_title(f'PSNR Comparison ({metric_name.upper()} optimized)', fontsize=14)
    ax.set_xlabel('Quality Level')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_bpp_heatmap(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    archs = sorted(grouped[metric_name].keys())
    all_qualities = set()
    for arch in archs:
        for r in grouped[metric_name][arch]:
            all_qualities.add(r['quality'])
    qualities = sorted(all_qualities)
    data = np.full((len(archs), len(qualities)), np.nan)
    for i, arch in enumerate(archs):
        for r in grouped[metric_name][arch]:
            j = qualities.index(r['quality'])
            data[i, j] = r['forward']['bpp']
    fig, ax = plt.subplots(figsize=(max(10, len(qualities) * 1.5), max(4, len(archs) * 0.8)))
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r')
    ax.set_xticks(range(len(qualities)))
    ax.set_xticklabels([f'q{q}' for q in qualities])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([ARCH_DISPLAY.get(a, a) for a in archs])
    for i in range(len(archs)):
        for j in range(len(qualities)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold')
    plt.colorbar(im, label='BPP')
    ax.set_title(f'BPP Comparison ({metric_name.upper()} optimized)', fontsize=14)
    ax.set_xlabel('Quality Level')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_per_variable_rmse(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    selected = {}
    for arch, results in grouped[metric_name].items():
        best = max(results, key=lambda x: x['quality'])
        selected[arch] = best
    pressure_vars = ['z', 'q', 'u', 'v', 't', 'r', 'w']
    single_vars = ['v10', 'u10', 'v100', 'u100', 't2m', 'tcc', 'sp', 'tp', 'msl']
    all_vars = pressure_vars + single_vars
    n_vars = len(all_vars)
    n_models = len(selected)
    if n_models == 0:
        return
    bar_width = 0.8 / n_models
    x = np.arange(n_vars)
    fig, ax = plt.subplots(figsize=(16, 7))
    for k, (arch, result) in enumerate(sorted(selected.items())):
        rmse_vals = []
        for vname in pressure_vars:
            rmse_vals.append(result['group_rmse'][vname]['mean_rmse'])
        for vname in single_vars:
            rmse_vals.append(result['group_rmse'][vname]['rmse'])
        display_name = f"{ARCH_DISPLAY.get(arch, arch)} q{result['quality']}"
        color = ARCH_COLORS.get(arch, 'gray')
        ax.bar(x + k * bar_width, rmse_vals, bar_width, label=display_name, color=color, alpha=0.8)
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(all_vars, rotation=45, ha='right')
    ax.set_ylabel('RMSE (denormalized)')
    ax.set_title(f'Per-Variable RMSE - Best Quality ({metric_name.upper()} optimized)', fontsize=14)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_combined_rd(grouped, output_path):
    metrics_available = list(grouped.keys())
    if not metrics_available:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for metric_name in sorted(metrics_available):
        linestyle = '-' if metric_name == 'mse' else '--'
        for arch in sorted(grouped[metric_name].keys()):
            results = grouped[metric_name][arch]
            bpps = [r['forward']['bpp'] for r in results]
            psnrs = [r['forward']['psnr'] for r in results]
            display_name = f"{ARCH_DISPLAY.get(arch, arch)} ({metric_name})"
            color = ARCH_COLORS.get(arch, 'gray')
            marker = ARCH_MARKERS.get(arch, 'o')
            ax.plot(bpps, psnrs, linestyle, color=color, marker=marker, markersize=6,
                    label=display_name, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('BPP (bits per pixel)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Rate-Distortion on ERA5 (All Models)', fontsize=14)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_channel_reconstruction(samples_dir, grouped, metric_name, channel_name, timestamp, output_path):
    if metric_name not in grouped:
        return
    orig_file = samples_dir / f"original_{timestamp}.npz"
    if not orig_file.exists():
        return
    orig_data = np.load(orig_file)
    if channel_name not in orig_data:
        return
    original = orig_data[channel_name]

    selected = {}
    for arch, results in grouped[metric_name].items():
        best = max(results, key=lambda x: x['quality'])
        selected[arch] = best

    available_models = []
    for arch in sorted(selected.keys()):
        result = selected[arch]
        recon_file = samples_dir / f"{result['model_id']}_{timestamp}.npz"
        if recon_file.exists():
            recon_data = np.load(recon_file)
            if channel_name in recon_data:
                available_models.append((arch, result, recon_data[channel_name]))

    if not available_models:
        return

    n_models = len(available_models)
    ncols = n_models + 1
    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10))

    cmap = CHANNEL_CMAP.get(channel_name, 'viridis')
    display = CHANNEL_DISPLAY.get(channel_name, channel_name)
    unit = CHANNEL_UNITS.get(channel_name, '')

    vmin, vmax = original.min(), original.max()

    im0 = axes[0, 0].imshow(original, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'Original\n{display}', fontsize=9)
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label=unit)

    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, f'Original\n{display}\n{unit}',
                    ha='center', va='center', fontsize=10, transform=axes[1, 0].transAxes)

    axes[2, 0].hist(original.flatten(), bins=50, color='blue', alpha=0.7, density=True)
    axes[2, 0].set_title('Original Dist.', fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)

    for col, (arch, result, recon) in enumerate(available_models, 1):
        diff = np.abs(original - recon)
        ch_psnr_val, ch_mse = calculate_psnr_simple(original, recon)
        display_name = ARCH_DISPLAY.get(arch, arch)

        im1 = axes[0, col].imshow(recon, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        axes[0, col].set_title(f'{display_name} q{result["quality"]}\nPSNR={ch_psnr_val:.1f}dB', fontsize=9)
        axes[0, col].axis('off')

        diff_max = diff.max()
        im2 = axes[1, col].imshow(diff, cmap='hot', vmin=0, origin='lower')
        axes[1, col].set_title(f'|Diff| max={diff_max:.2f}', fontsize=8)
        axes[1, col].axis('off')
        plt.colorbar(im2, ax=axes[1, col], fraction=0.046)

        axes[2, col].hist(original.flatten(), bins=50, alpha=0.5, label='Orig', density=True, color='blue')
        axes[2, col].hist(recon.flatten(), bins=50, alpha=0.5, label='Recon', density=True, color='red')
        axes[2, col].legend(fontsize=6)
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].set_title(f'MSE={ch_mse:.2f}', fontsize=8)

    fig.suptitle(f'{display} - Model Comparison ({metric_name.upper()} optimized)', fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def calculate_psnr_simple(original, reconstructed):
    orig64 = original.astype(np.float64)
    recon64 = reconstructed.astype(np.float64)
    mse = float(np.mean((orig64 - recon64) ** 2))
    if mse < 1e-10:
        return float('inf'), mse
    data_range = float(orig64.max() - orig64.min())
    if data_range < 1e-6:
        data_range = 1.0
    psnr = float(10 * np.log10(data_range ** 2 / mse))
    return psnr, mse


def plot_cr_heatmap(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    archs = sorted(grouped[metric_name].keys())
    all_qualities = set()
    for arch in archs:
        for r in grouped[metric_name][arch]:
            all_qualities.add(r['quality'])
    qualities = sorted(all_qualities)
    data = np.full((len(archs), len(qualities)), np.nan)
    for i, arch in enumerate(archs):
        for r in grouped[metric_name][arch]:
            j = qualities.index(r['quality'])
            data[i, j] = r['forward'].get('compression_ratio', 0)
    fig, ax = plt.subplots(figsize=(max(10, len(qualities) * 1.5), max(4, len(archs) * 0.8)))
    im = ax.imshow(data, aspect='auto', cmap='YlGn')
    ax.set_xticks(range(len(qualities)))
    ax.set_xticklabels([f'q{q}' for q in qualities])
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([ARCH_DISPLAY.get(a, a) for a in archs])
    for i in range(len(archs)):
        for j in range(len(qualities)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i, j]:.1f}x', ha='center', va='center',
                        fontsize=9, fontweight='bold')
    plt.colorbar(im, label='Compression Ratio')
    ax.set_title(f'Compression Ratio ({metric_name.upper()} optimized)', fontsize=14)
    ax.set_xlabel('Quality Level')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_cr_vs_psnr(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for arch in sorted(grouped[metric_name].keys()):
        results = grouped[metric_name][arch]
        crs = [r['forward'].get('compression_ratio', 0) for r in results]
        psnrs = [r['forward']['psnr'] for r in results]
        qualities = [r['quality'] for r in results]
        display_name = ARCH_DISPLAY.get(arch, arch)
        color = ARCH_COLORS.get(arch, 'gray')
        marker = ARCH_MARKERS.get(arch, 'o')
        ax.plot(crs, psnrs, '-', color=color, marker=marker, markersize=8,
                label=display_name, linewidth=2)
        for cr, psnr, q in zip(crs, psnrs, qualities):
            ax.annotate(f'q{q}', (cr, psnr), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color=color)
    ax.set_xlabel('Compression Ratio (x)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'Compression Ratio vs PSNR on ERA5 ({metric_name.upper()} optimized)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_summary_csv(results, output_path):
    lines = ['model_id,arch,metric,quality,params,psnr,mse,bpp,compression_ratio,time']
    for r in sorted(results, key=lambda x: (x['arch'], x['metric'], x['quality'])):
        cr = r['forward'].get('compression_ratio', 0)
        lines.append(
            f"{r['model_id']},{r['arch']},{r['metric']},{r['quality']},"
            f"{r.get('params', 0)},{r['forward']['psnr']:.4f},"
            f"{r['forward']['mse']:.6f},{r['forward']['bpp']:.4f},"
            f"{cr:.2f},{r['forward']['time']:.2f}"
        )
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {output_path.name}")


def plot_quality_scaling(grouped, metric_name, output_path):
    if metric_name not in grouped:
        return
    archs = sorted(grouped[metric_name].keys())
    n_archs = len(archs)
    if n_archs == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for arch in archs:
        results = grouped[metric_name][arch]
        qualities = [r['quality'] for r in results]
        psnrs = [r['forward']['psnr'] for r in results]
        bpps = [r['forward']['bpp'] for r in results]
        display_name = ARCH_DISPLAY.get(arch, arch)
        color = ARCH_COLORS.get(arch, 'gray')
        marker = ARCH_MARKERS.get(arch, 'o')
        ax1.plot(qualities, psnrs, '-', color=color, marker=marker, markersize=7,
                 label=display_name, linewidth=2)
        ax2.plot(qualities, bpps, '-', color=color, marker=marker, markersize=7,
                 label=display_name, linewidth=2)
    ax1.set_xlabel('Quality Level', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title(f'PSNR vs Quality ({metric_name.upper()})', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Quality Level', fontsize=12)
    ax2.set_ylabel('BPP', fontsize=12)
    ax2.set_title(f'BPP vs Quality ({metric_name.upper()})', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='/root/work/cra5/results/all_compressai')
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    summary_file = results_dir / 'summary.json'
    samples_dir = results_dir / 'samples'
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = load_summary(summary_file)
    if not results:
        print("No valid results found")
        return

    print(f"Loaded {len(results)} results")
    grouped = group_by_arch_metric(results)
    metrics = list(grouped.keys())
    print(f"Metrics: {metrics}")
    for m in metrics:
        print(f"  {m}: {list(grouped[m].keys())}")

    timestamp = results[0]['timestamp']

    generate_summary_csv(results, plots_dir / 'summary.csv')

    plot_combined_rd(grouped, plots_dir / 'rate_distortion_all.png')

    for metric_name in metrics:
        print(f"\nGenerating plots for {metric_name}:")
        plot_rate_distortion(grouped, metric_name, plots_dir / f'rate_distortion_{metric_name}.png')
        plot_psnr_heatmap(grouped, metric_name, plots_dir / f'psnr_heatmap_{metric_name}.png')
        plot_bpp_heatmap(grouped, metric_name, plots_dir / f'bpp_heatmap_{metric_name}.png')
        plot_cr_heatmap(grouped, metric_name, plots_dir / f'cr_heatmap_{metric_name}.png')
        plot_cr_vs_psnr(grouped, metric_name, plots_dir / f'cr_vs_psnr_{metric_name}.png')
        plot_per_variable_rmse(grouped, metric_name, plots_dir / f'per_variable_rmse_{metric_name}.png')
        plot_quality_scaling(grouped, metric_name, plots_dir / f'quality_scaling_{metric_name}.png')

        for channel_name in ['z_500', 't_850', 't2m', 'u10', 'sp']:
            plot_channel_reconstruction(
                samples_dir, grouped, metric_name, channel_name, timestamp,
                plots_dir / f'reconstruction_{channel_name}_{metric_name}.png'
            )

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == '__main__':
    main()
