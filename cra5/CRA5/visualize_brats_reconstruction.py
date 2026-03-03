import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from cra5.api.brats_api import brats_api

brats_API = brats_api(
    data_root='/bigdata/datasets/aiocta/brats2023-part-1/',
    save_root='/bigdata/datasets/BraTS_compressed',
    device='cuda',
    model_version=621,
    pretrained=True,
)

patient_id = 'BraTS-GLI-00000-000'

original_data = brats_API.load_patient_volume(patient_id)

y_hat = brats_API.encode_to_latent(patient_id=patient_id, latent_type='quantized')
normalized_x_hat = brats_API.latent_to_reconstruction(y_hat=y_hat)
x_hat = brats_API.de_normalization_cra5(normalized_x_hat.squeeze(0), patient_id=patient_id)
reconstructed = x_hat.cpu().numpy()

def calculate_psnr(original, reconstructed, data_range=None):
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    if data_range is None:
        data_range = original.max() - original.min()
        if data_range < 1e-6:
            data_range = 1.0
    return 10 * np.log10(data_range**2 / mse)

mse = np.mean((original_data - reconstructed) ** 2)
data_range = original_data.max() - original_data.min()
if data_range < 1e-6:
    data_range = 1.0
psnr = calculate_psnr(original_data, reconstructed, data_range=data_range)

print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

slices_per_modality = 155
modality_ranges = [
    (0, 155, 't1n'),
    (155, 310, 't1c'),
    (310, 465, 't2w'),
    (465, 620, 't2f'),
]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path('/root/work/cra5/results/visualizations')
save_dir.mkdir(parents=True, exist_ok=True)

for start_idx, end_idx, modality_name in modality_ranges:
    mid_slice = slices_per_modality // 2
    channel_idx = start_idx + mid_slice
    
    original_slice = original_data[channel_idx, :, :]
    reconstructed_slice = reconstructed[channel_idx, :, :]
    difference = np.abs(original_slice - reconstructed_slice)
    
    slice_mse = np.mean((original_slice - reconstructed_slice) ** 2)
    slice_data_range = original_slice.max() - original_slice.min()
    if slice_data_range < 1e-6:
        slice_data_range = 1.0
    slice_psnr = calculate_psnr(original_slice, reconstructed_slice, data_range=slice_data_range)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    vmin = original_slice.min()
    vmax = original_slice.max()
    im0 = axes[0].imshow(original_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'{modality_name.upper()} - Original\nSlice {mid_slice}', fontsize=10)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(reconstructed_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'{modality_name.upper()} - Reconstructed\nMSE: {slice_mse:.4f}', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    diff_max = difference.max()
    im2 = axes[2].imshow(difference, cmap='hot', vmin=0, vmax=diff_max, origin='lower')
    axes[2].set_title(f'{modality_name.upper()} - Difference\nMax diff: {diff_max:.4f}', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    axes[3].hist(original_slice.flatten(), bins=50, alpha=0.5, label='Original', density=True, color='blue')
    axes[3].hist(reconstructed_slice.flatten(), bins=50, alpha=0.5, label='Reconstructed', density=True, color='red')
    axes[3].set_title(f'{modality_name.upper()} - Intensity Distribution\nPSNR: {slice_psnr:.2f} dB', fontsize=10)
    axes[3].set_xlabel('Intensity')
    axes[3].set_ylabel('Density')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'{patient_id}_{modality_name}_slice{mid_slice}_comparison_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

modality_name = 't1n'
start_idx = 0
num_slices_to_show = 3
fig, axes = plt.subplots(num_slices_to_show, 4, figsize=(20, 5 * num_slices_to_show))

for slice_idx in range(num_slices_to_show):
    channel_idx = start_idx + slice_idx
    
    original_slice = original_data[channel_idx, :, :]
    reconstructed_slice = reconstructed[channel_idx, :, :]
    difference = np.abs(original_slice - reconstructed_slice)
    
    slice_mse = np.mean((original_slice - reconstructed_slice) ** 2)
    slice_data_range = original_slice.max() - original_slice.min()
    if slice_data_range < 1e-6:
        slice_data_range = 1.0
    slice_psnr = calculate_psnr(original_slice, reconstructed_slice, data_range=slice_data_range)
    
    vmin = original_slice.min()
    vmax = original_slice.max()
    diff_max = difference.max()
    
    im0 = axes[slice_idx, 0].imshow(original_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[slice_idx, 0].set_title(f'Original - Slice {slice_idx}', fontsize=9)
    axes[slice_idx, 0].axis('off')
    
    im1 = axes[slice_idx, 1].imshow(reconstructed_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[slice_idx, 1].set_title(f'Reconstructed - Slice {slice_idx}\nMSE: {slice_mse:.4f}', fontsize=9)
    axes[slice_idx, 1].axis('off')
    
    im2 = axes[slice_idx, 2].imshow(difference, cmap='hot', vmin=0, vmax=diff_max, origin='lower')
    axes[slice_idx, 2].set_title(f'Difference - Slice {slice_idx}', fontsize=9)
    axes[slice_idx, 2].axis('off')
    
    axes[slice_idx, 3].hist(original_slice.flatten(), bins=20, alpha=0.5, label='Original', color='blue')
    axes[slice_idx, 3].hist(reconstructed_slice.flatten(), bins=20, alpha=0.5, label='Reconstructed', color='red')
    axes[slice_idx, 3].set_title(f'PSNR: {slice_psnr:.2f} dB', fontsize=9)
    axes[slice_idx, 3].legend(fontsize=7)
    axes[slice_idx, 3].grid(True, alpha=0.3)
    
    if slice_idx == 0:
        plt.colorbar(im0, ax=axes[slice_idx, 0], fraction=0.046)
        plt.colorbar(im1, ax=axes[slice_idx, 1], fraction=0.046)
        plt.colorbar(im2, ax=axes[slice_idx, 2], fraction=0.046)

plt.suptitle(f'{modality_name.upper()} Modality - First {num_slices_to_show} Slices', fontsize=14, y=0.995)
plt.tight_layout()
save_path = save_dir / f'{patient_id}_{modality_name}_multiple_slices_comparison_{timestamp}.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
