import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cra5.api.brats_api import brats_api
import numpy as np
import time

brats_API = brats_api(
    data_root='/bigdata/datasets/aiocta/brats2023-part-1/',
    save_root='/bigdata/datasets/BraTS_compressed',
    device='cuda',
    model_version=621,
    pretrained=True
)

#checkpoint_path = "/bigdata/cra5/checkpoints/brats_finetune/stage1/lr_recon_1e-06/original.pth"
#torch.save({
    #'state_dict': model.state_dict(),
#}, checkpoint_path)
data_root = '/bigdata/datasets/aiocta/brats2023-part-1/'
all_patients = [
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d)) and d.startswith("BraTS")
]
all_patients = sorted(all_patients)
patient_list = all_patients[:10]
num_iterations_per_patient = 10

all_encoding_time = []
all_decoding_time = []
all_mse_list = []
all_psnr_list = []
all_compressed_sizes = []

for patient_id in patient_list:
    try:
        original_data = brats_API.load_patient_volume(patient_id)
        
        for i in range(num_iterations_per_patient):
            encoding_start = time.time()
            y = brats_API.encode_to_latent(patient_id=patient_id, latent_type='float')
            bin_file_path = f"/bigdata/datasets/BraTS_compressed/{patient_id}_cra5_format.bin"
            brats_API.latent_to_bin(y=y, save_path=bin_file_path)
            encoding_end = time.time()
            all_encoding_time.append(encoding_end - encoding_start)
            
            decoding_start = time.time()
            y_hat = brats_API.bin_to_latent(bin_path=bin_file_path)
            normalized_x_hat = brats_API.latent_to_reconstruction(y_hat=y_hat)
            x_hat = brats_API.de_normalization_cra5(normalized_x_hat.squeeze(0), patient_id=patient_id)
            decoding_end = time.time()
            all_decoding_time.append(decoding_end - decoding_start)
            
            reconstructed = x_hat.cpu().numpy()
            mse = np.mean((original_data - reconstructed) ** 2)
            
            data_max = original_data.max()
            data_min = original_data.min()
            data_range = data_max - data_min
            if data_range < 1e-6:
                data_range = 1.0
            
            if mse > 1e-10:
                psnr = 10 * np.log10(data_range**2 / mse)
            else:
                psnr = float('inf')
            
            all_mse_list.append(mse)
            all_psnr_list.append(psnr)
        
        compressed_size_bytes = 0
        for i in range(4):
            part_path = bin_file_path.replace('.bin', f'_part{i}.bin')
            if os.path.exists(part_path):
                compressed_size_bytes += os.path.getsize(part_path)
        if compressed_size_bytes > 0:
            all_compressed_sizes.append(compressed_size_bytes)
            
    except Exception as e:
        print(f'Error: {e}')
        continue

encoding_array = np.array(all_encoding_time)
decoding_array = np.array(all_decoding_time)
mse_array = np.array(all_mse_list)
psnr_array = np.array(all_psnr_list)
psnr_array = psnr_array[~np.isinf(psnr_array)]

print(f"Encoding time: {encoding_array.mean():.4f}s")
print(f"Decoding time: {decoding_array.mean():.4f}s")
print(f"MSE: {mse_array.mean():.2f}")
print(f"PSNR: {psnr_array.mean():.2f} dB")

if len(all_compressed_sizes) > 0:
    avg_compressed_size = np.mean(all_compressed_sizes)
    num_pixels = 620 * 240 * 240
    original_size = num_pixels * 4
    bpp = (avg_compressed_size * 8) / num_pixels
    compression_ratio = original_size / avg_compressed_size
    print(f"BPP: {bpp:.6f}")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
