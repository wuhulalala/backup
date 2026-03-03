import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cra5.api import cra5_api
import numpy as np

cra5_API = cra5_api()
time_stamp = "2024-06-01T00:00:00"

original_data = cra5_API.read_data_from_nc(time_stamp=time_stamp)
channels_to_vname, vname_to_channels = cra5_API.channel_vname_mapping()
num_channels = original_data.shape[0]

def get_data_range(data, default=1.0):
    data_range = data.max() - data.min()
    return data_range if data_range >= 1e-6 else default

def calculate_psnr(original, reconstructed, data_range=None):
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    if data_range is None:
        data_range = get_data_range(original)
    return 10 * np.log10(data_range**2 / mse)

data_range = get_data_range(original_data)

output = cra5_API.encode_era5_as_bin(time_stamp=time_stamp, save_root='./data/CRA5')
encoding_time = output['encoding_time']

output = cra5_API.decode_from_bin(time_stamp, return_format='de_normalized')
x_hat = output['x_hat']
decoding_time = output['decoding_time']

reconstructed = x_hat.cpu().numpy()
mse = np.mean((original_data - reconstructed) ** 2)
psnr = calculate_psnr(original_data, reconstructed, data_range=data_range)

cra5_API.show_image(
    reconstruct_data=x_hat.cpu().numpy(), 
    time_stamp="2024-06-01T00:00:00", 
    show_variables=['z_500', 'q_500', 'u_500', 'v_500', 't_500', 'w_500'],
    save_path='~/work/cra5/img'
)

print(f"Encoding time: {encoding_time:.4f}s")
print(f"Decoding time: {decoding_time:.4f}s")
print(f"MSE: {mse:.8f}")
print(f"PSNR: {psnr:.2f} dB")

bin_path = "./data/CRA5/2024/2024-06-01T00:00:00.bin"
if os.path.exists(bin_path):
    original_size_bytes = original_data.nbytes
    compressed_size_bytes = os.path.getsize(bin_path)
    compression_ratio = original_size_bytes / compressed_size_bytes
    bpp = (compressed_size_bytes * 8) / original_data.size
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"BPP: {bpp:.6f}")
