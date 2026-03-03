
import sys
import os
import json
import torch
import torch.nn.functional as F
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import filesize, write_uints, write_bytes, read_uints, read_bytes

from cra5.models.compressai.zoo import vaeformer_pretrained
from cra5.models.vaeformer import VAEformer

current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)


class brats_api:
    def __init__(
        self,
        data_root="/bigdata/datasets/aiocta/brats2023-part-1/",
        save_root="/bigdata/datasets/BraTS_compressed",
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_version=620,
        pretrained=True,
        statistics_path="/bigdata/datasets/BraTS_compressed/brats_statistics.json",
        auto_compute_stats=True,
        stats_patient_list=None,
        ):
        
        self.device = device 
        
        self.data_root = data_root
        self.save_root = save_root
        self.model_version = model_version
        
        self.modalities = ["t1n", "t1c", "t2w", "t2f"]
        
        self.statistics_path = statistics_path
        
        if statistics_path and os.path.exists(statistics_path):
            self.load_statistics(statistics_path)
            self.mean = torch.zeros(620, 1, 1).to(device)
            self.std = torch.ones(620, 1, 1).to(device)
        elif auto_compute_stats:
            if stats_patient_list is None:
                try:
                    all_patients = [
                        d
                        for d in os.listdir(data_root)
                                   if os.path.isdir(os.path.join(data_root, d)) 
                        and d.startswith("BraTS")
                    ]
                    stats_patient_list = sorted(all_patients)
                except Exception as e:
                    self.mean = torch.zeros(620, 1, 1).to(device)
                    self.std = torch.ones(620, 1, 1).to(device)
                    stats_patient_list = []
            
            if stats_patient_list:
                default_stats_path = os.path.join(save_root, "brats_statistics.json")
                self.mean = torch.zeros(620, 1, 1).to(device)
                self.std = torch.ones(620, 1, 1).to(device)
                self.compute_statistics(
                    stats_patient_list, save_path=default_stats_path
                )
        else:
            self.mean = torch.zeros(620, 1, 1).to(device)
            self.std = torch.ones(620, 1, 1).to(device)
        
        self.net = self._build_model(pretrained)
        
    def _build_model(self, pretrained):
        if pretrained:
            print("加载预训练模型")
            model = (
                vaeformer_pretrained(
                    quality=self.model_version,
                    pretrained=True,
                )
                .eval()
                .to(self.device)
            )
        else:
            print("重新开始训练")
            model = (
                VAEformer(model_version=self.model_version)
                .eval()
                .to(self.device)
            )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        model.update()
        
        return model
    
    def load_patient_volume(self, patient_id):
        patient_dir = os.path.join(self.data_root, patient_id)
        
        modality_data = []
        for modality in self.modalities:
            nii_path = os.path.join(patient_dir, f"{patient_id}-{modality}.nii")
            
            if not os.path.exists(nii_path):
                raise FileNotFoundError(f"文件不存在: {nii_path}")
            
            nii_img = nib.load(nii_path)
            data = np.array(nii_img.get_fdata(), dtype=np.float32)
            modality_data.append(data)
        
        volume = np.stack(modality_data, axis=0)
        volume = np.concatenate([volume[i] for i in range(4)], axis=2)
        volume = np.transpose(volume, (2, 0, 1))
        
        
        return volume
    
    def load_patient_volume_cra5_format(self, patient_id):
        volume = self.load_patient_volume(patient_id)
        
        padded_volume = []
        for i in range(4):
            modality_slices = volume[i*155:(i+1)*155, :, :]  # (155, 240, 240)
            # padding到268维，用循环方式填充（从头开始重复）
            padding_size = 268 - 155
            padded_modality = np.pad(
                modality_slices,
                ((0, padding_size), (0, 0), (0, 0)),
                mode='wrap'
            )  # (268, 240, 240)
            padded_volume.append(padded_modality)
        
        padded_volume = np.concatenate(padded_volume, axis=0)  # (1072, 240, 240)
        return padded_volume
    
    def encode_to_latent(self, patient_id, latent_type='float'):
        data = self.load_patient_volume_cra5_format(patient_id)
        data = torch.from_numpy(data).to(self.device)
        x = self.normalization_cra5(data, patient_id=patient_id).unsqueeze(0)
        
        batch1 = x[:, 0:268, :, :]
        batch2 = x[:, 268:536, :, :]
        batch3 = x[:, 536:804, :, :]
        batch4 = x[:, 804:1072, :, :]

        with torch.no_grad():
            if latent_type == "float":
                y1, _, _ = self.net.encode_latent(batch1)
                y2, _, _ = self.net.encode_latent(batch2)
                y3, _, _ = self.net.encode_latent(batch3)
                y4, _, _ = self.net.encode_latent(batch4)
                return [y1, y2, y3, y4]
            elif latent_type == "quantized":
                _, y_hat1, _ = self.net.encode_latent(batch1, type='quantized')
                _, y_hat2, _ = self.net.encode_latent(batch2, type='quantized')
                _, y_hat3, _ = self.net.encode_latent(batch3, type='quantized')
                _, y_hat4, _ = self.net.encode_latent(batch4, type='quantized')
                return [y_hat1, y_hat2, y_hat3, y_hat4]
            else:
                raise ValueError(f"Invalid latent_type: {latent_type}. Must be 'float' or 'quantized'")
    
    def latent_to_bin(self, y, save_path=None):
        with torch.no_grad():
            outputs = []
            for i, y_batch in enumerate(y):
                output = self.net.compress_from_latent(y_batch)
                outputs.append(output)

                if save_path:
                    batch_path = save_path.replace(".bin", f"_part{i}.bin")
                    os.makedirs(os.path.dirname(batch_path), exist_ok=True)

                    with Path(batch_path).open("wb") as f:
                        out_strings = output["strings"]
                        shape = output["z_shape"]

                        write_uints(f, (shape[0], shape[1], len(out_strings)))

                        for s in out_strings:
                            write_uints(f, (len(s[0]),))
                            write_bytes(f, s[0])
            return outputs
    
    def bin_to_latent(self, bin_path):
        latents = []
        for i in range(4):
            part_path = bin_path.replace(".bin", f"_part{i}.bin")
            with Path(part_path).open("rb") as f:
                lstrings = []
                shape = read_uints(f, 2)
                n_strings = read_uints(f, 1)[0]
                
                for _ in range(n_strings):
                    s = read_bytes(f, read_uints(f, 1)[0])
                    lstrings.append([s])
                
                with torch.no_grad():
                    latent = self.net.decompress(
                        lstrings, shape, return_format="latent"
                    )
                    latents.append(latent)
        return latents
    
    def latent_to_reconstruction(self, y_hat):
        with torch.no_grad():
            x_hat1 = self.net.decode_latent(y_hat[0])
            x_hat2 = self.net.decode_latent(y_hat[1])
            x_hat3 = self.net.decode_latent(y_hat[2])
            x_hat4 = self.net.decode_latent(y_hat[3])

            # 从每个模态的268维中提取前155维
            x_hat1 = x_hat1[:, :155, :, :]
            x_hat2 = x_hat2[:, :155, :, :]
            x_hat3 = x_hat3[:, :155, :, :]
            x_hat4 = x_hat4[:, :155, :, :]

            x_hat = torch.cat([x_hat1, x_hat2, x_hat3, x_hat4], dim=1)
            return x_hat
    
    def compress_patient_cra5_format(self, patient_id, save_bin=True):
        st1 = time.time()
        
        data = self.load_patient_volume_cra5_format(patient_id)
        data = torch.from_numpy(data).to(self.device)
        x = self.normalization_cra5(data, patient_id=patient_id).unsqueeze(0)
        
        st2 = time.time()
        
        with torch.no_grad():
            output = self.net.compress(x)
        
        st3 = time.time()
        
        if save_bin:
            file_url = f"{self.save_root}/{patient_id}_cra5_format.bin"
            os.makedirs(os.path.dirname(file_url), exist_ok=True)
            
            with Path(file_url).open("wb") as f:
                out_strings = output["strings"]
                shape = output["z_shape"]
                
                write_uints(f, (shape[0], shape[1], len(out_strings)))
                
                for s in out_strings:
                    write_uints(f, (len(s[0]),))
                    write_bytes(f, s[0])
        else:
            file_url = None
        
        st4 = time.time()
        
        original_size = data.numel() * 4
        compressed_size = sum(len(s[0]) for s in output["strings"])
        compression_ratio = original_size / compressed_size
        
        return dict(
            output=output,
            reading_time=st2 - st1,
            encoding_time=st3 - st2,
            saving_time=st4 - st3,
            save_path=file_url,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            bpp=compressed_size * 8.0 / (620 * 240 * 240),  # 原始数据是620个切片
        )
    
    def decompress_patient_cra5_format(self, patient_id, return_format="normalized"):
        bin_path = f"{self.save_root}/{patient_id}_cra5_format.bin"
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"压缩文件不存在: {bin_path}")
        
        decoding_start = time.time()
        
        with Path(bin_path).open("rb") as f:
            lstrings = []
            shape = read_uints(f, 2)
            n_strings = read_uints(f, 1)[0]
            
            for _ in range(n_strings):
                s = read_bytes(f, read_uints(f, 1)[0])
                lstrings.append([s])
            
            with torch.no_grad():
                output = self.net.decompress(lstrings, shape)
        
        decoding_time = time.time() - decoding_start
        
        # 从1072个切片中提取原始的620个切片（每个模态的前155个）
        x_hat_full = output["x_hat"]
        x_hat_original = torch.cat([
            x_hat_full[:, 0:155, :, :],      # 模态1
            x_hat_full[:, 268:423, :, :],    # 模态2 (268+155=423)
            x_hat_full[:, 536:691, :, :],    # 模态3 (536+155=691)
            x_hat_full[:, 804:959, :, :],    # 模态4 (804+155=959)
        ], dim=1)
        
        if return_format == "normalized":
            return dict(
                x_hat=x_hat_original,
                decoding_time=decoding_time,
            )
        elif return_format == "denormalized":
            x_hat = self.de_normalization_cra5(
                x_hat_original.squeeze(0),
                patient_id=patient_id
            )
            return dict(
                x_hat=x_hat,
                decoding_time=decoding_time,
            )
    
    def normalization_cra5(self, data, patient_id=None):
        if patient_id is not None:
            mean, std = self.get_patient_statistics(patient_id)
        else:
            mean, std = self.mean, self.std
        
        if data.shape[0] == 620:
            normalized = (data - mean) / std
            return normalized
        
        if data.shape[0] == 1072 and mean.shape[0] == 620:
            mean_1d = mean.squeeze()
            std_1d = std.squeeze()
            padded_mean = []
            padded_std = []
            for i in range(4):
                modality_mean = mean_1d[i*155:(i+1)*155]
                modality_std = std_1d[i*155:(i+1)*155]
                padding_size = 268 - 155
                # 用循环方式填充（与数据 padding 方式一致）
                wrap_mean = modality_mean[:padding_size]
                wrap_std = modality_std[:padding_size]
                padded_modality_mean = torch.cat([modality_mean, wrap_mean])
                padded_modality_std = torch.cat([modality_std, wrap_std])
                padded_mean.append(padded_modality_mean)
                padded_std.append(padded_modality_std)
            mean = torch.cat(padded_mean, dim=0)[:, None, None]
            std = torch.cat(padded_std, dim=0)[:, None, None]
        
        normalized = (data - mean) / std
        
        return normalized
    
    def de_normalization_cra5(self, data, patient_id=None):
        if patient_id is not None:
            mean, std = self.get_patient_statistics(patient_id)
        else:
            mean, std = self.mean, self.std
        
        if data.shape[0] == 620:
            denormalized = data * std + mean
            return denormalized
        
        if data.shape[0] == 1072 and mean.shape[0] == 620:
            mean_1d = mean.squeeze()
            std_1d = std.squeeze()
            padded_mean = []
            padded_std = []
            for i in range(4):
                modality_mean = mean_1d[i*155:(i+1)*155]
                modality_std = std_1d[i*155:(i+1)*155]
                padding_size = 268 - 155
                # 用循环方式填充（与数据 padding 方式一致）
                wrap_mean = modality_mean[:padding_size]
                wrap_std = modality_std[:padding_size]
                padded_modality_mean = torch.cat([modality_mean, wrap_mean])
                padded_modality_std = torch.cat([modality_std, wrap_std])
                padded_mean.append(padded_modality_mean)
                padded_std.append(padded_modality_std)
            mean = torch.cat(padded_mean, dim=0)[:, None, None]
            std = torch.cat(padded_std, dim=0)[:, None, None]
        
        denormalized = data * std + mean
        
        return denormalized
    
    def compute_statistics(self, patient_list, save_path=None):
        
        patient_stats = {}
        
        for patient_id in patient_list:
            print(f"  处理: {patient_id}")
            try:
                volume = self.load_patient_volume(patient_id)
                
                patient_mean = []
                patient_std = []
                
                for slice_idx in range(620):
                    slice_data = volume[slice_idx, :, :]
                    patient_mean.append(float(np.mean(slice_data)))
                    std_val = float(np.std(slice_data))
                    if std_val == 0:
                        std_val = 1.0
                    patient_std.append(std_val)
                
                patient_stats[patient_id] = {
                    "mean": patient_mean,
                    "std": patient_std,
                }
                
            except Exception as e:
                print(f"  ⚠️  跳过 {patient_id}: {e}")
        
        if patient_stats:
            all_means = np.array([stats["mean"] for stats in patient_stats.values()])
            all_stds = np.array([stats["std"] for stats in patient_stats.values()])
            mean = np.mean(all_means, axis=0).astype(np.float32)
            std = np.mean(all_stds, axis=0).astype(np.float32)
            std[std == 0] = 1.0
        else:
            mean = np.zeros(620, dtype=np.float32)
            std = np.ones(620, dtype=np.float32)
        
        if save_path:
            stats = {
                "per_patient": patient_stats,
            }
            
            with open(save_path, "w") as f:
                json.dump(stats, f, indent=2)
        self.mean = torch.from_numpy(mean[:, None, None]).to(self.device)
        self.std = torch.from_numpy(std[:, None, None]).to(self.device)
        
        self.patient_stats = patient_stats
        
        return mean, std
    
    def load_statistics(self, path):
        with open(path, "r") as f:
            stats = json.load(f)
        
        if "per_patient" in stats:
            self.patient_stats = stats.get("per_patient", {})
        else:
            self.patient_stats = {}
    
    def get_patient_statistics(self, patient_id):
        if hasattr(self, 'patient_stats') and patient_id in self.patient_stats:
            patient_stat = self.patient_stats[patient_id]
            mean = np.array(patient_stat["mean"], dtype=np.float32)
            std = np.array(patient_stat["std"], dtype=np.float32)
            std[std == 0] = 1.0
            mean = torch.from_numpy(mean[:, None, None]).to(self.device)
            std = torch.from_numpy(std[:, None, None]).to(self.device)
            return mean, std
        else:
            return self.mean, self.std

