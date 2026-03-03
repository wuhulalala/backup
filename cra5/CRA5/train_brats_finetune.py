import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
import random
from torch.utils.data.sampler import Sampler
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import wandb
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
def ssim_loss(x, y, window_size=11, reduction='mean'):
    """计算 SSIM Loss (1 - SSIM)，保护图像结构"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 创建高斯窗口
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    
    window = gaussian_window(window_size).to(x.device)
    
    # 对每个通道计算 SSIM
    channels = x.shape[1]
    ssim_vals = []
    
    for c in range(channels):
        x_c = x[:, c:c+1, :, :]
        y_c = y[:, c:c+1, :, :]
        
        mu_x = F.conv2d(x_c, window, padding=window_size//2)
        mu_y = F.conv2d(y_c, window, padding=window_size//2)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x_c ** 2, window, padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.conv2d(y_c ** 2, window, padding=window_size//2) - mu_y_sq
        sigma_xy = F.conv2d(x_c * y_c, window, padding=window_size//2) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        ssim_vals.append(ssim_map.mean())
    
    ssim_val = torch.stack(ssim_vals).mean()
    return 1 - ssim_val  # 返回 loss (1 - SSIM)

# 设置GPU

from cra5.api.brats_api import brats_api


def get_reconstruction_params(model):
    params = []
    param_names = []
    
    for name, param in model.named_parameters():
        # 编码器和解码器
        if name.startswith('g_a.') or name.startswith('g_s.'):
            params.append(param)
            param_names.append(name)
        # 量化卷积层
        elif name.startswith('quant_conv.') or name.startswith('post_quant_conv.'):
            params.append(param)
            param_names.append(name)
    
    return params, param_names


def get_entropy_params(model):
    params = []
    param_names = []
    
    for name, param in model.named_parameters():
        # 超先验编码器/解码器
        if name.startswith('h_a.') or name.startswith('h_s.'):
            params.append(param)
            param_names.append(name)
        # 熵瓶颈
        elif name.startswith('entropy_bottleneck.'):
            params.append(param)
            param_names.append(name)
        # 高斯条件
        elif name.startswith('gaussian_conditional.'):
            params.append(param)
            param_names.append(name)
    
    return params, param_names


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def setup_training_stage(model, stage, lr_reconstruction=1e-4, lr_entropy=1e-4):
    recon_params, recon_names = get_reconstruction_params(model)
    entropy_params, entropy_names = get_entropy_params(model)
    
    if stage == 'stage1':
        # 第一阶段：只训练重建部分
        logging.info("\n" + "="*60)
        logging.info("第一阶段：训练重建部分（g_a, g_s, quant_conv, post_quant_conv）")
        logging.info("="*60)
        
        unfreeze_params(recon_params)
        freeze_params(entropy_params)
        
        optimizer = optim.Adam(recon_params, lr=lr_reconstruction)
        trainable_params = recon_params
        
        logging.info(f"可训练参数: {len(recon_names)} 组")
        logging.info(f"冻结参数: {len(entropy_names)} 组（熵编码部分）")
        
    elif stage == 'stage2':
        # 第二阶段：只训练熵编码部分
        logging.info("\n" + "="*60)
        logging.info("第二阶段：训练熵编码部分（h_a, h_s, entropy_bottleneck）")
        logging.info("="*60)
        
        freeze_params(recon_params)
        unfreeze_params(entropy_params)
        
        optimizer = optim.Adam(entropy_params, lr=lr_entropy)
        trainable_params = entropy_params
        
        logging.info(f"冻结参数: {len(recon_names)} 组（重建部分）")
        logging.info(f"可训练参数: {len(entropy_names)} 组")
        
    elif stage == 'both':
        # 同时训练两部分（使用较小的学习率）
        logging.info("\n" + "="*60)
        logging.info("联合微调：同时训练重建部分和熵编码部分")
        logging.info("="*60)
        
        unfreeze_params(recon_params)
        unfreeze_params(entropy_params)
        
        # 分组学习率
        optimizer = optim.Adam([
            {'params': recon_params, 'lr': lr_reconstruction},
            {'params': entropy_params, 'lr': lr_entropy}
        ])
        trainable_params = recon_params + entropy_params
        
        logging.info(f"重建部分参数: {len(recon_names)} 组, lr={lr_reconstruction}")
        logging.info(f"熵编码部分参数: {len(entropy_names)} 组, lr={lr_entropy}")
    
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # 统计可训练参数数量
    total_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"可训练参数量: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.1f}%)")
    
    return optimizer, trainable_params


class ShuffleAfterBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        idx = self.indices[self.current_idx]
        self.current_idx += 1
        return idx
    
    def shuffle_remaining(self):
        if self.current_idx < len(self.indices):
            remaining = self.indices[self.current_idx:]
            random.shuffle(remaining)
            self.indices[self.current_idx:] = remaining
    
    def __len__(self):
        return len(self.indices)
    
    def reset(self):
        self.indices = list(range(len(self.dataset)))
        random.shuffle(self.indices)
        self.current_idx = 0


class BraTSDataset(Dataset):
    def __init__(self, brats_api, patient_list, device='cuda'):
        self.brats_api = brats_api
        self.patient_list = patient_list.copy()
        self.device = device
        
    def __len__(self):
        return len(self.patient_list)
    
    def shuffle(self):
        random.shuffle(self.patient_list)
    
    def __getitem__(self, idx):
        patient_id = self.patient_list[idx]
        
        data = self.brats_api.load_patient_volume_cra5_format(patient_id)
        data = torch.from_numpy(data).to(self.device)
        x = self.brats_api.normalization_cra5(data, patient_id=patient_id).unsqueeze(0)
        
        batch1 = x[:, 0:268, :, :]
        batch2 = x[:, 268:536, :, :]
        batch3 = x[:, 536:804, :, :]
        batch4 = x[:, 804:1072, :, :]
        
        return {
            'batch1': batch1.squeeze(0),
            'batch2': batch2.squeeze(0),
            'batch3': batch3.squeeze(0),
            'batch4': batch4.squeeze(0),
            'patient_id': patient_id
        }


def compute_loss(model, out_net, x, stage, lmbda_mse=1.0, lmbda_bpp=0.01, lmbda_kl=1.0): # 增加 lmbda_kl 参数
    loss_dict = {}
    
    # 1. MSE 损失
    mse_loss = nn.functional.mse_loss(out_net['x_hat'], x)
    loss_dict['mse'] = mse_loss.item()
    
    # 2. BPP 损失
    bpp_loss = torch.tensor(0.0, device=x.device)
    if 'likelihoods' in out_net:
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]  # B * H * W
        for key, likelihoods in out_net['likelihoods'].items():
            if likelihoods is not None:
                bpp_loss += -torch.log(likelihoods).sum() / (math.log(2) * num_pixels)
    loss_dict['bpp'] = bpp_loss.item()

    # 3. KL 散度损失（论文公式8）------------------------------------------------
    # 模型返回 posterior (DiagonalGaussianDistribution)，它有内置的 kl() 方法
    # kl() 实现: 0.5 * mean(mu^2 + var - 1 - log_var)，对应标准正态先验
    kl_loss = torch.tensor(0.0, device=x.device)
    if 'posterior' in out_net and out_net['posterior'] is not None:
        posterior = out_net['posterior']
        # posterior.kl() 返回 shape [B] 的 KL 散度
        kl_loss = posterior.kl().mean()
    loss_dict['kl'] = kl_loss.item()
    # ---------------------------------------------------------------------

    # 根据训练阶段组合损失
    if stage == 'stage1':
        # 修正：Stage 1 = MSE + KL (论文 Eq. 8)
        # 注意：论文中各项系数为 0.5，这里可以通过调整 lambda 实现
        # 通常 VAE 训练中 KL 权重 beta 可能需要退火 (Annealing)，但论文未提及
        loss = lmbda_mse * mse_loss + lmbda_kl * kl_loss
        loss_dict['total'] = loss.item()
        
    elif stage == 'stage2':
        # Stage 2: MSE + BPP (论文 Eq. 2)
        # 此时通常不再显式加 KL，因为 BPP Loss (Rate) 本身就在最小化熵，
        # 起到了类似的作用（让分布紧凑）
        loss = lmbda_mse * mse_loss + lmbda_bpp * bpp_loss
        loss_dict['total'] = loss.item()
        
    elif stage == 'both':
        # 联合训练
        loss = lmbda_mse * mse_loss + lmbda_bpp * bpp_loss + lmbda_kl * kl_loss
        loss_dict['total'] = loss.item()

    return loss, loss_dict


def train_epoch(model, dataloader, optimizer, optimizer_aux, device, epoch, stage, 
                lmbda_mse=1.0, lmbda_bpp=0.01, lmbda_kl=1.0, use_wandb=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_bpp = 0.0
    total_kl = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{stage}]')
    for batch_idx, batch_data in enumerate(pbar):
        batch_loss = 0.0
        batch_mse = 0.0
        batch_bpp = 0.0
        batch_kl = 0.0
        
        optimizer.zero_grad()  # 在循环外清零一次
        
        for batch_name in ['batch1', 'batch2', 'batch3', 'batch4']:
            x = batch_data[batch_name].to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            
            out_net = model(x)
            
            loss, loss_dict = compute_loss(model, out_net, x, stage, lmbda_mse, lmbda_bpp, lmbda_kl)
            
            aux_loss = model.aux_loss()
            
            # 累积梯度，除以 4 取平均
            if stage == 'stage1':
                (loss / 4).backward()
            else:
                ((loss + 0.01 * aux_loss) / 4).backward()
            
            batch_loss += loss_dict.get('total', 0)
            batch_mse += loss_dict.get('mse', 0)
            batch_bpp += loss_dict.get('bpp', 0)
            batch_kl += loss_dict.get('kl', 0)
        
        # 循环外：梯度裁剪并更新一次
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if stage != 'stage1' and optimizer_aux is not None:
            optimizer_aux.zero_grad()
            aux_loss = model.aux_loss()
            aux_loss.backward()
            optimizer_aux.step()
        
        batch_loss /= 4
        batch_mse /= 4
        batch_bpp /= 4
        batch_kl /= 4
        
        total_loss += batch_loss
        total_mse += batch_mse
        total_bpp += batch_bpp
        total_kl += batch_kl
        num_batches += 1
        
        # 每个batch后对剩余的数据做shuffle
        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, ShuffleAfterBatchSampler):
            dataloader.sampler.shuffle_remaining()
        
        # 记录每个batch的损失到wandb
        if use_wandb:
            batch_log_dict = {
                'train/batch_loss': batch_loss,
                'train/batch_mse': batch_mse,
                'train/batch_kl': batch_kl,
            }
            if stage != 'stage1':
                batch_log_dict['train/batch_bpp'] = batch_bpp
            # 使用 step 参数来记录batch级别的指标
            wandb.log(batch_log_dict, step=epoch * len(dataloader) + batch_idx)
        
        # 更新进度条（根据阶段显示不同指标）
        if stage == 'stage1':
            pbar.set_postfix({
                'loss': f'{batch_loss:.6f}',
                'mse': f'{batch_mse:.6f}',
                'kl': f'{batch_kl:.4f}'
            })
        else:
            pbar.set_postfix({
                'loss': f'{batch_loss:.6f}',
                'mse': f'{batch_mse:.6f}',
                'bpp': f'{batch_bpp:.4f}',
                'kl': f'{batch_kl:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_bpp = total_bpp / num_batches
    avg_kl = total_kl / num_batches
    
    # 记录epoch级别的平均损失和总损失到wandb
    if use_wandb:
        # 使用最后一个batch的step，这样可以在同一图表中对比batch和epoch级别的指标
        last_batch_step = epoch * len(dataloader) + (num_batches - 1) if num_batches > 0 else epoch * len(dataloader)
        epoch_log_dict = {
            'train/avg_loss': avg_loss,
            'train/total_loss': total_loss,
            'train/avg_mse': avg_mse,
            'train/avg_kl': avg_kl,
        }
        if stage != 'stage1':
            epoch_log_dict['train/avg_bpp'] = avg_bpp
        wandb.log(epoch_log_dict, step=last_batch_step)
    
    return avg_loss, avg_mse, avg_bpp, avg_kl


def validate(model, brats_api, patient_list, device):
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for patient_id in tqdm(patient_list, desc='Validating'):
            original_data = brats_api.load_patient_volume(patient_id)
            
            y_hat = brats_api.encode_to_latent(patient_id=patient_id, latent_type='quantized')
            normalized_x_hat = brats_api.latent_to_reconstruction(y_hat=y_hat)
            x_hat = brats_api.de_normalization_cra5(normalized_x_hat.squeeze(0), patient_id=patient_id)
            reconstructed = x_hat.cpu().numpy()
            
            data_range = original_data.max() - original_data.min()
            if data_range < 1e-6:
                data_range = 1.0
            
            mse = np.mean((original_data - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            
            if mse > 1e-10:
                psnr = 10 * np.log10(data_range**2 / mse)
            else:
                psnr = float('inf')
            
            total_mse += mse
            total_rmse += rmse
            total_psnr += psnr
    
    avg_mse = total_mse / len(patient_list)
    avg_rmse = total_rmse / len(patient_list)
    avg_psnr = total_psnr / len(patient_list)
    
    return avg_mse, avg_rmse, avg_psnr


def setup_logging(output_dir, stage):
    # output_dir 已经包含了 stage 和学习率信息，直接使用
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'training_{stage}_{timestamp}.log'
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    
    return logger, log_file


def parse_args():
    parser = argparse.ArgumentParser(description='BraTS 数据微调脚本')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='/bigdata/datasets/aiocta/brats2023-part-1/',
                        help='BraTS数据根目录')
    parser.add_argument('--save_root', type=str, default='/bigdata/datasets/BraTS_compressed',
                        help='压缩结果保存目录')
    
    # 训练参数
    parser.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'stage2', 'both'],
                        help='训练阶段: stage1(重建), stage2(压缩), both(联合)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--lr_recon', type=float, default=1e-5, help='重建部分学习率')
    parser.add_argument('--lr_entropy', type=float, default=1e-5, help='熵编码部分学习率')
    parser.add_argument('--lr_aux', type=float, default=1e-3, help='辅助优化器学习率')
    
    # 损失权重
    parser.add_argument('--lmbda_mse', type=float, default=0.8, help='MSE损失权重')
    parser.add_argument('--lmbda_bpp', type=float, default=1, help='BPP损失权重')
    parser.add_argument('--lmbda_kl', type=float, default=0.2, help='KL散度损失权重（第一阶段使用）')
    
    # 学习率调度器参数
    parser.add_argument('--T_max', type=int, default=None, 
                        help='余弦退火周期（epoch数），默认等于总epoch数')
    parser.add_argument('--eta_min', type=float, default=1e-6, 
                        help='余弦退火最小学习率')
    
    # 模型参数
    parser.add_argument('--model_version', type=int, default=621, help='模型版本')
    parser.add_argument('--pretrained', type=int, default=False, help='是否预训练权重')
    parser.add_argument('--checkpoint', type=str, default=None, help='恢复训练的检查点路径')
    
    # 保存参数
    parser.add_argument('--save_interval', type=int, default=5, help='保存间隔（epoch）')
    parser.add_argument('--output_dir', type=str, default='/bigdata/cra5/checkpoints/brats_finetune',
                        help='输出目录')
    
    # Wandb 参数
    parser.add_argument('--use_wandb', type=int, default=True, help='使用 wandb 记录训练过程')
    parser.add_argument('--wandb_project', type=str, default='cra5-FineTune', help='Wandb 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb 运行名称（默认使用时间戳+stage）')
    parser.add_argument('--wandb_entity', type=str, default="liscopye-university-of-chinese-academy-of-sciences", help='liscopye-university-of-chinese-academy-of-sciences')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 配置参数
    data_root = args.data_root
    save_root = args.save_root
    device = 'cuda'
    model_version = args.model_version
    pretrained = args.pretrained
    
    # 训练参数
    batch_size = args.batch_size
    num_epochs = args.epochs
    save_interval = args.save_interval
    stage = args.stage
    
    # 构建包含时间戳的输出目录路径
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 输出目录：base_dir/stage/timestamp
    output_dir = Path(args.output_dir) / stage / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger, log_file = setup_logging(str(output_dir), stage)
    
    # 记录输出目录信息
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化 Wandb
    if args.use_wandb:
        wandb_name = args.wandb_name
        if wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_name = f"{stage}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            entity=args.wandb_entity,
            config={
                'stage': stage,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr_recon': args.lr_recon,
                'lr_entropy': args.lr_entropy,
                'lr_aux': args.lr_aux,
                'lmbda_mse': args.lmbda_mse,
                'lmbda_bpp': args.lmbda_bpp,
                'lmbda_kl': args.lmbda_kl,
                'model_version': args.model_version,
                'data_root': args.data_root,
                'save_interval': args.save_interval,
                'pretrained': args.pretrained,
                'T_max': args.T_max if args.T_max else num_epochs,
                'eta_min': args.eta_min,
            },
            tags=[stage, 'brats', 'finetune']
        )
        logger.info(f"Wandb 已初始化: project={args.wandb_project}, name={wandb_name}")
    
    logger.info("="*80)
    logger.info("BraTS 数据微调")
    logger.info("="*80)
    logger.info(f"训练阶段: {stage}")
    logger.info(f"  - stage1: 训练重建部分（g_a, g_s），优化MSE")
    logger.info(f"  - stage2: 冻结重建部分，训练熵编码部分（h_a, h_s），优化BPP")
    logger.info(f"  - both: 联合训练两部分")
    
    # 初始化API
    logger.info("\n初始化 BraTS API...")
    brats_api_instance = brats_api(
        data_root=data_root,
        save_root=save_root,
        device=device,
        model_version=model_version,
        pretrained=pretrained,

    )
    
    # 获取患者列表
    logger.info("\n获取患者列表...")
    all_patients = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and d.startswith("BraTS")
    ]
    all_patients = sorted(all_patients)
    
    # 划分数据集：训练集60%、测试集20%、验证集20%
    # 排序后的顺序：测试集、验证集、训练集
    total = len(all_patients)
    test_split = int(total * 0.2)      # 前20%：测试集
    val_split = int(total * 0.4)        # 20%-40%：验证集
    # 剩余60%：训练集
    
    test_patients = all_patients[:test_split]                    # [0:0.2N]
    val_patients = all_patients[test_split:val_split]           # [0.2N:0.4N]
    train_patients = all_patients[val_split:]                   # [0.4N:N]
    
    logger.info(f"数据集划分（6:2:2）:")
    logger.info(f"  测试集: {len(test_patients)} 个患者 (0-{test_split})")
    logger.info(f"  验证集: {len(val_patients)} 个患者 ({test_split}-{val_split})")
    logger.info(f"  训练集: {len(train_patients)} 个患者 ({val_split}-{total})")
    
    # 保存数据集划分信息（可选）
    split_info_path = output_dir / 'dataset_split.json'
    import json
    split_info = {
        'total_patients': total,
        'test_patients': test_patients,
        'val_patients': val_patients,
        'train_patients': train_patients,
        'split_ratios': {'train': 0.6, 'test': 0.2, 'val': 0.2},
        'split_indices': {'test': (0, test_split), 'val': (test_split, val_split), 'train': (val_split, total)}
    }
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"  数据集划分信息已保存: {split_info_path}")
    
    # 记录数据集信息到 Wandb
    if args.use_wandb:
        wandb.config.update({
            'dataset/total_patients': total,
            'dataset/train_patients': len(train_patients),
            'dataset/val_patients': len(val_patients),
            'dataset/test_patients': len(test_patients),
            'dataset/split_ratios': {'train': 0.6, 'test': 0.2, 'val': 0.2},
        })
    
    # 创建数据集和数据加载器
    train_dataset = BraTSDataset(brats_api_instance, train_patients, device)
    # 使用自定义的Sampler，支持每个batch后重新shuffle
    custom_sampler = ShuffleAfterBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=custom_sampler,  # 使用自定义sampler
        shuffle=False,  # 使用sampler时shuffle必须为False
        num_workers=0,  # 避免多进程问题
    )
    logger.info(f"数据加载器已创建: batch_size={batch_size}, 每个batch后自动shuffle")
    
    # 使用 brats_api 已加载的模型
    model = brats_api_instance.net
    logger.info(f"\n使用 brats_api 中已加载的模型")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 记录模型信息到 Wandb
    if args.use_wandb:
        wandb.config.update({
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
        })
    
    # 根据训练阶段设置优化器
    optimizer, trainable_params = setup_training_stage(
        model, stage, 
        lr_reconstruction=args.lr_recon, 
        lr_entropy=args.lr_entropy
    )
    
    # 记录当前阶段的参数信息到 Wandb
    if args.use_wandb:
        stage_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        wandb.config.update({
            f'model/{stage}_trainable_params': stage_trainable,
        })
    
    # 辅助优化器（用于更新熵模型的quantiles）
    aux_params = [p for n, p in model.named_parameters() if 'quantiles' in n]
    optimizer_aux = optim.Adam(aux_params, lr=args.lr_aux) if aux_params else None
    
    # 设置为训练模式
    model.train()
    
    # 学习率调度器 - 余弦退火策略
    T_max = args.T_max if args.T_max is not None else num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=args.eta_min
    )
    logger.info(f"使用余弦退火学习率调度: T_max={T_max}, eta_min={args.eta_min}")
    
    # 如果有检查点，加载
    start_epoch = 1
    if args.checkpoint is not None:
        logger.info(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"从 epoch {start_epoch} 继续训练")
        
        # 重新初始化CDF表
        logger.info("重新初始化熵模型的CDF表...")
        model.update()
    
    logger.info("\n开始训练...")
    logger.info("="*80)
    logger.info(f"损失权重: lmbda_mse={args.lmbda_mse}, lmbda_bpp={args.lmbda_bpp}, lmbda_kl={args.lmbda_kl}")
    if stage == 'stage1':
        logger.info("  Stage1 损失: L = lmbda_mse * MSE + lmbda_kl * KL（论文公式8）")
    elif stage == 'stage2':
        logger.info("  Stage2 损失: L = lmbda_mse * MSE + lmbda_bpp * BPP（论文公式2）")
    else:
        logger.info("  Both 损失: L = lmbda_mse * MSE + lmbda_bpp * BPP + lmbda_kl * KL")
    logger.info("="*80)
    
    best_mse = float('inf')
    best_bpp = float('inf')
    
    for epoch in range(start_epoch, num_epochs + 1):
        # 每个epoch开始时重置sampler
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, ShuffleAfterBatchSampler):
            train_loader.sampler.reset()
        
        
        # 训练
        train_loss, train_mse, train_bpp, train_kl = train_epoch(
            model, train_loader, optimizer, optimizer_aux, 
            device, epoch, stage, args.lmbda_mse, args.lmbda_bpp, args.lmbda_kl,
            use_wandb=bool(args.use_wandb)
        )
        # 验证
        val_mse, val_rmse, val_psnr = validate(
            model, brats_api_instance, val_patients, device
        )
        
        # 更新学习率（余弦退火策略）
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # 余弦退火不需要传入指标
        new_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        logger.info(f"\nEpoch {epoch}/{num_epochs}:")
        if stage == 'stage1':
            logger.info(f"  训练 - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, KL: {train_kl:.4f}")
        else:
            logger.info(f"  训练 - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, BPP: {train_bpp:.4f}, KL: {train_kl:.4f}")
        logger.info(f"  验证 - MSE: {val_mse:.8f}, RMSE: {val_rmse:.6f}, PSNR: {val_psnr:.2f} dB")
        if old_lr != new_lr:
            logger.info(f"  ⚠️  学习率已调整: {old_lr:.2e} -> {new_lr:.2e}")
        else:
            logger.info(f"  学习率: {new_lr:.2e}")
        
        # 记录到 Wandb
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/mse': train_mse,
                'train/kl': train_kl,
                'val/mse': val_mse,
                'val/rmse': val_rmse,
                'val/psnr': val_psnr,
                'learning_rate': new_lr,
            }
            
            if stage != 'stage1':
                log_dict['train/bpp'] = train_bpp
            
            # 记录最佳指标
            if stage == 'stage1':
                log_dict['best/val_mse'] = best_mse
            elif stage == 'stage2':
                log_dict['best/train_bpp'] = best_bpp
            else:
                log_dict['best/val_mse'] = best_mse
            
            wandb.log(log_dict)
        
        # 根据训练阶段选择保存条件
        is_best = False
        if stage == 'stage1':
            # 第一阶段：根据MSE判断
            if val_mse < best_mse:
                best_mse = val_mse
                is_best = True
        elif stage == 'stage2':
            # 第二阶段：根据BPP判断（在MSE相近的情况下）
            if train_bpp < best_bpp:
                best_bpp = train_bpp
                is_best = True
        else:
            # 联合训练：根据MSE判断
            if val_mse < best_mse:
                best_mse = val_mse
                is_best = True
        
        # 保存最佳模型
        if is_best:
            checkpoint_path = output_dir / f'best_model_{stage}.pth'
            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_aux_state_dict': optimizer_aux.state_dict() if optimizer_aux else None,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_psnr': val_psnr,
                'train_bpp': train_bpp,
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"  ✓ 保存最佳模型: {checkpoint_path}")
            
            # 记录最佳模型到 Wandb
            if args.use_wandb:
                wandb.log({
                    'best_model/epoch': epoch,
                    'best_model/val_mse': val_mse,
                    'best_model/val_psnr': val_psnr,
                })
                # 可选：保存模型文件到 wandb
                # wandb.save(str(checkpoint_path))
        
        # 定期保存
        if epoch % save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_{stage}_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'stage': stage,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_aux_state_dict': optimizer_aux.state_dict() if optimizer_aux else None,
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_psnr': val_psnr,
                'train_bpp': train_bpp,
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"  ✓ 保存检查点: {checkpoint_path}")
        
        # 更新CDF表（用于压缩/解压）
        model.update()
        
        logger.info("-"*80)
    
    logger.info("\n" + "="*80)
    logger.info("训练完成！")
    logger.info("="*80)
    logger.info(f"训练阶段: {stage}")
    if stage == 'stage1':
        logger.info(f"最佳验证MSE: {best_mse:.8f}")
        logger.info("\n下一步建议:")
        logger.info(f"  python train_brats_finetune.py --stage stage2 --checkpoint {output_dir}/best_model_{stage}.pth")
    elif stage == 'stage2':
        logger.info(f"最佳训练BPP: {best_bpp:.6f}")
        logger.info("\n模型已完成两阶段训练！")
    else:
        logger.info(f"最佳验证MSE: {best_mse:.8f}")
    logger.info(f"检查点保存在: {output_dir}")
    logger.info(f"日志文件保存在: {log_file}")
    
    # 完成 Wandb 运行
    if args.use_wandb:
        wandb.finish()
        logger.info("Wandb 运行已结束")


if __name__ == '__main__':
    main()

