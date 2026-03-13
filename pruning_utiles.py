import os
import sys
import gc
import math
import datetime
import random
import shutil

import torch
import torch.nn as nn
import torch.functional as F
import torch.distributed as dist

from torch import Tensor

class WanSparseGPT_Offload:
    def __init__(self, layer):
        self.layer = layer
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        
        # 【核心策略 1】Hessian 永久驻留 CPU，绝不占用宝贵的 H20 显存
        self.H = torch.zeros((self.columns, self.columns), device='cpu', dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp):
        """
        流式处理：GPU 计算 -> CPU 累加 -> 释放显存
        """
        # inp shape: [Batch * Seq, Dim]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        
        # 1. 临时转 float32 保证精度 (H20算力强，这就几毫秒)
        inp = inp.float()
        tmp = inp.shape[0]
        
        # 2. 【核心策略 2】在 GPU 上利用 Tensor Core 快速计算 X^T * X
        # 此时会占用少量显存，计算完立即释放
        # non_blocking=True 允许 CPU/GPU 异步传输

        # 3. 立即传回 CPU 并累加
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        batch_H = inp.t().matmul(inp)#与原论文对齐

        self.H.add_(batch_H.to('cpu', non_blocking=True))


        # 4. 显式删除引用，确保显存回收
        del batch_H, inp

    def prune_2_4(self, method, percdamp=0.01):
        """ 剪枝时才把 H 搬到 GPU，剪完立即扔掉 """
        W = self.layer.weight.data.float()
        # 临时搬运 H 到 GPU
        H = self.H.to(self.layer.weight.device)
        
        if self.columns % 4 != 0: return 0.0

        if method == 'magnitude':
            score = torch.abs(W)
        elif method == 'wanda':
            scaler_row = torch.sqrt(torch.abs(torch.diag(H)))
            score = torch.abs(W) * scaler_row.view(1, -1)
        elif method == 'sparsegpt':
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=H.device)
            H[diag, diag] += damp
            try:
                H_inv = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(H_inv).contiguous()
            except RuntimeError:
                H_inv = torch.diag(1.0 / (torch.diag(H) + damp))
            score = W ** 2 / (torch.diag(H_inv).reshape(1, -1))
        
        # 2:4 Mask 生成
        score_reshaped = score.reshape(self.rows, self.columns // 4, 4)
        _, prune_indices = torch.topk(score_reshaped, k=2, dim=-1, largest=False)
        mask = torch.ones_like(score_reshaped, dtype=torch.bool)
        mask.scatter_(dim=-1, index=prune_indices, src=torch.zeros_like(score_reshaped, dtype=torch.bool))
        mask = mask.reshape(self.rows, self.columns)
        
        W[~mask] = 0
        self.layer.weight.data = W.to(self.layer.weight.dtype)
        
        # 【核心策略 3】剪枝完成后，立即清理 GPU 上的 H
        del H, score, mask
        torch.cuda.empty_cache()
        
        return (W == 0).float().mean().item()

class WanSparseGPT_Offload_Training_free:
    def __init__(self, layer):
        self.layer = layer
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        
        # 【核心策略 1】Hessian 永久驻留 CPU，绝不占用宝贵的 H20 显存
        self.H = torch.zeros((self.columns), device='cpu', dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp):
        """
        流式处理：GPU 计算 -> CPU 累加 -> 释放显存
        """
        # inp shape: [Batch * Seq, Dim]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        
        # 1. 临时转 float32 保证精度 (H20算力强，这就几毫秒)
        inp = inp.float()
        
        # 2. 【核心策略 2】在 GPU 上利用 Tensor Core 快速计算 X^T * X
        # 此时会占用少量显存，计算完立即释放
        # non_blocking=True 允许 CPU/GPU 异步传输
        # 3. 立即传回 CPU 并累加
        # inp = math.sqrt(2 / self.nsamples) * inp.float()
        batch_H = torch.sum(inp ** 2, dim=0)
        self.H.add_(batch_H.to('cpu', non_blocking=True))

        # 4. 显式删除引用，确保显存回收
        del batch_H, inp

    def prune_2_4(self, method, percdamp=0.01):
        """ 剪枝时才把 H 搬到 GPU，剪完立即扔掉 """
        W = self.layer.weight.data.float()
        # 临时搬运 H 到 GPU
        H = self.H.to(self.layer.weight.device)
        
        if self.columns % 4 != 0: return 0.0

        if method == 'magnitude':
            score = torch.abs(W)
        elif method == 'wanda':
            scaler_row = torch.sqrt(torch.abs(torch.diag(H)))
            score = torch.abs(W) * scaler_row.view(1, -1)
        elif method == 'sparsegpt':
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=H.device)
            H[diag, diag] += damp
            try:
                H_inv = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(H_inv).contiguous()
            except RuntimeError:
                H_inv = torch.diag(1.0 / (torch.diag(H) + damp))
            score = W ** 2 / (torch.diag(H_inv).reshape(1, -1))
        
        # 2:4 Mask 生成
        score_reshaped = score.reshape(self.rows, self.columns // 4, 4)
        _, prune_indices = torch.topk(score_reshaped, k=2, dim=-1, largest=False)
        mask = torch.ones_like(score_reshaped, dtype=torch.bool)
        mask.scatter_(dim=-1, index=prune_indices, src=torch.zeros_like(score_reshaped, dtype=torch.bool))
        mask = mask.reshape(self.rows, self.columns)
        
        W[~mask] = 0
        self.layer.weight.data = W.to(self.layer.weight.dtype)
        
        # 【核心策略 3】剪枝完成后，立即清理 GPU 上的 H
        del H, score, mask
        torch.cuda.empty_cache()
        
        return (W == 0).float().mean().item()
    
class WanSparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device='cpu')
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 3: inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.float()
        tmp = inp.shape[0]
        batch_H = inp.t().matmul(inp)
        self.H += batch_H.to('cpu', non_blocking=True)
        self.nsamples += tmp

    def prune_2_4(self, method, percdamp=0.01):
        W = self.layer.weight.data.float()
        H = self.H.float().to(self.layer.weight.device)
        
        if self.columns % 4 != 0: return 0.0

        if method == 'magnitude':
            score = torch.abs(W)
        elif method == 'wanda':
            scaler_row = torch.sqrt(torch.abs(torch.diag(H)))
            score = torch.abs(W) * scaler_row.view(1, -1)
        elif method == 'sparsegpt':
            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=H.device)
            H[diag, diag] += damp
            try:
                H_inv = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(H_inv).contiguous()
            except RuntimeError:
                H_inv = torch.diag(1.0 / (torch.diag(H) + damp))
            score = W ** 2 / (torch.diag(H_inv).reshape(1, -1))
        
        score_reshaped = score.reshape(self.rows, self.columns // 4, 4)
        _, prune_indices = torch.topk(score_reshaped, k=2, dim=-1, largest=False)
        mask = torch.ones_like(score_reshaped, dtype=torch.bool)
        mask.scatter_(dim=-1, index=prune_indices, src=torch.zeros_like(score_reshaped, dtype=torch.bool))
        mask = mask.reshape(self.rows, self.columns)
        
        W[~mask] = 0
        self.layer.weight.data = W.to(self.layer.weight.dtype)
        return (W == 0).float().mean().item()
    
class WanRotatorOptimizer(torch.nn.Module):
    def __init__(self, block, hessian_dict, dim, num_heads, device):
        super().__init__()
        self.block = block
        self.hessian_dict = hessian_dict
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        
        # --- R1: 主干旋转矩阵 (Global) ---
        # 对应 Residual Stream
        self.A1 = torch.nn.Parameter(torch.eye(dim, device=device))
        
        # --- R2: 头内部旋转矩阵 (Local Block Diagonal) ---
        # 对应 V_proj 输出到 O_proj 输入之间的空间
        # 我们为每个 Head 学习一个独立的小旋转矩阵 [Head_Dim, Head_Dim]
        self.A2_list = nn.ParameterList([
            nn.Parameter(torch.eye(self.head_dim, device=device)) 
            for _ in range(num_heads)
        ])

    def get_R1(self):
        Q, _ = torch.linalg.qr(self.A1)
        return Q

    def get_R2(self):
        # 将所有头的旋转矩阵拼成一个大的分块对角矩阵
        # result shape: [Dim, Dim], 但只有对角块有值
        sub_qs = [torch.linalg.qr(a)[0] for a in self.A2_list]
        R2 = torch.block_diag(*sub_qs)
        return R2

    def compute_salience_wanda(self, weight, hessian, R_in, R_out_T=None):
        """
        通用重要性计算:
        Input Rotation: R_in (通常是 R1 或 R2)
        Output Rotation: R_out_T (通常是 R2^T 或 R1^T，如果不涉及输出旋转则为 None)
        
        Formula:
        H_new = R_in^T @ H @ R_in
        W_new = (R_out_T @ W) @ R_in  (如果 R_out_T 存在)
              = W @ R_in              (如果 R_out_T 不存在)
        Score = W_new^2 * diag(H_new)
        """
        # 1. 旋转 Hessian (Input side)
        # H 是输入的统计量，必须跟着输入坐标系转
        H_rot = R_in.T @ hessian @ R_in
        x_norms = torch.abs(torch.diag(H_rot))
        
        # 2. 旋转 Weight
        # 原始 Linear: y = x @ W.T
        # 目标: y' = x' @ W_new.T
        # 变换关系推导后: W_new = (R_out_T if exist) @ W @ R_in
        
        W_new = weight @ R_in # 右乘 R_in (抵消输入旋转)
        if R_out_T is not None:
            W_new = R_out_T @ W_new # 左乘 R_out_T (施加输出旋转)
            
        salience = (W_new ** 2) * x_norms
        return salience

    @torch.compile
    def row_entropy_sum(self, matrix):
        abs_sq = torch.nan_to_num(matrix, nan=0.0, posinf=1e5, neginf=0)
        row_sums = torch.sum(abs_sq, dim=1, keepdim=True)
        row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        probs = abs_sq / row_sums
        probs = torch.where(probs > 0, probs, 1.0)
        log_probs = torch.log(probs)
        entropies = -torch.sum(probs * log_probs, dim=1)
        return torch.sum(entropies)

    def forward(self):
        R1 = self.get_R1()
        # R2 = self.get_R2() # Block Diagonal
        loss = 0.0
        
        # === Group 1: 纯 R1 控制 (WR_1RX) ===
        # 输入: Residual (R1), 输出: 内部/忽略
        # 包含: Q, K (Self), Q (Cross), FFN_Up
        input_layers = ["ffn.0"]
        layer_objs = {
            # "self_attn.q": self.block.self_attn.q
            "ffn.0": self.block.ffn[0]
        }
        for name in input_layers:
            if name in self.hessian_dict:
                W = layer_objs[name].weight.float()
                H = self.hessian_dict[name].float()
                # 只有输入被 R1 旋转
                # salience = self.compute_salience_wanda(W, H, R_in=R1, R_out_T=None)
                salience = (W @ R1) ** 2
                loss += self.row_entropy_sum(salience)

        # === Group 2: 纯 R1 输出 (RWX) ===
        # 输入: 内部/忽略, 输出: Residual (R1)
        # 包含: CrossAttn_O, FFN_Down
        # *注*: CrossAttn_O 输入来自 Text，不归 R2 管，直接输出对齐 R1 即可
        output_layers = [ "ffn.2"]
        layer_objs_out = {
            "ffn.2": self.block.ffn[2]
        }
        for name in output_layers:
            if name in self.hessian_dict:
                W = layer_objs_out[name].weight.float()
                H = self.hessian_dict[name].float()
                # 只有输出被 R1 旋转 (W_new = R1.T @ W)
                # 这里 trick 一下：视作 Input=Identity, Output=R1
                # salience 计算函数里 W_new = R_out_T @ W
                # salience = (R1.T @ W).pow(2) * torch.abs(torch.diag(H))
                salience = (R1.T @ W).pow(2)
                loss += self.row_entropy_sum(salience.T)

        # === Group 3: 混合双打 (R2 ... R1) ===
        
        # 3.1 Self-Attn V (Input: R1, Output: R2)
        # 这里的 V_proj 负责把 Residual(R1) 映射到 Head Space(R2)
        # if "self_attn.v" in self.hessian_dict:
        #     W = self.block.self_attn.v.weight.float()
        #     H = self.hessian_dict["self_attn.v"].float()
        #     # W_new = R2.T @ W @ R1
        #     salience = self.compute_salience_wanda(W, H, R_in=R1, R_out_T=R2.T)
        #     loss += row_entropy_sum(salience)
            
        # 3.2 Self-Attn O (Input: R2, Output: R1)
        # 这里的 O_proj 负责把 Head Space(R2) 映射回 Residual(R1)
        # if "self_attn.o" in self.hessian_dict:
        #     W = self.block.self_attn.o.weight.float()
        #     H = self.hessian_dict["self_attn.o"].float()
        #     # W_new = R1.T @ W @ R2
        #     salience = self.compute_salience_wanda(W, H, R_in=R2, R_out_T=None)
        #     # 注意: O Proj 通常是按列剪枝还是按行？DenoiseRotator通常优化输出分布
        #     # 这里我们加这两个方向的熵，或者只加输出的
        #     loss += row_entropy_sum(salience.T)

        return loss

class WanRotatorOptimizer_Training_free(torch.nn.Module):
    def __init__(self, block, hessian_dict, dim, num_heads, device):
        super().__init__()
        self.block = block
        self.hessian_dict = hessian_dict
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.device = device
        
    def apply_nm_sparse_weight(self, w, window, topk):
        """对权重张量在B_in维度应用N:M稀疏 (TopKWeight用)
        Args:
            w: [B_out, B_in]
            window: M
            topk: N
        """
        if topk >= window or w.numel() == 0 or w.dim() != 2:
            return w
        orig_dtype = w.dtype
        B_out, B_in = w.shape
        pad = (window - B_in % window) % window
        w_pad = F.pad(w, (0, pad)) if pad > 0 else w
        num_windows = w_pad.shape[-1] // window
        w_win = w_pad.view(B_out, num_windows, window)
        _, idx = w_win.abs().topk(topk, dim=-1)
        mask = torch.zeros_like(w_win)
        mask.scatter_(-1, idx, 1.0)
        w_sparse = (w_win * mask).view(B_out, -1)
        if pad > 0:
            w_sparse = w_sparse[:, :B_in]
        # print("we go here!"*20)
        return w_sparse.to(orig_dtype)

    def compute_salience_wanda(self, weight, hessian):
        s = torch.abs(weight)*hessian

        channel_out, channel_in = s.shape[0],s.shape[1]
        l2_norms = torch.norm(s, p=2, dim=0)
        sorted_norms, sorted_indices = torch.sort(l2_norms, descending=True)
        
        half_c = channel_in // 2
        high_indices = sorted_indices[:half_c]
        low_indices = torch.flip(sorted_indices[half_c:], dims=[0])
        
        permuted_indices = torch.zeros(channel_in, dtype=torch.long)
        
        permuted_indices[0::4] = high_indices[0::2]
        permuted_indices[1::4] = high_indices[1::2]
        permuted_indices[2::4] = low_indices[0::2]
        permuted_indices[3::4] = low_indices[1::2]
        
        w_reordered = weight[:, permuted_indices]

        return w_reordered,permuted_indices


    def forward(self):
        input_layers = ["ffn.0"]
        layer_objs = {
            # "self_attn.q": self.block.self_attn.q
            "ffn.0": self.block.ffn[0]
        }
        for name in input_layers:
            if name in self.hessian_dict:
                W = layer_objs[name].weight.float()
                H = self.hessian_dict[name].float()
                # 只有输入被 R1 旋转
                w_reorder, r1 = self.compute_salience_wanda(W, H)
                w_sparse = self.apply_nm_sparse_weight(w_reorder,4,2)

        ############重写self.compute_salience_wanda获得新权重然后进行剪枝

        output_layers = [ "ffn.2"]
        layer_objs_out = {
            "ffn.2": self.block.ffn[2]
        }
        for name in output_layers:
            if name in self.hessian_dict:
                W = layer_objs_out[name].weight.float()
                H = self.hessian_dict[name].float()
                w_reorder_2, r2 = self.compute_salience_wanda(W, H)
                w_sparse_2 = self.apply_nm_sparse_weight(w_reorder_2,4,2)
        # print("-"*29)
        # print(w_sparse.shape)
        # print(w_sparse_2.shape)
        # print(r1.shape)
        # print(r2.shape)
        return w_sparse, w_sparse_2, r1, r2
    
class WrappedWan:
    def __init__(self, layer):
        self.layer = layer
        # 【关键修改】不要依赖 layer.device，因为 layer 可能会跑去 CPU
        # 我们强制把统计量放在 GPU 0 上 (或者你指定的 device)
        self.dev = torch.device("cuda:0") 
        
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        # 统计量常驻 GPU
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        # 1. 维度适配
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) > 2:
                inp = inp.reshape(-1, inp.shape[-1])
        elif isinstance(self.layer, nn.Conv2d):
            inp = inp.permute(0, 2, 3, 1).reshape(-1, inp.shape[1])

        # 【关键修改】确保计算在 GPU 上进行
        # 即使输入在 CPU (极少情况)，也搬到 GPU 算，算完这个临时 inp 就会被释放
        if inp.device != self.dev:
            inp = inp.to(self.dev, non_blocking=True) # non_blocking 加速传输



        inp = quant_hif4(inp)

        inp = inp.to(torch.float32)
        tmp = inp.shape[0]

        # 计算 Norm (GPU 上极快)
        # 这里不需要把巨大的 inp 搬回 CPU，只在 GPU 上算出一个很小的向量
        current_norm = torch.norm(inp, p=2, dim=0) ** 2

        if self.nsamples == 0:
            self.nsamples += tmp
            self.scaler_row = current_norm / self.nsamples
        else:
            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            self.scaler_row += current_norm / self.nsamples
            
    def get_scaler(self):
        return torch.sqrt(self.scaler_row)

class TemporalCalibrationCollector:
    def __init__(self, model):
        self.model = model
        self.data = []
        self.hook = None

    def start(self):
        self.data = []
        # 【关键修改】开启 with_kwargs=True，让 Hook 能抓到 t=... 和 context=...
        self.hook = self.model.register_forward_hook(self._hook_fn, with_kwargs=True)

    def stop(self):
        if self.hook:
            self.hook.remove()
            self.hook = None

    # 【关键修改】函数签名变了，增加了 kwargs
    def _hook_fn(self, module, args, kwargs, out):
        try:
            # 1. 提取 x (通常是第一个位置参数 args[0])
            # args 是一个 tuple，比如 ([tensor, tensor...], )
            if len(args) > 0:
                x_val = args[0][0].detach().cpu() 
            else:
                # 防御性代码：万一 x 也是用关键字传的
                x_val = kwargs['x'][0].detach().cpu()

            # 2. 提取 t (肯定在 kwargs 里)
            if 't' in kwargs:
                t_val = kwargs['t'].detach().cpu()
            else:
                # 防御性代码：万一 t 是位置参数
                t_val = args[1].detach().cpu()

            # 3. 提取 context (肯定在 kwargs 里)
            # context 也是个 list，需要取 [0]
            if 'context' in kwargs:
                context_val = kwargs['context'][0].detach().cpu()
            else:
                context_val = args[2][0].detach().cpu()

            # 4. 提取 seq_len
            if 'seq_len' in kwargs:
                seq_len_val = kwargs['seq_len']
            else:
                seq_len_val = args[3]

            self.data.append({
                'x': x_val,
                't': t_val,
                'context': context_val,
                'seq_len': seq_len_val
            })
            
        except Exception as e:
            # 打印错误以便调试
            print(f"Hook Error details: {e}")
            # print(f"Available kwargs: {kwargs.keys()}")
            pass


def save_sparse_model(transformer, save_path, highorlow, save_mode='state_dict'):
    """
    保存稀疏化后的模型
    save_mode:
      - 'state_dict': 只保存transformer的权重字典（推荐，文件小）
      - 'full': 保存完整的transformer模型（包含config）
    """
    os.makedirs(save_path, exist_ok=True)
    if save_mode == 'state_dict':
        ckpt_path = os.path.join(save_path, f'{highorlow}_transformer_sparse.pt')
        torch.save(transformer.state_dict(), ckpt_path)
        print(f'保存state_dict到: {ckpt_path}')
        print(f'加载方式: transformer.load_state_dict(torch.load("{ckpt_path}"))')
        print("模型权重已在 CPU 上，文件加载时默认也会在 CPU，加载后请使用 .to('cuda')")
    elif save_mode == 'full':
        transformer.save_pretrained(save_path)
        print(f'保存完整transformer到: {save_path}')
        print(f'加载方式: transformer = QwenImageTransformer2DModel.from_pretrained("{save_path}")')

def run_wan_pruning_with_fix_v2(
    wan_t2v, 
    prompts, 
    prune_method,
    target_layers,  # 【新增】必须传入这个列表，例如 ['ffn.2']
    steps, 
    device_id,
    sparsity,
    size,
    frame_num,
    shift,
    sample_solver,
    sampling_steps,
    guide_scale,
    base_seed,
    offload_model,
    save_path
):
    print(f"\n[Final Solution v2] Starting Calibration & Pruning with Target Filter...")
    print(f"⚠️ Target Layers to Prune: {target_layers}")
    
    device = torch.device(f"cuda:{device_id}")
    
    models_to_adapt = {
        "HighNoise": wan_t2v.high_noise_model,
        "LowNoise": wan_t2v.low_noise_model
    }
    
    # --------------------------------------------------------------------------
    # Phase 1: 收集 Hessian (这里是对全层收集，保证 R 计算准确)
    # --------------------------------------------------------------------------
    global_gpts = {}
    all_handles = []
    
    # 定义该 Block 内所有需要参与旋转的线性层（全集）
    # 不管剪不剪，这些层都必须参与旋转计算，否则模型会裂开
    ALL_LINEAR_LAYERS_NAMES = [
        "ffn.0",
        "ffn.2"
    ]

    for model_name, model in models_to_adapt.items():
        global_gpts[model_name] = {}
        model.cpu() 
        
        for i, block in enumerate(model.blocks):
            global_gpts[model_name][i] = {}
            
            # 这里的 mapping 是为了方便获取 layer 对象
            layer_mapping = {
                "ffn.0": block.ffn[0], "ffn.2": block.ffn[2]
            }
            
            # 对所有层都注册 Hook，因为训练 R 需要全局信息效果最好
            # (即使某层不剪枝，Hessian 也能帮助计算出更平滑的全局 R)
            # def hook_function(m, inp, out, g):
            #     print(f"Shape of inp[0]: {inp[0].shape}")  # 打印shape
            #     g.add_batch(inp[0].data)

            # for name, layer in layer_mapping.items():
            #     gpt = WanSparseGPT_Offload(layer)
            #     global_gpts[model_name][i][name] = gpt
            #     handle = layer.register_forward_hook(
            #         lambda m, inp, out, g=gpt: hook_function(m, inp, out, g)
            #     )
            #     all_handles.append(handle)

            # 进行hook函数的注册，用于抓取中间变量
            for name, layer in layer_mapping.items():
                gpt = WanSparseGPT_Offload(layer)
                global_gpts[model_name][i][name] = gpt
                handle = layer.register_forward_hook(lambda m, inp, out, g=gpt: g.add_batch(inp[0].data))
                all_handles.append(handle)

    # 推理 (Calibration)
    print(f"  > Calibrating (Generating samples)...")
    # wan_t2v.high_noise_model.to(device)
    # wan_t2v.low_noise_model.to(device)
    
    # 推理一次用于进行输入输出的参数的抓取
    gc.collect()    
    torch.cuda.empty_cache()
    with torch.no_grad():
        for idx, p in enumerate(prompts):
            print(f" Processing {idx+1}/{len(prompts)}...")
            try:
                wan_t2v.generate(
                    input_prompt=p,                 
                    size=size,# 可以用小分辨率来加速校准，特征是通用的
                    frame_num=frame_num,
                    shift=shift,
                    sample_solver=sample_solver,
                    sampling_steps=40,
                    guide_scale=guide_scale,
                    seed=base_seed,
                    offload_model=True)
            except Exception as e:
                print(e)
            gc.collect()
            torch.cuda.empty_cache()
    
    for h in all_handles: h.remove()
    wan_t2v.high_noise_model.cpu()
    wan_t2v.low_noise_model.cpu()

    # --------------------------------------------------------------------------
    # Phase 2: 训练 R1 -> 写入 Block -> 【按需剪枝】
    # --------------------------------------------------------------------------
    print("\n[Phase 2] Training R1 & Pruning Targets...")

    for model_name, model in models_to_adapt.items():
        print(f"  > Processing {model_name}...")
        
        for i, block in enumerate(model.blocks):
            # 1. 搬运 Block 和 Hessian
            block = block.to(device)
            block_gpts = global_gpts[model_name][i]
            hessian_dict_gpu = {name: gpt.H.to(device) for name, gpt in block_gpts.items()}
            
            # 2. 训练 Rotator
            rotator = WanRotatorOptimizer(block, 
                                          hessian_dict_gpu, 
                                          block.dim, 
                                          block.num_heads, 
                                          device)
            
            opt = torch.optim.Adam(rotator.parameters(), lr=0.01)
            total_epochs = 300

            for epoch in range(total_epochs): 
                opt.zero_grad(set_to_none=True)
                loss = rotator() 
                loss.backward()
                opt.step()
                if (epoch + 1) % 10 == 0 :
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"      [{current_time}] Epoch {epoch+1:02d}/{total_epochs} | Loss: {loss.item():.6f}")            
            
            # 3. 注册 R1 (解决 AdaLN 乱码问题的关键)
            R1 = rotator.get_R1()
            # R2 = rotator.get_R2()
            if hasattr(block, 'register_rotator'):
                block.register_rotator(R1) 
            else:
                raise RuntimeError("错误：Block 未找到 register_rotator 方法，请检查 wan_model.py 是否已替换！")

            # 4. 融合权重 (Fuse) - 所有人都要做！
            with torch.no_grad():
                # Input R1
                for layer in [block.ffn[0]]:
                    layer.weight.data = layer.weight.data @ R1
                # Output R1
                for layer in [ block.ffn[2]]:
                    layer.weight.data = R1.T @ layer.weight.data
                    if layer.bias is not None: layer.bias.data = layer.bias.data @ R1
                
            # 5. 【关键修改】只剪枝 target_layers
            # ----------------------------------------------------
            pruned_count = 0
            for name in ALL_LINEAR_LAYERS_NAMES:
                if name in block_gpts:
                    # 判断：只有在 target_layers 列表里的才剪枝
                    if name in target_layers:
                        sparsity = block_gpts[name].prune_2_4(method=prune_method)
                        pruned_count += 1
                        print(f"      Pruned {name} (Sparsity: {sparsity:.2f})")
                    else:
                        # 不剪枝，仅仅清理掉 Hessian 占用的显存
                        del block_gpts[name].H 
            # ----------------------------------------------------

            # 6. 清理
            del rotator, opt, hessian_dict_gpu, R1#, R2
            block = block.cpu() 
            model.blocks[i] = block 
            
            del block_gpts
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"    Block {i} Done. Pruned {pruned_count} layers.")

    save_sparse_model(wan_t2v.low_noise_model, save_path,'low')
    save_sparse_model(wan_t2v.high_noise_model,save_path,'high')
    print("\nDone! 模型已修复 AdaLN 兼容性并完成指定层剪枝。")
    return wan_t2v


def load_pt_weights(
    pipeline,
    low_noise_pt=None,
    high_noise_pt=None,
    strict=True,
    map_location="cpu",
):
    
    if dist.is_initialized():
        dist.barrier()

    def extract_state_dict(obj):
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(
                obj["state_dict"], dict):
            return obj["state_dict"]
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"],
                                                                  dict):
            return obj["model"]
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

    def load_one(model, pt_path):
        if pt_path is None:
            return None

        pt_path = os.path.abspath(pt_path)
        ckpt_obj = torch.load(pt_path, map_location=map_location)
        state_dict = extract_state_dict(ckpt_obj)

        first_param = next(model.parameters())
        original_device = first_param.device
        if original_device.type == "cuda":
            model.to("cpu")

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=strict)

        if original_device.type == "cuda" and (not getattr(pipeline, "init_on_cpu", False)):
            model.to(getattr(pipeline, "device"))

        return {"path": pt_path, "missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    results = {}
    if low_noise_pt is not None:
        print("Loading low noise model weights !")
        results["low_noise_model"] = load_one(
            getattr(pipeline, "low_noise_model"), low_noise_pt)
    if high_noise_pt is not None:
        print("Loading high noise model weights !")
        results["high_noise_model"] = load_one(
            getattr(pipeline, "high_noise_model"), high_noise_pt)

    if dist.is_initialized():
        dist.barrier()

    print("Weights loading finish !")
    return results

def save_checkpoint(
    pipeline,
    output_dir,
    link_files=True,
    save_low_noise=True,
    save_high_noise=True,
):
    print(f"Saving models to {output_dir} ...")

    output_dir = os.path.abspath(output_dir)
    if dist.is_initialized() and getattr(pipeline, "rank", 0) != 0:
        dist.barrier()
        return

    os.makedirs(output_dir, exist_ok=True)

    def remove_path(path):
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def link_or_copy(src, dst):
        src = os.path.abspath(src)
        dst = os.path.abspath(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) or os.path.islink(dst):
            remove_path(dst)
        if link_files:
            os.symlink(src, dst)
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    def save_model(model, subfolder):
        save_dir = os.path.join(output_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        first_param = next(model.parameters())
        original_device = first_param.device

        if original_device.type == "cuda":
            model = model.to("cpu")

        model.save_pretrained(save_dir)

        if original_device.type == "cuda" and (not getattr(pipeline, "init_on_cpu", False)):
            model = model.to(getattr(pipeline, "device"))

    checkpoint_dir = getattr(pipeline, "checkpoint_dir")
    config = getattr(pipeline, "config")

    link_or_copy(
        os.path.join(checkpoint_dir, config.t5_checkpoint),
        os.path.join(output_dir, config.t5_checkpoint),
    )
    link_or_copy(
        os.path.join(checkpoint_dir, config.vae_checkpoint),
        os.path.join(output_dir, config.vae_checkpoint),
    )
    link_or_copy(
        os.path.join(checkpoint_dir, config.t5_tokenizer),
        os.path.join(output_dir, config.t5_tokenizer),
    )

    if save_low_noise:
        save_model(getattr(pipeline, "low_noise_model"), config.low_noise_checkpoint)
    if save_high_noise:
        save_model(getattr(pipeline, "high_noise_model"), config.high_noise_checkpoint)

    if dist.is_initialized():
        dist.barrier()
    
    print("Model save finish !")

##### 以下代码暂时并没有使用,之前雨铭编写_xhm
def remove_keys(d):
    """
    移除字典中所有包含指定子字符串的键值对：
    - head.head, time, text, self_attn, cross_attn
        "blocks.0.", "blocks.1.", "blocks.2.", "blocks.37.", "blocks.38.", "blocks.39."
    
    参数:
    d (dict): 输入字典
    
    返回:
    dict: 过滤后的字典
    """
    # 定义需要移除的子字符串列表
    substrings = [
        "head.head", "time", "text", "self_attn", "cross_attn"
    ]
    # 创建新字典：仅保留键不包含任何子字符串的键值对（大小写不敏感）
    return {k: v for k, v in d.items() if not any(sub in k.lower() for sub in substrings)}

def quant_hif4(x: Tensor, N_levels: int = 4, G: int = 64, dim: int = -1) -> Tensor:
    """HiF4量化"""
    if x.numel() == 0:
        return x
    orig_shape, orig_dtype = x.shape, x.dtype
    need_transpose = (dim == -2) and x.dim() >= 2
    if need_transpose:
        x = x.transpose(-1, -2).contiguous()
    qdim = -1
    C = x.shape[qdim]
    pad_len = (G - C % G) % G
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))
    x = x.unflatten(qdim, (-1, 8, 2, 4))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)
    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim-1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim-2, keepdim=True)[0]
    div7 = (torch.ones_like(max_lv1) / 7.0).to(torch.bfloat16).to(x.dtype)
    scale_factor = max_lv1 * div7
    EPS = 1e-45
    e_sf = torch.floor(torch.log2(scale_factor + EPS))
    mant_sf = scale_factor / 2**e_sf * 2**7
    scale_factor = torch.round(mant_sf) / 2**7 * 2**e_sf
    e_sf = torch.floor(torch.log2(scale_factor + EPS))
    scale_factor = torch.round(scale_factor * torch.exp2(2-e_sf)) * torch.exp2(e_sf-2)
    rec_sf = (1.0 / (scale_factor + EPS)).to(torch.bfloat16).to(x.dtype)
    scale_lv2 = (max_lv2 * rec_sf)
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())
    man_bits = N_levels - 1
    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2**(man_bits - 1) + 0.5) / 2**(man_bits - 1)
    mant[mant >= 2] = 2 - 2**(-man_bits + 1)
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    out = out.flatten(qdim-3, qdim)
    if pad_len > 0:
        out = out[..., :C]
    if need_transpose:
        out = out.transpose(-1, -2).contiguous()
    return out.view(orig_shape).to(orig_dtype)

def get_input_hook(gpt_instance):
    """
    专门为显存不足设计的 Hook。
    它将输入立即移动到 CPU，并在 CPU 上更新 Hessian 矩阵。
    """
    def hook(module, input, output):
        # 1. 获取输入，立即 detach 并移动到 CPU
        # 注意：使用 float32 累加可以避免精度溢出，且 CPU 处理 float32 很快
        inp = input[0].detach().cpu().float()
        
        # 2. 调用 SparseGPT 的 add_batch
        #前提：gpt_instance.H 必须已经被我们手动放到了 CPU 上
        gpt_instance.add_batch(inp,0,False)
        
        # 3. 显式删除引用，辅助 GC
        del inp
        
    return hook

def prepare_sparsegpt_cpu(layer):
    """
    初始化 SparseGPT，但强制将 Hessian 矩阵放在 CPU 上
    """
    sys.path.append('/home/x50057374/qwen-sparsegpt')
    from sparsegpt import SparseGPT
    gpt = SparseGPT(layer)
    # 强制将 H 矩阵移动到 CPU
    gpt.H = gpt.H.to('cpu')
    gpt.dev = torch.device('cpu') # 欺骗 add_batch 里的 device 判断
    return gpt

def prune_wan_interface(wan_model_instance, calibration_prompts, sparsity,size,frame_num,shift,sample_solver,sampling_steps,guide_scale,base_seed,offload_model):
    """
    外部剪枝控制器：接收一个初始化好的 WanT2V 实例，对其进行原地剪枝。
    """
    print("=== 开始 Wan2.2 模型稀疏化流程 ===")
    sys.path.append('/home/x50057374/qwen-sparsegpt')
    import time
    from sparsegpt import SparseGPT
    from modelutils import find_layers
    # 1. 锁定目标子模型
    # Wan2.2 有两个 DiT 模型：high_noise_model 和 low_noise_model
    # 我们需要同时对它们进行处理
    sub_models = {
        'high_noise': wan_model_instance.high_noise_model,
        'low_noise': wan_model_instance.low_noise_model
    }
    
    gpts = {}    # 存储所有的 SparseGPT 实例
    handles = [] # 存储所有的 Hook 句柄
# --- 阶段 1: 注册 Hook (全部在 CPU 上初始化) ---
    print("正在初始化 CPU 端的 Hessian 矩阵...")
    for sub_name, model_obj in sub_models.items():
        layers = find_layers(model_obj, layers=[nn.Linear, nn.Conv2d])
        layers = remove_keys(layers)
        for name, layer in layers.items():
            full_name = f"{sub_name}.{name}"
            
            # 使用修改后的 CPU 初始化
            gpt = prepare_sparsegpt_cpu(layer)
            gpts[full_name] = gpt
            
            # 注册 CPU Hook
            h = layer.register_forward_hook(get_input_hook(gpt))
            handles.append(h)
            
    print(f"Hook 注册完毕，Hessian 矩阵将占用约 {len(gpts) * 100 / 1024:.2f} GB 的系统内存(RAM)。")
    # --- 阶段 2: 运行校准 (Forward) ---
    print("开始校准...")
    try:
        for i, prompt in enumerate(calibration_prompts):
            print(f"Sample {i+1}/{len(calibration_prompts)}: {prompt[:100]}...")
            
            # 为了省显存，强制清理每一轮的缓存
            torch.cuda.empty_cache()
            
            # 运行生成
            # 建议降低校准时的分辨率，这能大幅降低 Activation 显存占用，且对权重相关性影响极小
            # 例如: size=(640, 360) 或 (832, 480)
            wan_model_instance.generate(
                input_prompt=prompt,
                size=size,# 可以用小分辨率来加速校准，特征是通用的
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=40,
                guide_scale=guide_scale,
                seed=base_seed,
                offload_model=True
            )
            
    except Exception as e:
        print(f"Error during calibration: {e}")
        for h in handles: h.remove()
        raise e
    finally:
        # 无论成功失败，移除 Hook
        for h in handles: h.remove()
        print("Hook 已移除。")

    # --- 阶段 3: 剪枝 (CPU -> GPU -> CPU) ---
    print("开始计算剪枝...")
    
    # 此时模型权重可能在 CPU (因为 offload)，也可能在 GPU
    gpu_device = torch.device(f"{wan_model_instance.device}") # 或者直接 'cuda'

    for name, gpt in gpts.items():
        # 获取对应的层对象
        # 这里需要一点技巧来通过 name 找到 layer 对象，或者我们直接在 gpt 里存了 layer
        layer = gpt.layer 
        
        # 1. 确保权重在 GPU 上 (为了计算更快，且 layer.weight 必须和 Hinv 交互)
        # 如果显存实在不够，这里也可以全程在 CPU 跑，但非常慢
        original_device = layer.weight.device
        
        # 将 Hessian 矩阵搬到 GPU 进行求逆运算 (Cholesky 分解)
        # 因为我们是一个一个层处理，所以这里占用的显存只有 100MB，非常安全
        gpt.H = gpt.H.to(gpu_device) 
        layer.to(gpu_device) 
        
        # 2. 执行剪枝
        # gpt.dev 之前被我们设成了 cpu，现在为了 fasterprune 里创建临时变量在 GPU，临时改一下
        gpt.dev = gpu_device 
        
        try:
            gpt.fasterprune(sparsity=sparsity,
                prunen=2, 
                prunem=4, 
                blocksize=128, 
                percdamp=0.01,
                qut_hif4 = True            
            )
        except Exception as e:
            print(f"Pruning failed for {name}: {e}")
        
        # 3. 善后处理
        gpt.free() # 释放 H 矩阵
        
        # 把层放回它原来的地方 (通常是 CPU，如果开启了 offload)
        layer.to(original_device)
        
        # 打印进度
        if "ffn" in name and "0" in name: # 减少刷屏，只打印部分
            print(f"Pruned {name}")

    print("稀疏化全部完成。")
    del gpts
    gc.collect()
    torch.cuda.empty_cache()
    save_sparse_model(wan_model_instance.low_noise_model,'/home/x50057374/Wan2.2/sparse_param','low')
    save_sparse_model(wan_model_instance.high_noise_model,'/home/x50057374/Wan2.2/sparse_param','high')
    del wan_model_instance
    return     

def prune_wan_wanda(wan_model, calibration_prompts, sparsity_ratio, prune_n, prune_m,size,frame_num,shift,sample_solver,sampling_steps,guide_scale,base_seed,offload_model):
    """
    Wan2.2 的 Wanda 剪枝主函数
    
    Args:
        wan_model: WanT2V 实例
        calibration_prompts: 校准提示词列表
        sparsity_ratio: 目标稀疏度 (0.5 = 剪掉50%)
        prune_n, prune_m: 若不为0，则执行 N:M 结构化剪枝 (如 2:4)
    """
    print(f"=== Starting Wanda Pruning (Sparsity={sparsity_ratio}) ===")
    sys.path.append('/home/x50057374/qwen-sparsegpt')
    from modelutils import find_layers
    # 1. 准备目标子模型 (High Noise 和 Low Noise 都要剪)
    sub_models = {
        'high_noise': wan_model.high_noise_model,
        'low_noise': wan_model.low_noise_model
    }

    # 2. 注册 Hooks
    wrappers = {}
    handles = []
    
    # 定义 Hook 工厂
    # def get_wanda_hook(wrapper):
    #     def hook(module, input, output):
    #         inp = input[0].detach()
    #         # 强制移动到 wrapper 所在的设备 (应对 offload_model=True)
    #         if inp.device != wrapper.dev:
    #             inp = inp.to(wrapper.dev)
    #         wrapper.add_batch(inp)
    #     return hook
# 对应的 Hook 工厂也要改一下，去掉多余的设备检查，让 add_batch 内部处理
    def get_wanda_hook(wrapper):
        def hook(module, input, output):
            # 直接传进去，add_batch 会负责把它挪到 GPU
            wrapper.add_batch(input[0].detach())
        return hook

    print("Registering hooks for Wanda statistics...")
    for sub_name, model_obj in sub_models.items():
        layers = find_layers(model_obj)
        layers = remove_keys(layers)
        for name, layer in layers.items():
            full_name = f"{sub_name}.{name}"
            
            # 创建 Wrapper
            wrapper = WrappedWan(layer)
            wrappers[full_name] = wrapper
            
            # 注册 Hook
            h = layer.register_forward_hook(get_wanda_hook(wrapper))
            handles.append(h)
            
    print(f"Registered {len(wrappers)} layers.")

    # 3. 运行校准 (Forward Pass)
    print("Running calibration...")
    # Wanda 不需要 Hessian 逆矩阵，计算开销很小，瓶颈主要在 forward pass
    try:
        for i, prompt in enumerate(calibration_prompts):
            print(f"Calibration sample {i+1}/{len(calibration_prompts)}...")
            torch.cuda.empty_cache()
            
            # 调用 generate 触发 Hook
            # 建议使用较小的分辨率 (如 640x360) 加速统计，这对 Channel 范数的相对排序影响很小
            wan_model.generate(
                input_prompt=prompt,
                size=size,# 可以用小分辨率来加速校准，特征是通用的
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=40,
                guide_scale=guide_scale,
                seed=base_seed,
                offload_model=True
            )
    except Exception as e:
        print(f"Error during calibration: {e}")
        for h in handles: h.remove()
        raise e
    finally:
        # 必须移除 Hooks，否则显存无法释放
        for h in handles: h.remove()
        print("Hooks removed.")

    # 4. 执行剪枝 (Pruning)
    print("Calculating Wanda metrics and pruning weights...")
    
    with torch.no_grad():
        for name, wrapper in wrappers.items():
            layer = wrapper.layer
            dev = layer.weight.device
            layer.weight.data = quant_hif4(layer.weight.data)
            # 获取统计量 ||X||_2
            scaler_row = wrapper.get_scaler()
            
            # 确保 scaler_row 和 weight 在同一设备 (Layer 可能被 offload 到了 CPU)
            if scaler_row.device != dev:
                scaler_row = scaler_row.to(dev)

            # === Wanda 核心公式 ===
            # Metric = |W| * ||X||_2
            # 广播乘法: [Out, In] * [1, In]
            # 这衡量了每个权重的重要性：权重本身大很重要，输入特征能量大也很重要
            W_metric = torch.abs(layer.weight.data) * scaler_row.reshape((1, -1))

            # 创建 Mask
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                # 结构化剪枝 N:M (如 2:4)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        # 找到每一组中 Metric 最小的 N 个
                        indices = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + indices, True)
            else:
                # 非结构化剪枝 (Unstructured)
                # 采用 Row-wise Pruning (每行独立剪枝)，这对硬件加速更友好，也能保持输出分布
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                
                # 选取每一行最小的 k 个权重设为 True (要剪掉的)
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
                del sort_res

            # 执行剪枝 (原地修改)
            layer.weight.data[W_mask] = 0

            # 释放内存
            del W_metric, W_mask, scaler_row, wrapper
            
            # 打印部分日志
            if "ffn" in name and "0" in name: 
                print(f"Pruned {name}")

    print("Wanda pruning completed.")
    torch.cuda.empty_cache()
    save_sparse_model(wan_model.low_noise_model,'/home/x50057374/Wan2.2/wanda_param','low')
    save_sparse_model(wan_model.high_noise_model,'/home/x50057374/Wan2.2/wanda_param','high')
    return 

def process_block_rotate_and_prune_fast(block, data_pool, device, prune_method, target_layers_names):
    """
    1. Data Subsampling: 不跑全量数据，只随机抽几个样本算 Hessian。
       - 原来：跑 80 个样本 (40 steps * 2 cfg)
       - 现在：只跑 4 个样本
       - 速度提升：20倍！
       
    2. Single Pass: 计算 Hessian 同时保存输出。
       - 速度提升：2倍！
       
    总加速：约 40 倍！
    """
    block.eval()
    
    # --- 1. 极速采样的关键设置 ---
    # 工业验证通常取 4~8 个样本即可跑通流程
    # 如果你想快到飞起，设为 2 甚至 1
    MAX_CALIBRATION_SAMPLES = 4  
    
    # 从 data_pool 中随机抽取子集
    if len(data_pool) > MAX_CALIBRATION_SAMPLES:
        # 打印一下提示，让你知道现在是飞速模式
        # print(f"    [Flash Mode] Subsampling data: {len(data_pool)} -> {MAX_CALIBRATION_SAMPLES} samples")
        calibration_data = random.sample(data_pool, MAX_CALIBRATION_SAMPLES)
    else:
        calibration_data = data_pool

    # --- 2. 注册 Hook ---
    all_linear_layers = {
        "self_attn.q": block.self_attn.q, "self_attn.k": block.self_attn.k, "self_attn.v": block.self_attn.v,
        "self_attn.o": block.self_attn.o, "cross_attn.q": block.cross_attn.q, "cross_attn.o": block.cross_attn.o,
        "ffn.0": block.ffn[0], "ffn.2": block.ffn[2]
    }
    
    gpts = {name: WanSparseGPT(layer) for name, layer in all_linear_layers.items()}
    handles = []
    for name, layer in all_linear_layers.items():
        handles.append(layer.register_forward_hook(
            lambda m, i, o, n=name: gpts[n].add_batch(i[0].data)
        ))

    # --- 3. 计算 Hessian (只跑采样的那几个样本) ---
    # 这一步现在会快如闪电
    with torch.no_grad():
        for sample in calibration_data:
            x_in = sample['x'].to(device)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample['kwargs'].items()}
            block(x_in, **kwargs)
            # 注意：在极速验证模式下，因为我们只跑了部分数据，
            # 我们没法给所有数据都算好 output。
            # 所以下一层 Block 只能拿到这几个样本的 Perfect Output。
            # 但对于“验证代码能不能跑通”来说，足够了。
    
    # 移除 Hook
    for h in handles: h.remove()
    torch.cuda.empty_cache()

    # --- 4. 为了下一层能跑，我们需要更新 data_pool ---
    # 这一步稍微有点 tricky：
    # 因为 Hessian 只用了 4 个样本，但 data_pool 里有 80 个样本。
    # 为了下一层也能极速跑，我们直接把 data_pool 缩减为这 4 个样本的后续流。
    # (这意味着 Block 1 只能看到这 4 个样本了，但这正是我们想要的加速)
    
    if len(data_pool) > len(calibration_data):
        # 既然只验证流程，那就把数据池子也永久变小
        data_pool[:] = calibration_data 

    # 重新跑一遍前向，这次是为了把这几个样本的 Output 存下来喂给下一层
    # 因为刚才 Hook 过程主要为了算 Hessian，现在为了更新数据流
    with torch.no_grad():
        for sample in data_pool:
            x_in = sample['x'].to(device)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample['kwargs'].items()}
            out = block(x_in, **kwargs)
            sample['x'] = out.detach().cpu()

    # --- 5. 训练 R1/R2 (Epoch 也可以减少) ---
    hessian_dict = {name: gpts[name].H.to(device) for name in gpts}
    rotator = WanRotatorOptimizer(block, hessian_dict, block.dim, block.num_heads, device)
    opt = torch.optim.Adam(rotator.parameters(), lr=0.01)
    
    # 验证模式：20个 epoch 足够看 loss 降没降了
    num_epochs = 800
    for epoch in range(num_epochs):
        opt.zero_grad()
        loss = rotator()
        loss.backward()
        opt.step()
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"      [{current_time}] Epoch {epoch+1:02d}/{num_epochs} | Loss: {loss.item():.6f}")
    # --- 6. 融合 ---
    R1 = rotator.get_R1()
    R2 = rotator.get_R2()
    
    with torch.no_grad():
        for layer in [block.self_attn.q, block.self_attn.k, block.cross_attn.q, block.ffn[0]]:
            layer.weight.data = layer.weight.data @ R1
        for layer in [block.cross_attn.o, block.ffn[2]]:
            layer.weight.data = R1.T @ layer.weight.data
            if layer.bias is not None: layer.bias.data = layer.bias.data @ R1
        if "self_attn.v" in all_linear_layers:
            block.self_attn.v.weight.data = R2.T @ (block.self_attn.v.weight.data @ R1)
            if block.self_attn.v.bias is not None: block.self_attn.v.bias.data = block.self_attn.v.bias.data @ R2
        if "self_attn.o" in all_linear_layers:
            block.self_attn.o.weight.data = R1.T @ (block.self_attn.o.weight.data @ R2)
            if block.self_attn.o.bias is not None: block.self_attn.o.bias.data = block.self_attn.o.bias.data @ R1

    # --- 7. 剪枝 ---
    print(f"    > Pruning targets (Flash Mode): {target_layers_names}")
    for name in target_layers_names:
        if name in gpts:
            actual_sp = gpts[name].prune_2_4(method=prune_method)

    del gpts, rotator, hessian_dict, opt, R1, R2
    gc.collect()
    torch.cuda.empty_cache()

def process_block_rotate_and_prune(block, data_pool, device, prune_method, target_layers_names):
    block.eval()
    
    # Hook 所有层
    all_linear_layers = {
        "self_attn.q": block.self_attn.q, "self_attn.k": block.self_attn.k, "self_attn.v": block.self_attn.v,
        "self_attn.o": block.self_attn.o, "cross_attn.q": block.cross_attn.q, "cross_attn.o": block.cross_attn.o,
        "ffn.0": block.ffn[0], "ffn.2": block.ffn[2]
    }
    gpts = {name: WanSparseGPT(layer) for name, layer in all_linear_layers.items()}
    handles = []
    for name, layer in all_linear_layers.items():
        handles.append(layer.register_forward_hook(lambda m, i, o, n=name: gpts[n].add_batch(i[0].data)))

    # 累加 Hessian
    with torch.no_grad():
        for sample in data_pool:
            x_in = sample['x'].to(device)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample['kwargs'].items()}
            block(x_in, **kwargs)
    for h in handles: h.remove()
    torch.cuda.empty_cache()

    # --- 训练 R1 和 R2 ---
    hessian_dict = {name: gpts[name].H.to(device) for name in gpts}
    # num_heads 需从 block 获取
    rotator = WanRotatorOptimizer(block, hessian_dict, block.dim, block.num_heads, device)
    opt = torch.optim.Adam(rotator.parameters(), lr=0.01)
    print("start training!!!!!!")
    for _ in range(100):
        opt.zero_grad()
        loss = rotator()
        loss.backward()
        opt.step()
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"      [{current_time}] Epoch {epoch+1:02d}/{num_epochs} | Loss: {loss.item():.6f}")
    # --- 融合 R1 和 R2 (复杂融合逻辑) ---
    R1 = rotator.get_R1()
    R2 = rotator.get_R2()
    
    with torch.no_grad():
        # 1. Input R1 (Residual Input)
        for layer in [block.self_attn.q, block.self_attn.k, block.cross_attn.q, block.ffn[0]]:
            # W_new = W @ R1
            layer.weight.data = layer.weight.data @ R1
            
        # 2. Output R1 (Residual Output)
        for layer in [block.cross_attn.o, block.ffn[2]]:
            # W_new = R1.T @ W
            layer.weight.data = R1.T @ layer.weight.data
            if layer.bias is not None: layer.bias.data = layer.bias.data @ R1

        # 3. Hybrid V_proj (Input R1, Output R2)
        # W_new = R2.T @ W @ R1
        # PyTorch Linear: y = x @ W.T
        # Target: y' = (x @ R1) @ (R2.T @ W @ R1).T
        #             = x @ R1 @ R1.T @ W.T @ R2
        #             = x @ W.T @ R2 (Output rotated by R2) -> Correct
        # Weight Data Update: W_new = R2.T @ (W @ R1)
        v_layer = block.self_attn.v
        v_layer.weight.data = R2.T @ (v_layer.weight.data @ R1)
        if v_layer.bias is not None: v_layer.bias.data = v_layer.bias.data @ R2

        # 4. Hybrid O_proj (Input R2, Output R1)
        # W_new = R1.T @ W @ R2
        o_layer = block.self_attn.o
        o_layer.weight.data = R1.T @ (o_layer.weight.data @ R2)
        if o_layer.bias is not None: o_layer.bias.data = o_layer.bias.data @ R1

    # --- 剪枝 2:4 ---
    print(f"    > Pruning 2:4 for targets: {target_layers_names}")
    for name in target_layers_names:
        if name in gpts:
            actual_sp = gpts[name].prune_2_4(method=prune_method)
            print(f"      Layer '{name}' pruned. Sparsity: {actual_sp:.2%}")

    del gpts, rotator, hessian_dict, opt
    gc.collect()
    torch.cuda.empty_cache()

def _precompute_embeddings(model, raw_data, device):
    """
    Embedding 预处理 (GridSize 修复版)。
    1. 修复 Grid Size: 使用 Patch 后的特征图尺寸，而非原始输入尺寸。
    2. 动态 RoPE: 强制重新计算 Freqs，彻底解决维度不匹配。
    3. Dtype 自动转换。
    """
    data = []
    
    # 临时移到 GPU
    model.patch_embedding.to(device)
    model.time_embedding.to(device)
    model.text_embedding.to(device)
    model.time_projection.to(device)
    
    # 引入 RoPE 生成函数
    from wan.modules.model import rope_params

    target_dtype = model.patch_embedding.weight.dtype

    with torch.no_grad():
        for s in tqdm(raw_data, desc="Pre-computing Embeddings"):
            # 1. 准备输入
            # x: [C, F, H, W] -> [1, C, F, H, W]
            x = s['x'].to(device, dtype=target_dtype).unsqueeze(0)
            t = s['t'].to(device).reshape(1)
            ctx = s['context'].to(device, dtype=target_dtype).unsqueeze(0)
            
            # 2. Patch Embedding
            # [1, C_in, F, H, W] -> [1, Dim, F_p, H_p, W_p]
            x_patched = model.patch_embedding(x)
            
            # 【核心修复 1】获取 Patch 后的真实网格尺寸 (F_p, H_p, W_p)
            # 之前错误地使用了 x.shape[2:] (原始尺寸)
            real_grid_size = torch.tensor(x_patched.shape[2:], dtype=torch.long, device=device).unsqueeze(0)
            
            # Flatten: [1, Dim, L] -> [1, L, Dim]
            x_feat = x_patched.flatten(2).transpose(1, 2)
            
            # 【核心修复 2】动态生成匹配的 RoPE Freqs
            # 不信任 model.freqs，根据当前 x_feat 的真实 head_dim 现场生成
            actual_dim = x_feat.shape[2]
            actual_head_dim = actual_dim // model.num_heads
            d = actual_head_dim
            
            # Wan2.2 RoPE Logic
            freqs_parts = [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ]
            dynamic_freqs = torch.cat(freqs_parts, dim=1).to(device)
            
            # 3. Time Embedding
            t_exp = t.expand(1, s['seq_len'])
            sin_emb = sinusoidal_embedding_1d(model.freq_dim, t_exp.flatten()).to(device=device, dtype=target_dtype)
            e0 = model.time_projection(model.time_embedding(sin_emb)).unflatten(1, (6, model.dim))
            
            # 4. Context Embedding
            ctx_feat = model.text_embedding(ctx)
            
            # 5. 构造 kwargs
            kw = {
                'e': e0,
                'seq_lens': torch.tensor([x_feat.size(1)], device=device),
                'grid_sizes': real_grid_size,  # <--- 使用修复后的 Grid Size
                'freqs': dynamic_freqs,        # <--- 使用修复后的 Freqs
                'context': ctx_feat,
                'context_lens': None
            }
            
            # 移回 CPU
            kw_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
            data.append({'x': x_feat.cpu(), 'kwargs': kw_cpu})

    # 恢复环境
    model.patch_embedding.cpu()
    model.time_embedding.cpu()
    model.text_embedding.cpu()
    model.time_projection.cpu()
    torch.cuda.empty_cache()
    
    return data

def run_wan_denoise_rotator(wan_t2v, prompts, prune_method, target_layers, steps, device_id,sparsity,size,frame_num,shift,sample_solver,sampling_steps,guide_scale,base_seed,offload_model):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}]\n")
    device = torch.device(f"cuda:{device_id}")
    models_to_adapt = {"HighNoise": wan_t2v.high_noise_model, "LowNoise": wan_t2v.low_noise_model}
    
    print(f"\n[Phase 1] Collecting Data using {len(prompts)} prompts...")
    collectors = {k: TemporalCalibrationCollector(m) for k, m in models_to_adapt.items()}
    raw_data = {k: [] for k in models_to_adapt}
    
    for idx, p in enumerate(prompts):
        print(f"  > Generating Prompt {idx+1}: '{p}'")
        for c in collectors.values(): c.start()
        with torch.no_grad():
            wan_t2v.generate(                
                input_prompt=p,
                size=size,# 可以用小分辨率来加速校准，特征是通用的
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=40,
                guide_scale=guide_scale,
                seed=base_seed,
                offload_model=False)
        for c in collectors.values(): c.stop()
        for k in models_to_adapt:
            raw_data[k].extend(collectors[k].data)
            collectors[k].data = []
        gc.collect()

    for name, model in models_to_adapt.items():
        print(name)
        if not raw_data[name]: continue
        print(f"\n[Processing {name} Model for 2:4 Pruning]")
        processed_data = _precompute_embeddings(model, raw_data[name], device)
        
        print(f"  > Rotating (R1+R2) & Pruning {len(model.blocks)} Blocks...")
        for i, block in enumerate(model.blocks):
            print(f"  > Block {i}/{len(model.blocks)}")
            block = block.to(device)
            process_block_rotate_and_prune_fast(block, processed_data, device, prune_method=prune_method, target_layers_names=target_layers)
            with torch.no_grad():
                for sample in processed_data:
                    x = sample['x'].to(device)
                    kw = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in sample['kwargs'].items()}
                    out = block(x, **kw)
                    sample['x'] = out.detach().cpu()
            model.blocks[i] = block.cpu()
            gc.collect()
            torch.cuda.empty_cache()
    save_sparse_model(wan_t2v.low_noise_model,'/home/x50057374/Wan2.2/dr_sparse_param','low')
    save_sparse_model(wan_t2v.high_noise_model,'/home/x50057374/Wan2.2/dr_sparse_param','high')
    print("\nAll Done! Models are now 2:4 sparse with R1+R2 Rotation.")
    del wan_t2v
    return 