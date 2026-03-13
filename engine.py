import os
import gc
import datetime
import numpy as np

import torch
import torch.nn as nn

from .pruning_utiles import WanRotatorOptimizer
from .pruning_utiles import WanRotatorOptimizer_Training_free
from .pruning_utiles import WanSparseGPT_Offload
from .pruning_utiles import WanSparseGPT_Offload_Training_free

from .pruning_utiles import save_sparse_model

from .obs import OBS_Wan

class Prune_Engine:
    def __init__(self, 
                 wan_t2v,
                 prompts,
                 models_to_adapt,
                 size,
                 frame_num,
                 shift,
                 sample_solver,
                 sampling_steps,
                 guide_scale,
                 base_seed,
                 target_layers,
                 sparsity_ratio,
                 percdamp,
                 prune_n,
                 prune_m
                 ):

        # model inference args
        self.size=size,# 可以用小分辨率来加速校准，特征是通用的
        self.frame_num=frame_num,
        self.shift=shift,
        self.sample_solver=sample_solver,
        self.sampling_steps=sampling_steps,
        self.guide_scale=guide_scale,
        self.seed=base_seed,

        # prompts to start inference 
        self.prompts = prompts

        # model
        self.wan_t2v = wan_t2v

        # prepare for model pruning
        self.models_to_adapt = models_to_adapt 
        #用于存储需要剪枝的模型， 实际上就是选择是high还是low noise模型

        # self.selected_names = [
        #     "ffn.0",
        #     "ffn.2"
        # ]

        self.target_layers = target_layers

        self.timestep_weight = None
        self.step_info = {"current": 0}

        self.target_pruned_modules = []
        self.pruner_dict = {}
        self.all_hooks = []

        # args for pruning setting
        self.sparsity_ratio = sparsity_ratio
        self.percdamp = percdamp
        self.prune_n = prune_n
        self.prune_m = prune_m

    def create_hook_fn(self,
                       block_idx, 
                       layer_name, 
                       pruner_dict, 
                       timestep_weight):
        
        def hook_fn(module, input, output):
            
            # step = step_info["current"]

            pruner = pruner_dict[(block_idx, layer_name)]
            
            # get the input data
            input_data = input[0].data

            if timestep_weight is not None:
                step = self.step_info["current"]
                current_weight = timestep_weight[step]
            else:
                current_weight = 1.0

            num_samples = input_data.shape[0]
            W_new = current_weight * num_samples
            
            input_data = input_data * np.sqrt(current_weight)
            # call add_batch, pass the weighted input data
            pruner.add_batch(input_data, output.data, W_new)
            #msg = f"Updated Hessian for Block {block_idx}, {layer_name}, Step {step}, Input Shape: {input[0].shape}, Weight: {timestep_weight[step]:.4f}"
            #logger.info(msg)  

        return hook_fn
    
    @torch.no_grad
    def registration(self):

        print("\n[Phase 1] Registrate hook function and inference...")


        # record modules need to be pruned
        for model_name, model in self.models_to_adapt.items():
            # target_pruned_modules = [] #TODO need to be all accseeable

            for i, block in enumerate(model.blocks):
                layer_mapping = {
                    'ffn.0': block.ffn[0],
                    'ffn.2': block.ffn[2]
                }

                for name, layer in layer_mapping.items():
                    self.target_pruned_modules.append((model_name, i, name, layer)) #
        
        num_modules = len(self.target_pruned_modules)
        print(self.target_pruned_modules)
        print(f" need to prune {num_modules} modules !")

        # registrate hook function for inference
        for model_name, block_id, name, layer in self.target_pruned_modules:
            self.pruner_dict[(block_id, name)] = OBS_Wan(layer, args=None)
            hook_fn = self.create_hook_fn(block_id, name, self.pruner_dict, self.timestep_weight)
            self.all_hooks.append(layer.register_forward_hook(hook_fn))

        # start inference
        for idx, p in enumerate(self.prompts):
            print(f" Processing {idx+1}/{len(self.prompts)}...")
            try:
                self.wan_t2v.generate(
                    input_prompt=p,                 
                    size=self.size,# 可以用小分辨率来加速校准，特征是通用的
                    frame_num=self.frame_num,
                    shift=self.shift,
                    sample_solver=self.sample_solver,
                    sampling_steps=self.sampling_steps,
                    guide_scale=self.guide_scale,
                    seed=self.seed,
                    offload_model=True)
            except Exception as e:
                    print(e)

        for h in self.all_hooks: h.remove()
    
    def pruning(self):

        print("\n[Phase 2] Pruning Targets...")
        # registrate hook function for inference
        for model_name, block_id, name, layer in self.target_pruned_modules:
            print(f"Pruning {model_name} Block {block_id}: {name}")
            sparsity = self.sparsity_ratio[block_id] if isinstance(self.sparsity_ratio, list) else self.sparsity_ratio
            self.pruner_dict[(block_id, name)].fasterprune(
                sparsity=sparsity,
                percdamp=self.percdamp,
                prune_n=self.prune_n,
                prune_m=self.prune_m
            )
            self.pruner_dict[(block_id, name)].free()    
    
    def run(self):

        self.registration()
        self.pruning()

        print("\nDone! 模型已完成指定层剪枝。")

class Engine():
    def __init__(self, 
                 wan_t2v,
                 training,
                 epoch,
                 lr,
                 prompts,
                 models_to_adapt,
                 size,
                 frame_num,
                 shift,
                 sample_solver,
                 sampling_steps,
                 guide_scale,
                 base_seed,
                 target_layers,
                 prune_method,
                 device_id,
                 save_path
                 ):
        # base args for pruning
        self.training = training
        if self.training:
            print("Setting for training R matrix")
        else:
            print("Setting for training-free")

        self.rotator = WanRotatorOptimizer if self.training else WanRotatorOptimizer_Training_free

        self.epoch = epoch
        self.lr = lr

        self.save_path = save_path

        # model inference args
        self.size=size,# 可以用小分辨率来加速校准，特征是通用的
        self.frame_num=frame_num,
        self.shift=shift,
        self.sample_solver=sample_solver,
        self.sampling_steps=sampling_steps,
        self.guide_scale=guide_scale,
        self.seed=base_seed,

        # prompts to start inference 
        self.prompts = prompts

        # model
        self.wan_t2v = wan_t2v

        # prepare for model pruning
        self.models_to_adapt = models_to_adapt #用于存储需要剪枝的目标结构字典
        self.selected_names = [
            "ffn.0",
            "ffn.2"
        ]
        self.target_layers = target_layers
        self.prune_method = prune_method
    
        self.global_gpts = {}
        self.all_handles = []

        self.R1 = None

        self.device = torch.device(f"cuda:{device_id}")

    def init_rotator(self,
                     block, 
                     hessian_dict_gpu, 
                     ):
        rotator_init = self.rotator(
                block, 
                hessian_dict_gpu, 
                block.dim, 
                block.num_heads, 
                self.device
            )
        
        return rotator_init

    def registration(self):
        for model_name, model in self.models_to_adapt.items():
            self.global_gpts[model_name] = {}
            model.cpu() 
            
            for i, block in enumerate(model.blocks):
                self.global_gpts[model_name][i] = {}
                
                # 这里的 mapping 是为了方便获取 layer 对象
                layer_mapping = {
                    "ffn.0": block.ffn[0], "ffn.2": block.ffn[2]
                }

                # 进行hook函数的注册，用于抓取中间变量
                for name, layer in layer_mapping.items():
                    gpt = WanSparseGPT_Offload(layer) if self.training else WanSparseGPT_Offload_Training_free(layer)
                    self.global_gpts[model_name][i][name] = gpt
                    handle = layer.register_forward_hook(lambda m, inp, out, g=gpt: g.add_batch(inp[0].data))
                    self.all_handles.append(handle)

        # 推理 (Calibration)
        print(f"  > Calibrating (Generating samples)...")
        # wan_t2v.high_noise_model.to(device)
        # wan_t2v.low_noise_model.to(device)
        
        # 推理一次用于进行输入输出的参数的抓取
        gc.collect()    
        torch.cuda.empty_cache()
        with torch.no_grad():
            for idx, p in enumerate(self.prompts):
                print(f" Processing {idx+1}/{len(self.prompts)}...")
                try:
                    self.wan_t2v.generate(
                        input_prompt=p,                 
                        size=self.size,# 可以用小分辨率来加速校准，特征是通用的
                        frame_num=self.frame_num,
                        shift=self.shift,
                        sample_solver=self.sample_solver,
                        sampling_steps=self.sampling_steps,
                        guide_scale=self.guide_scale,
                        seed=self.seed,
                        offload_model=True)
                except Exception as e:
                    print(e)
                gc.collect()
                torch.cuda.empty_cache()
        
        for h in self.all_handles: h.remove()
        self.wan_t2v.high_noise_model.cpu()
        self.wan_t2v.low_noise_model.cpu()

    def optimize(self, 
                 block, 
                 hessian_dict_gpu,
                 ):
        
        # init ratator
        rotator = self.init_rotator(
            block,
            hessian_dict_gpu
        )
        # self.rotator(
        #         block, 
        #         hessian_dict_gpu, 
        #         block.dim, 
        #         block.num_heads, 
        #         self.device
        #     )
        
        if self.training:
            # 2. 训练 Rotator
            
            opt = torch.optim.Adam(rotator.parameters(), lr=self.lr)

            for epoch in range(self.epoch): 
                opt.zero_grad(set_to_none=True)
                loss = rotator() 
                loss.backward()
                opt.step()
                if (epoch + 1) % 10 == 0 :
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"      [{current_time}] Epoch {epoch+1:02d}/{self.epoch} | Loss: {loss.item():.6f}")            
            # 3. 注册 R1 (解决 AdaLN 乱码问题的关键)
            R1 = rotator.get_R1()

            return R1
        else:
            # rotator = self.rotator
            w_sparse, w_sparse_2, R1, R2 = rotator()
            return w_sparse, w_sparse_2, R1, R2

    @torch.no_grad
    def weight_fusion(self, 
                      block, 
                      optimize_results):
            
        if self.training:
            R1 = optimize_results

            if hasattr(block, 'register_rotator'):
                block.register_rotator(R1) 
            else:
                raise RuntimeError("错误：Block 未找到 register_rotator 方法，请检查 wan_model.py 是否已替换！")

            # 4. 融合权重 (Fuse) - 所有人都要做！
            # Input R1
            for layer in [block.ffn[0]]:
                layer.weight.data = layer.weight.data @ R1
            # Output R1
            for layer in [ block.ffn[2]]:
                layer.weight.data = R1.T @ layer.weight.data
                if layer.bias is not None: layer.bias.data = layer.bias.data @ R1

            return 0
        else:
            w_sparse, w_sparse_2, R1, R2 = optimize_results

            if hasattr(block, 'register_rotator'):
                block.register_rotator(R1) 
                block.register_rotator_2(R2)
            else:
                raise RuntimeError("错误：Block 未找到 register_rotator 方法，请检查 wan_model.py 是否已替换！")
            
            # 针对training free的结构融合不需要做其他的事情
            # 只需要把优化后的结果返回到原始模型即可

            # 4. 融合权重 (Fuse) - 所有人都要做！
            pruned_count = 0
            with torch.no_grad():
                # Input R1
                for layer in [block.ffn[0]]:
                    pruned_count += 1
                    layer.weight.data = w_sparse
                # Output R1
                for layer in [ block.ffn[2]]:
                    pruned_count += 1
                    layer.weight.data = w_sparse_2
            
            return pruned_count

    def pruning(self, 
                block_gpts,
                already_pruned_count):
        
        if already_pruned_count > 0:
            return already_pruned_count
        else:
            pruned_count = 0
            for name in self.selected_names:
                if name in block_gpts:
                    # 判断：只有在 target_layers 列表里的才剪枝
                    if name in self.target_layers:
                        sparsity = block_gpts[name].prune_2_4(method=self.prune_method)
                        pruned_count += 1
                        print(f"      Pruned {name} (Sparsity: {sparsity:.2f})")
                    else:
                        # 不剪枝，仅仅清理掉 Hessian 占用的显存
                        del block_gpts[name].H 

            return pruned_count


    def run(self):

        print("\n[Phase 2] Training R1 & Pruning Targets...")

        # print("self.global_gpts:{}".format(self.global_gpts))

        for model_name, model in self.models_to_adapt.items():
            print(f"  > Processing {model_name}...")
            
            for i, block in enumerate(model.blocks):
                # 1. 搬运 Block 和 Hessian
                block = block.to(self.device)
                block_gpts = self.global_gpts[model_name][i]
                hessian_dict_gpu = {name: gpt.H.to(self.device) for name, gpt in block_gpts.items()}
                
                optimize_results = self.optimize(block, hessian_dict_gpu)

                already_pruned_count = self.weight_fusion(block, optimize_results)

                pruned_count = self.pruning(block_gpts, already_pruned_count)

                # 6. 清理
                del hessian_dict_gpu, optimize_results #TODO self.opt/rotator need to be cleaned
                block = block.cpu() 
                model.blocks[i] = block 
                
                del block_gpts
                gc.collect()
                torch.cuda.empty_cache()
                
                print(f"    Block {i} Done. Pruned {pruned_count} layers.")

        save_sparse_model(self.wan_t2v.low_noise_model,  self.save_path,'low')
        save_sparse_model(self.wan_t2v.high_noise_model, self.save_path,'high')
        print("\nDone! 模型已修复 AdaLN 兼容性并完成指定层剪枝。")