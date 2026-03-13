import os
from .engine import Engine

def run_wan_pruning(
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
    R_training,
    save_path
):
    print(f"\n[Final Solution v2] Starting Calibration & Pruning with Target Filter...")
    print(f"⚠️ Target Layers to Prune: {target_layers}")
    
    # device = torch.device(f"cuda:{device_id}")
    
    models_to_adapt = {
        "HighNoise": wan_t2v.high_noise_model,
        "LowNoise": wan_t2v.low_noise_model
    }
    
    # # --------------------------------------------------------------------------
    # # Phase 1: 收集 Hessian (这里是对全层收集，保证 R 计算准确)
    # # --------------------------------------------------------------------------
    # global_gpts = {}
    # all_handles = []
    
    # # 定义该 Block 内所有需要参与旋转的线性层（全集）
    # # 不管剪不剪，这些层都必须参与旋转计算，否则模型会裂开
    # ALL_LINEAR_LAYERS_NAMES = [
    #     "ffn.0",
    #     "ffn.2"
    # ]

    engine = Engine(
        wan_t2v,
        R_training, # set True for training R matrix False for training-free
        epoch=300,
        lr=0.01,
        prompts=prompts,
        models_to_adapt=models_to_adapt,
        size=size,
        frame_num=frame_num,
        shift=shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        base_seed=base_seed,
        target_layers=target_layers,
        prune_method=prune_method,
        device_id=device_id,
        save_path=save_path
    )

    # step 1 进行校准数据注册
    engine.registration()
    
    # step 2 运行训练以及剪枝
    engine.run()

    return True