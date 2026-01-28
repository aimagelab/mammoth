from typing import Tuple

import torch
import torch.nn as nn
import warnings
from .supported_layers_grad_samplers_dotprod import (
    _supported_layers_dotprod,
)

# done: 修改ghost_dot_prod的输入
def requires_grad(module: nn.Module) -> bool:
    """
    检查指定模块中的任何参数是否需要梯度。
    """
    for p in module.parameters():
        # 如果参数没有 initially_requires_grad 属性，则使用 requires_grad
        if not hasattr(p, 'initially_requires_grad'):
            p.initially_requires_grad = p.requires_grad
        if p.initially_requires_grad:
            return True
    return False

# done: 修改create_backward_hook的输入，增加val data的input activation and grad output
def create_backward_hook(layer: nn.Module, engine, compute_layer_dotprod):
    def hook(_module, _grad_input, grad_output):
        # 快速路径：如果正在计算验证梯度或无梯度输出，直接返回
        if engine._computing_val_grads:
            return None
        
        if not grad_output or grad_output[0] is None:
            return None

        try:
            # 钩子被 loss_train.backward() 触发，所以 grad_output 是纯净的 B_train
            with torch.no_grad():
                B_train = grad_output[0]

                # 直接从 layer 属性读取激活，避免频繁 dict 查找
                A_train = getattr(layer, "_act_train", None)
                if A_train is None:
                    # 仅在找不到激活时警告
                    warnings.warn(f"Training activations for layer {layer.name} not found.")
                    return None

                # --- 计算点积 ---
                # 直接调用预先解析的函数，避免字典查找
                compute_layer_dotprod(layer, A_train, B_train, engine._val_grads)

        except Exception as e:
            # 捕获所有异常，避免训练中断
            warnings.warn(f"Error in backward hook for layer {layer.name}: {e}")

        return None
    return hook


def add_hooks(model: nn.Module, engine):
    r"""
    Adds hooks to a model to compute gradient dot products and accumulate
    training gradients.
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    handles = []
    target_layers = []
    
    def collect_target_layers(module, name=""):
        """递归收集目标层"""
        # 优化：先检查是否需要梯度，避免不必要的类型检查
        if not requires_grad(module):
            # 如果当前模块不需要梯度，仍需检查子模块
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                collect_target_layers(child_module, full_name)
            return
            
        # 检查是否为支持的层类型
        for cls in _supported_layers_dotprod.keys():
            if isinstance(module, cls):
                target_layers.append((name, module))
                return  # 找到匹配的层后直接返回，不再检查子模块
        
        # 如果不是支持的层类型，继续检查子模块
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            collect_target_layers(child_module, full_name)
    # 从根模块开始收集
    collect_target_layers(model)
    
    # 为收集到的层注册 hooks
    for name, layer in target_layers:
        # 为层设置唯一的名称属性
        layer.name = name
        layer.layer_name = name  # 用于grad字典中的键名
        
        # 找到对应的层类型
        layer_cls = None
        for cls in _supported_layers_dotprod.keys():
            if isinstance(layer, cls):
                layer_cls = cls
                break
        
        if layer_cls is None:
            continue  # 这种情况理论上不应该发生
        
        # 预先获取计算函数，避免在 hook 中重复查找
        compute_fn = _supported_layers_dotprod[layer_cls]
            
        # 修正闭包问题：使用默认参数捕获当前值
        def make_forward_hook(layer_ref):
            return lambda mod, i, o: _capture_activations(layer_ref, i, engine)
        
        def make_backward_hook(layer_ref, compute_fn_ref):
            return create_backward_hook(layer_ref, engine, compute_fn_ref)
        
        handles.append(layer.register_forward_hook(make_forward_hook(layer)))
        handles.append(layer.register_full_backward_hook(make_backward_hook(layer, compute_fn)))

    model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)
    # check_hooks_registered(model)

def remove_hooks(model: nn.Module):
    """Removes hooks added by `add_hooks()`."""
    if hasattr(model, "autograd_grad_sample_hooks"):
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks


def _capture_activations(layer: nn.Module, inputs: Tuple, engine):
    """根据 engine 的模式，将激活值保存到 engine 的对应字典中。"""
    mode = engine._capture_mode
    act = inputs[0].detach()
    # 使用 layer 的唯一 name 作为 key
    engine._activations[mode][layer.name] = act
    # 直接挂到 layer，供 backward 快速读取
    if mode == "train":
        layer._act_train = act


def check_hooks_registered(model):
    """检查模型中的层是否注册了 hook"""
    hook_count = 0
    hook_types = [
        ('_forward_hooks', 'forward hook'),
        ('_forward_pre_hooks', 'forward pre-hook'),
        ('_backward_hooks', 'backward hook'),
        ('_backward_pre_hooks', 'backward pre-hook')
    ]
    
    for name, module in model.named_modules():
        for attr_name, hook_type in hook_types:
            if hasattr(module, attr_name) and getattr(module, attr_name):
                print(f"✓ 层 '{name}' 已注册 {hook_type}")
                hook_count += 1
                break  # 每个模块只计数一次
            
    print(f"总共发现 {hook_count} 个注册的 hook")
    
    # 检查模型是否存储了hook handles列表
    if hasattr(model, "autograd_grad_sample_hooks"):
        print(f"模型已存储 {len(model.autograd_grad_sample_hooks)} 个hook handles")
        return True
    return False
