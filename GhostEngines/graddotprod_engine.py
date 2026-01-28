from typing import Dict, Optional, Sequence, List, Tuple
import os
import warnings
import json
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple
import heapq
import torch
from torch import nn

# 延迟导入以避免循环导入 - 将在 attach() 方法中导入
# from . import autograd_grad_sample_dotprod
from .supported_layers_grad_samplers_dotprod import _supported_layers_dotprod


class SampleTracker:
    """
    简化的样本追踪器,专注于累计grad_dot_prod和类级别统计
    """
    def __init__(self, *, enable_epoch_history: bool = True, max_samples: Optional[int] = None):
        """样本追踪器.

        Args:
            enable_epoch_history: 是否记录逐 epoch 的样本历史(会有大量字典写入)。
            max_samples: 预分配的最大样本数,为 None 时按需扩容(指数式),
                         为正整数时一次性预分配,避免频繁 np.concatenate。
        """
        # 映射: 全局样本ID -> 稠密数组中的 offset
        self._global_id_to_offset: Dict[int, int] = {}
        # 反向映射: offset -> sample_id 字符串
        self._offset_to_sample_id: List[str] = []

        # 采用可扩展的稠密数组存储累计值
        if max_samples is not None and max_samples > 0:
            self._cumulative_values = np.zeros(int(max_samples), dtype=np.float64)
            self._capacity = int(max_samples)
        else:
            # 从 0 开始,按需指数扩容
            self._cumulative_values = np.zeros(0, dtype=np.float64)
            self._capacity = 0
        # 当前已使用的样本槽位数
        self._size = 0

        self._cumulative_cache: Dict[str, float] = {}
        self._cumulative_cache_dirty = False

        # 存储样本的元信息: {sample_id: {"class": class_id, "original_index": idx}}
        self.sample_metadata = {}

        # 是否记录每个 epoch 的逐样本历史(开销较大)
        self.enable_epoch_history = enable_epoch_history
        # 存储历史记录: {epoch: {class_id: {sample_id: grad_dot_prod_value}}}
        self.epoch_history = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            if enable_epoch_history
            else None
        )

        # 当前 batch 的样本索引映射
        self.current_batch_samples: List[str] = []
    
    def register_batch_samples(self, sample_indices: List[int], sample_classes: List[int]):
        """
        注册当前 batch 中的样本信息(使用全局稳定样本ID)
        
        重要;这里的 sample_indices 应该是“全局稳定ID”,即跨任务/跨数据源不变的样本标识,
        而不是当前 DataLoader 或当前数据集切片中的局部下标。

        支持的 ID 形式;
        - 整型(int、numpy.integer、torch scalar tensor)
        - 字符串;形如 "sample_123" 或 "123"

        Args:
            sample_indices: 全局样本ID(或可解析为此的值)
            sample_classes: 样本对应的类别
        """
        import numpy as _np
        import torch as _torch

        def _to_global_int_id(x) -> int:
            # 允许多种输入类型,统一转为内置 int
            if isinstance(x, int):
                return int(x)
            if isinstance(x, _np.integer):
                return int(x)
            if isinstance(x, str):
                xs = x.strip()
                if xs.startswith("sample_"):
                    xs = xs[len("sample_"):]
                # 若剩余是数字则转为 int
                try:
                    return int(xs)
                except Exception:
                    raise ValueError(f"无法将样本ID字符串解析为整数: {x}")
            if _torch.is_tensor(x):
                # 支持 0-dim tensor
                if x.dim() == 0:
                    return int(x.item())
                raise ValueError(f"张量ID必须是标量张量,收到形状: {tuple(x.shape)}")
            # 其他类型尝试直接转 int
            try:
                return int(x)
            except Exception:
                raise ValueError(f"不支持的样本ID类型: {type(x)} -> {x}")

        # 规范化类别为内置 int
        # 优化:批量转换
        if hasattr(sample_classes, 'tolist'):
            norm_classes = sample_classes.tolist()
        else:
            norm_classes = [int(c.item()) if hasattr(c, 'item') else int(c) for c in sample_classes]
            
        if hasattr(sample_indices, 'tolist'):
            raw_indices = sample_indices.tolist()
        else:
            raw_indices = sample_indices

        self.current_batch_samples = []
        new_global_ids: List[int] = []
        new_sample_ids: List[str] = []
        
        # 优化:预先获取字典方法
        meta_get = self.sample_metadata.get
        global_id_to_offset_contains = self._global_id_to_offset.__contains__
        
        for raw_idx, class_id in zip(raw_indices, norm_classes):
            # 内联 _to_global_int_id 的常见情况逻辑
            if isinstance(raw_idx, int):
                gid = raw_idx
            elif isinstance(raw_idx, str):
                # 简单处理 "sample_123" 或 "123"
                if raw_idx.startswith("sample_"):
                    gid = int(raw_idx[7:])
                else:
                    gid = int(raw_idx)
            else:
                # Fallback for other types
                gid = _to_global_int_id(raw_idx)
                
            sample_id = f"sample_{gid}"
            self.current_batch_samples.append(sample_id)

            meta = meta_get(sample_id)
            if meta is None:
                self.sample_metadata[sample_id] = {
                    "class": int(class_id),
                    "original_index": int(gid),
                    "global_id": int(gid),
                }
            else:
                # 只有在类别不匹配时才警告,减少常规操作
                if meta["class"] != int(class_id):
                    warnings.warn(
                        f"全局样本 {sample_id} 已注册为类别 {meta['class']},本批次收到类别 {int(class_id)}。"
                        f" 将保留首次登记的类别 {meta['class']} 并忽略本次类别,用于一致的累计统计。"
                    )
                # 更新其他字段(如果需要)
                # meta["global_id"] = int(gid) # 已经是这个值了

            if not global_id_to_offset_contains(gid):
                new_global_ids.append(int(gid))
                new_sample_ids.append(sample_id)

        if new_global_ids:
            # 需要新增的样本数量
            need = len(new_global_ids)
            # 确保容量足够,不足时指数扩容以避免频繁 np.concatenate
            if self._size + need > self._capacity:
                # 新容量: 至少能容纳 size+need,并且不小于 2x 旧容量
                new_capacity = max(self._size + need, max(1, self._capacity * 2))
                new_array = np.zeros(new_capacity, dtype=self._cumulative_values.dtype)
                if self._size > 0:
                    new_array[: self._size] = self._cumulative_values[: self._size]
                self._cumulative_values = new_array
                self._capacity = new_capacity

            # 为新样本分配 offset
            for gid, sid in zip(new_global_ids, new_sample_ids):
                offset = self._size
                self._size += 1
                self._global_id_to_offset[gid] = offset
                self._offset_to_sample_id.append(sid)

            self._cumulative_cache_dirty = True
    
    def record_grad_dot_prod(self, grad_dot_prod_values: torch.Tensor, epoch: int):
        """
        记录当前 batch 的 grad_dot_prod 值,同时更新累计值
        
        Args:
            grad_dot_prod_values: 形状为 (train_batch_size,) 的张量
            epoch: 当前 epoch
        """

        values_tensor = grad_dot_prod_values.detach()
        if values_tensor.dim() > 1:
            values_tensor = values_tensor.flatten()

        values_cpu = values_tensor.to(device="cpu", non_blocking=True)
        grad_dot_prod_cpu = values_cpu.numpy()

        expected = len(self.current_batch_samples)
        actual = grad_dot_prod_cpu.shape[0]
        if actual != expected:
            warnings.warn(
                f"Mismatch: grad_dot_prod length ({actual}) != batch samples length ({expected})"
            )

        effective = min(actual, expected)
        if effective == 0:
            return

        batch_sample_ids = self.current_batch_samples[:effective]
        batch_values = grad_dot_prod_cpu[:effective]
        # 如果需要记录逐 epoch 历史,预先取出 bucket
        epoch_bucket = self.epoch_history[epoch] if self.enable_epoch_history else None

        # 预先构造 offsets 数组,避免 Python generator 带来的额外开销
        offsets = [
            self._global_id_to_offset[self.sample_metadata[sid]["global_id"]]
            for sid in batch_sample_ids
        ]
        offsets = np.asarray(offsets, dtype=np.int64)

        # 仅在已使用的前 self._size 范围内更新
        np.add.at(
            self._cumulative_values,
            offsets,
            batch_values.astype(self._cumulative_values.dtype, copy=False),
        )
        self._cumulative_cache_dirty = True

        # 可选:记录逐 epoch 的样本级历史(开销较大,可关闭)
        if self.enable_epoch_history and epoch_bucket is not None:
            for sample_id, value in zip(batch_sample_ids, batch_values):
                class_id = self.sample_metadata[sample_id]["class"]
                epoch_bucket[class_id][sample_id] = float(value)

    @property
    def sample_cumulative_grad_dot_prod(self) -> Dict[str, float]:
        if self._cumulative_cache_dirty:
            self._cumulative_cache = {
                sid: float(self._cumulative_values[idx])
                for idx, sid in enumerate(self._offset_to_sample_id)
            }
            self._cumulative_cache_dirty = False
        return self._cumulative_cache
    
    def get_sample_cumulative_grad_dot_prod(self, sample_id: str) -> float:
        """获取特定样本的累计grad_dot_prod值"""
        meta = self.sample_metadata.get(sample_id)
        if not meta:
            return 0.0
        offset = self._global_id_to_offset.get(meta["global_id"])
        if offset is None or offset >= self._cumulative_values.shape[0]:
            return 0.0
        return float(self._cumulative_values[offset])
    
    def get_class_cumulative_statistics(self, class_id: int) -> Dict[str, float]:
        """
        获取特定类别的累计统计信息
        
        Args:
            class_id: 类别ID
        """
        # 使用列表推导式一次性获取该类别所有样本的累计值
        offsets = [
            self._global_id_to_offset[self.sample_metadata[sample_id]["global_id"]]
            for sample_id, metadata in self.sample_metadata.items()
            if metadata["class"] == class_id
        ]

        if not offsets:
            return {"sample_count": 0}
        
        values_array = self._cumulative_values[np.asarray(offsets, dtype=np.int64)]
        
        return {
            "sample_count": len(offsets),
            "cumulative_sum": float(values_array.sum()),
            "cumulative_mean": float(values_array.mean()),
            "cumulative_std": float(values_array.std()),
            "cumulative_min": float(values_array.min()),
            "cumulative_max": float(values_array.max()),
        }
    
    def get_all_class_statistics(self) -> Dict[int, Dict[str, float]]:
        """获取所有类别的累计统计信息"""
        # 使用单次遍历构建按类别分组的数据
        class_values = defaultdict(list)
        for offset, sample_id in enumerate(self._offset_to_sample_id):
            metadata = self.sample_metadata.get(sample_id)
            if not metadata:
                continue
            class_id = metadata["class"]
            cumulative_value = self._cumulative_values[offset]
            class_values[class_id].append(cumulative_value)
        
        # 对每个类别计算统计信息
        statistics = {}
        for class_id, values in class_values.items():
            if not values:
                statistics[class_id] = {"sample_count": 0}
            else:
                values_array = np.array(values)
                positive_sum = float(np.maximum(values_array, 0.0).sum())
                statistics[class_id] = {
                    "sample_count": len(values),
                    "cumulative_sum": float(values_array.sum()),
                    "cumulative_mean": float(values_array.mean()),
                    "cumulative_std": float(values_array.std()),
                    "cumulative_min": float(values_array.min()),
                    "cumulative_max": float(values_array.max()),
                    "positive_sum": positive_sum,
                }
        
        return statistics


class GradDotProdAnalyzer:
    """
    简化的grad_dot_prod数据分析工具
    """
    def __init__(self, sample_tracker: SampleTracker):
        self.tracker = sample_tracker
    
    def export_statistics(self, save_path: str, current_epoch: Optional[int] = None) -> None:
        """
        导出简化的统计数据
        
        Args:
            save_path: 保存路径
            current_epoch: 当前epoch(用于文件命名)
        """
        def save_json_file(data: dict, filepath: str) -> None:
            """通用的JSON文件保存函数"""
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        base_path = save_path
        
        # 准备要保存的数据
        sample_metadata_data = {"sample_metadata": self.tracker.sample_metadata}
        class_statistics_data = {"class_statistics": self.tracker.get_all_class_statistics()}
        
        # epoch_history_data按照类别组织,样本按grad_dot_prod值从大到小排序
        # 注意: 当 SampleTracker.enable_epoch_history=False 时,epoch_history 为 None
        if getattr(self.tracker, "epoch_history", None) is not None:
            raw_epoch_data = self.tracker.epoch_history.get(current_epoch, {})
        else:
            raw_epoch_data = {}
        sorted_epoch_data = {}
        
        for class_id, samples_dict in raw_epoch_data.items():
            # 将样本按grad_dot_prod值从大到小排序
            sorted_samples = sorted(samples_dict.items(), key=lambda x: x[1], reverse=True)
            # 转换为有序字典保持排序
            # from collections import OrderedDict
            sorted_epoch_data[class_id] = OrderedDict(sorted_samples)
        
        epoch_history_data = {"epoch_history": sorted_epoch_data}
        
        # 样本累积值数据按照类别组织,样本按累积grad_dot_prod值从大到小排序
        cumulative_by_class = {}
        for sample_id, metadata in self.tracker.sample_metadata.items():
            class_id = metadata["class"]
            cumulative_value = self.tracker.get_sample_cumulative_grad_dot_prod(sample_id)
            
            if class_id not in cumulative_by_class:
                cumulative_by_class[class_id] = []
            cumulative_by_class[class_id].append((sample_id, cumulative_value))
        
        # 对每个类别内的样本按累积值排序
        sorted_cumulative_data = {}
        for class_id, samples_list in cumulative_by_class.items():
            # 按累积值从大到小排序
            sorted_samples = sorted(samples_list, key=lambda x: x[1], reverse=True)
            sorted_cumulative_data[class_id] = OrderedDict(sorted_samples)
        
        cumulative_data = {"cumulative_by_class": sorted_cumulative_data}

        # 根据epoch确定文件路径
        if current_epoch is not None:
            # 只在第一个epoch保存样本元数据
            if current_epoch == 0:
                metadata_file = os.path.join(base_path, "meta_data", "sample_metadata.json")
                save_json_file(sample_metadata_data, metadata_file)
            
            # 每个epoch保存类别统计和历史数据
            class_stats_file = os.path.join(base_path, "class_statistics", f"class_statistics_epoch_{current_epoch}.json")
            epoch_history_file = os.path.join(base_path, "epoch_history", f"epoch_history_epoch_{current_epoch}.json")
            cumulative_file = os.path.join(base_path, "cumulative_by_class", f"cumulative_by_class_epoch_{current_epoch}.json")
        else:
            # 没有指定epoch时的默认文件名
            class_stats_file = os.path.join(base_path, "class_statistics", "class_statistics.json")
            epoch_history_file = os.path.join(base_path, "epoch_history", "epoch_history.json")
            cumulative_file = os.path.join(base_path, "cumulative_by_class", "cumulative_by_class.json")

        # 保存文件
        save_json_file(class_statistics_data, class_stats_file)
        save_json_file(epoch_history_data, epoch_history_file)
        save_json_file(cumulative_data, cumulative_file)

    def compute_class_values(
        self,
        *,
        truncate_negative: bool = True,
        agg: str = "mean",
        beta: float = 1.0,
        classes: Optional[Sequence[int]] = None,
        first_n_epochs: Optional[int] = None,
    ) -> Dict[int, float]:
        """根据样本累计 grad_dot_prod 计算每个类的 class_values.

        Args:
            truncate_negative: 是否将负值截断为 0。
            agg: 聚合方式: 'mean' | 'sum' | 'mixed'。
            beta: 仅当 agg='mixed' 时使用。mixed = |D_c|^beta * mean(tilde_phi)。
            classes: 仅计算指定类；为 None 时使用 tracker 中出现过的类。

        Args:
            first_n_epochs: 仅使用前 n 个 epoch (包含端点, 0-based) 的累计值;
                             为 None 时使用全程累计。

        Returns:
            dict[class_id] -> class_value
        """

        agg_l = (agg or "mean").lower().strip()
        if agg_l not in {"mean", "sum", "mixed"}:
            raise ValueError(f"Unsupported agg: {agg}. Use 'mean'/'sum'/'mixed'.")

        # 如果需要按 epoch 截断, 使用 epoch_history 重建样本累计值
        sample_values_override: Optional[Dict[str, float]] = None
        if first_n_epochs is not None:
            if getattr(self.tracker, "epoch_history", None) is None:
                warnings.warn(
                    "first_n_epochs 指定但 epoch_history 未开启; 回退为全程累计值。"
                )
            else:
                upper = int(first_n_epochs)
                if upper < 0:
                    warnings.warn(
                        "first_n_epochs < 0; 回退为全程累计值。"
                    )
                else:
                    epoch_hist = self.tracker.epoch_history
                    sample_values_override = defaultdict(float)
                    # 累加 0..upper 的样本值
                    for ep, class_bucket in epoch_hist.items():
                        if ep > upper:
                            continue
                        for _cls, samples_dict in class_bucket.items():
                            for sid, val in samples_dict.items():
                                sample_values_override[sid] += float(val)

        # 收集每个类的样本值
        class_to_values: Dict[int, List[float]] = defaultdict(list)
        for sample_id, meta in self.tracker.sample_metadata.items():
            cls = int(meta.get("class", -1))
            if classes is not None and cls not in classes:
                continue
            if sample_values_override is not None:
                v = sample_values_override.get(sample_id, 0.0)
            else:
                v = self.tracker.get_sample_cumulative_grad_dot_prod(sample_id)
            if truncate_negative and v < 0:
                v = 0.0
            class_to_values[cls].append(float(v))

        # 如果指定了 classes,但某些类没有样本,也要在输出中出现
        if classes is not None:
            for cls in classes:
                class_to_values.setdefault(int(cls), [])

        class_values: Dict[int, float] = {}
        for cls, values in class_to_values.items():
            if not values:
                class_values[int(cls)] = 0.0
                continue
            values_arr = np.asarray(values, dtype=np.float64)
            # note: test31: clip top 2% extreme values
            # 过滤每类中前 2% 的极大值,缓和异常样本的影响
            if values_arr.size > 0:
                k = int(values_arr.size * 0.02)
                if k > 0:
                    values_arr = np.sort(values_arr)
                    values_arr = values_arr[:-k]
            if agg_l == "sum":
                class_values[int(cls)] = float(values_arr.sum())
            elif agg_l == "mean":
                class_values[int(cls)] = float(values_arr.mean())
            else:
                # mixed: |D_c|^beta * mean
                b = float(beta)
                n = float(values_arr.shape[0])
                m = float(values_arr.mean())
                class_values[int(cls)] = (n ** b) * m

        return class_values
    # todo: test new allocation logic, minmax
    def softmax_allocate(
        self,
        class_values: Dict[int, float],
        *,
        M: int,
        tau: float = 1.0,
        eps: float = 1e-8,
        minmax: bool = True,
        zscore_eps: float = 1e-6,
        mix_lambda: float = 0.9,
        min_per_class: int = 1,
        return_debug: bool = False,
    ):
        """根据 class_values 进行 softmax+均匀混合分配,输出每类整数 quota。

        步骤:
          1) task 内标准化: \tilde{V} = (V - mu) / (sigma + zscore_eps) 当 zscore=True
          2) softmax 得到 q_{t,c}
          3) 均匀混合: p_{t,c} = (1-lambda)/C + lambda * q_{t,c}
          4) 先发放每类保底 min_per_class, 剩余预算用最大余数法补齐,确保总和 = M

        Args:
            class_values: dict[class_id] -> value(建议非负,若有负值会先截断为 0)
            M: 总 memory 预算(整数)
            tau: softmax 温度,越小越“偏科”
            eps: 数值稳定项
            zscore: 是否在 task 内做 z-score
            zscore_eps: z-score 中的数值稳定项
            mix_lambda: 均匀混合系数,0 表示纯均匀,1 表示纯 softmax
            min_per_class: 每类保底配额(会在 M 不足时自动下调)
            return_debug: True 时返回 (quota_dict, debug_dict)
        """

        if M < 0:
            raise ValueError(f"M must be >= 0, got {M}")
        keys = sorted(int(k) for k in class_values.keys())
        if len(keys) == 0:
            out = {}
            return (out, {"keys": [], "v": [], "q": [], "p": [], "entropy": 0.0}) if return_debug else out

        v = np.asarray([float(class_values[k]) for k in keys], dtype=np.float64)
        # 截断负值,避免分配出现异常
        v = np.maximum(v, 0.0)

        t = float(tau)
        if t <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        c = len(keys)
        lam = float(mix_lambda)
        if lam < 0.0 or lam > 1.0:
            lam = min(max(lam, 0.0), 1.0)

        # minmax 标准化
        if minmax:
            logits = np.zeros_like(v)
            v_min = float(np.min(v))
            v_max = float(np.max(v))
            v_range = v_max - v_min
            if v_range > 0.0:
                # todo: test2-30
                # logits = (1 - (v - v_min) / (v_range + zscore_eps))
                logits = ((v - v_min) / (v_range + zscore_eps))
        else:
            logits = v

        z = logits / t
        z = z - float(np.max(z))  # for stability
        exp_z = np.exp(z)
        q = exp_z / (float(exp_z.sum()) + eps)

        # 均匀混合,避免尾部类别被饿死
        uniform = np.full_like(q, 1.0 / float(c))
        p = (1.0 - lam) * uniform + lam * q

        # 每类保底;若预算不足,自动下调保底
        mmin = int(min_per_class)
        if mmin < 0:
            raise ValueError(f"min_per_class must be >= 0, got {min_per_class}")
        effective_min = min(mmin, M // c) if c > 0 else 0
        base = np.full(shape=(c,), fill_value=effective_min, dtype=np.int64)
        min_total = int(effective_min * c)
        remainder_budget = int(M - min_total)

        alloc_float = p * float(max(remainder_budget, 0))
        extra = np.floor(alloc_float).astype(np.int64)
        base += extra
        remainder = int(remainder_budget - int(extra.sum()))

        if remainder > 0:
            frac = alloc_float - extra.astype(np.float64)
            order = np.lexsort((np.asarray(keys, dtype=np.int64), -frac))
            for idx in order[:remainder]:
                base[int(idx)] += 1
        elif remainder < 0:
            need = int(-remainder)
            frac = alloc_float - extra.astype(np.float64)
            order = np.lexsort((np.asarray(keys, dtype=np.int64), frac))
            for idx in order[:need]:
                if base[int(idx)] > 0:
                    base[int(idx)] -= 1

        quota = {k: int(qv) for k, qv in zip(keys, base.tolist())}

        if return_debug:
            entropy = float(-(q * np.log(q + eps)).sum())
            debug = {
                "keys": keys,
                "v": v.tolist(),
                "minmax": bool(minmax),
                "logits": logits.tolist() if isinstance(logits, np.ndarray) else [float(x) for x in logits],
                "q": q.tolist(),
                "p": p.tolist(),
                "mix_lambda": float(lam),
                "min_per_class": int(mmin),
                "effective_min": int(effective_min),
                "min_total": int(min_total),
                "remainder_budget": int(remainder_budget),
                "entropy": entropy,
                "sum_quota": int(sum(quota.values())),
            }
            return quota, debug

        return quota

    def get_top_samples_by_cumulative(self, top_k: int = 10) -> List[Tuple[str, float, int]]:
        """
        获取累计grad_dot_prod值最高的前k个样本
        
        Returns:
            List of (sample_id, cumulative_value, class_id)
        """
        sample_values = []
        for sample_id, cumulative_value in self.tracker.sample_cumulative_grad_dot_prod.items():
            metadata = self.tracker.sample_metadata.get(sample_id, {})
            class_id = metadata.get("class", -1)
            sample_values.append((sample_id, cumulative_value, class_id))
        
        # 按累计值排序
        sample_values.sort(key=lambda x: x[1], reverse=True)
        return sample_values[:top_k]

    def get_top_samples_by_class(self, class_id: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        获取指定类中累计grad_dot_prod值最高的前k个样本

        Args:
            class_id: 类别ID
            top_k: 要返回的样本数量

        Returns:
            List of (sample_id, cumulative_value)
        """
        sample_values = []
        for sample_id, cumulative_value in self.tracker.sample_cumulative_grad_dot_prod.items():
            metadata = self.tracker.sample_metadata.get(sample_id, {})
            if metadata.get("class", -1) == class_id:
                sample_values.append((sample_id, cumulative_value))

        # 按累计值排序
        sample_values.sort(key=lambda x: x[1], reverse=True)
        return sample_values[:top_k]

    def water_filling_allocate(
        self, 
        M: int, 
        m_min: int = 0, 
        relu: bool = False,
        classes: Optional[Sequence[int]] = None
    ) -> Dict[int, int]:
        """实现: 以每类中单个样本的gdp值为价值信号, 进行 water filling 分配

        Args:
            M: 总预算
            m_min: 每类保底
            relu: 是否对 value 做 ReLU
            classes: 指定参与分配的类别列表。如果为 None，则使用 tracker 中所有出现过的类别。
                     注意：M 是分配给这些 classes 的总预算。
        """
        class_to_values: Dict[int, List[float]] = defaultdict(list)
        
        # 优化查询集
        valid_classes = set(classes) if classes is not None else None

        for sample_id, meta in self.tracker.sample_metadata.items():
            cls = int(meta.get("class", -1))
            
            # 如果指定了 scope，且当前类不在 scope 中，跳过
            if valid_classes is not None and cls not in valid_classes:
                continue

            v = self.tracker.get_sample_cumulative_grad_dot_prod(sample_id)
            if relu:
                v = max(0.0, v)
            class_to_values[cls].append(float(v))
        
        # 如果指定了 classes，确保所有指定类都在 keys 中（即使没有样本，也要占位以分配 m_min 或 0）
        if valid_classes is not None:
            for c in valid_classes:
                if c not in class_to_values:
                    class_to_values[c] = []

        # 根据边际收益递减原则进行quota allocation
        sorted_classes = sorted(class_to_values.keys())
        C = len(sorted_classes)
        if C == 0:
            return {}

        # 2) Initialize quotas with m_min (cap by class sample count)
        quota = {c: 0 for c in sorted_classes}
        # If you enforce m_min, you must ensure feasibility
        for c in sorted_classes:
            # 如果该类没有样本，配额只能是 0；如果有样本，至少给 m_min (但不超过实际样本数)
            available_samples = len(class_to_values[c])
            if available_samples > 0:
                quota[c] = min(m_min, available_samples)
            else:
                quota[c] = 0

        used = sum(quota.values())
        if used > M:
            # If infeasible, fall back: scale down m_min or just cut
            # simplest: trim uniformly
            remaining = M
            quota = {c: 0 for c in sorted_classes}
            used = 0
            # 简单的均匀回退重新分配，或者直接报错。这里做个简单的重新尽力分配
            # (略过复杂逻辑，直接进入下面的堆分配，因为 remaining = M)

        remaining = M - used

        # 3) Build per-class sorted marginal gains Delta_c(k)
        # Only need top (quota[c] + remaining) values at most; simplest: full sort
        deltas: Dict[int, List[float]] = {}
        for c in sorted_classes:
            vals = np.array(class_to_values[c], dtype=np.float64)
            if len(vals) > 0:
                vals.sort()
                vals = vals[::-1]  # descending
                deltas[c] = vals.tolist()
            else:
                deltas[c] = []

        # 4) Max-heap over next marginal gain for each class
        # heap elements: (-gain, class_id, next_index)
        heap: List[Tuple[float, int, int]] = []
        for c in sorted_classes:
            next_idx = quota[c]  # because first quota[c] slots already "taken" by m_min
            if next_idx < len(deltas[c]):
                gain = deltas[c][next_idx]
                heapq.heappush(heap, (-gain, c, next_idx))

        # 5) Allocate remaining slots
        while remaining > 0 and heap:
            neg_gain, c, idx = heapq.heappop(heap)
            # allocate one to class c
            quota[c] += 1
            remaining -= 1

            # push next marginal gain for this class
            next_idx = idx + 1
            if next_idx < len(deltas[c]):
                next_gain = deltas[c][next_idx]
                heapq.heappush(heap, (-next_gain, c, next_idx))

        return quota

class GradDotProdEngine:
    """
    简化的GradDotProdEngine,专注于累计grad_dot_prod计算
    """
    def __init__(
        self,
        module: nn.Module,
        *,
        val_batch_size: int,
        loss_reduction: str = 'mean',
        average_grad: bool = True,
        origin_params: Optional[Sequence[str]] = None,
        **unused_kwargs,
    ):
        """
        Initializes the GradDotProdEngine.

        Args:
            module: The PyTorch module to which the engine will be attached.
            val_batch_size: The number of samples in the fixed validation batch.
            loss_reduction: The reduction used for the loss function ('mean' or 'sum').
                          This is needed to correctly scale the backpropagated gradients.
            average_grad: If True, the model is updated with the average of the
                          training gradients. If False (default), the sum is used.
            origin_params: A list of parameter names to be used for the ghost
                           differentiation trick. Gradients will only be computed
                           starting from these parameters.
        """
        del unused_kwargs
        # 删除错误的super()调用 - GradDotProdEngine不继承任何类
        # super().__init__()

        self.module = module
        self.val_batch_size = val_batch_size
        self.loss_reduction = loss_reduction
        self.average_grad = average_grad
        self.named_params = list(
            (name, param) for (name, param) in module.named_parameters() if param.requires_grad
        )
        # 新增;用于存储分离的激活值
        self._activations: Dict[str, Dict] = {"train": {}, "val": {}}
        # 新增;用于在不同阶段之间传递验证梯度
        self._val_grads: Dict[str, torch.Tensor] = {}
        # 新增;用于控制钩子的行为模式 ('train' or 'val')
        self._capture_mode = 'train'
        # 新增;标记当前是否在计算验证梯度(不需要执行点积计算)
        self._computing_val_grads = False

        # 简化的数据追踪系统
        # 默认关闭 epoch_history,训练结束或需要详细分析时再单独构造带历史的 tracker
        self.sample_tracker = SampleTracker(enable_epoch_history=True)
        
        # 记录当前的 epoch
        self.current_epoch = 0
        
        # 缓存参数列表,避免每次step重复遍历
        self._cache_params()
        
        for name, param in module.named_parameters():

            # Store the original requires_grad status
            param.initially_requires_grad = bool(param.requires_grad)

        #     # Only set requires_grad to True for the very first layer (e.g., embedding layer)
        #     param.requires_grad = False
        # self._recursive_set_initially_requires_grads(self.module)
    
    # 新增;一个上下文管理器,用于设置钩子的捕获模式
    def capture_mode(self, mode: str):
        class ModeManager:
            def __init__(self, engine, mode):
                self.engine = engine
                self.mode = mode
                self.original_mode = self.engine._capture_mode
            def __enter__(self):
                self.engine._capture_mode = self.mode
            def __exit__(self, type, value, traceback):
                self.engine._capture_mode = self.original_mode
        return ModeManager(self, mode)

    # 新增;在每一步开始前,清空上一轮的存储
    def clear_hooks_data(self) -> None:
        """清空上一轮存储的激活值和验证梯度。"""
        self._activations = {"train": {}, "val": {}}
        self._val_grads = {}
        self._computing_val_grads = False

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        """Attach a modified hook logic."""        
        self.optimizer = optimizer

        # 延迟导入以避免循环导入
        from . import autograd_grad_sample_dotprod
        autograd_grad_sample_dotprod.add_hooks(
            model=self.module,
            engine=self,  # <-- 关键修改
        )

        # Keep a reference to the engine on the optimizer for convenience
        optimizer.grad_dot_prod_engine = self

    # 新增;核心计算方法,取代了旧的、分散的计算逻辑
    def step(self, loss_train: torch.Tensor, loss_val: torch.Tensor):
        """
        执行“两次前向,一次混合反向”的核心步骤。
        这个方法会计算验证梯度、触发训练梯度的反向传播,并准备好一切。
        """

        # 优化:使用缓存的参数列表
        params = self.cached_params
        named_params = self.cached_named_params

        # [Backward #1: Ghost Pass for Val]
        # 设置标记;当前正在计算验证梯度,不需要执行点积计算
        self._computing_val_grads = True
        
        try:
            # 计算验证梯度,但不存入 .grad;允许无效参数返回 None
            val_grads_flat = torch.autograd.grad(
                loss_val,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        finally:
            # 确保标记总是被重置,即使发生异常
            self._computing_val_grads = False

        # 将扁平的验证梯度列表转换成按层名组织的字典,过滤掉未参与验证反向传播的参数
        self._val_grads = {}
        unused_val_params: List[str] = []
        for param, grad in zip(params, val_grads_flat):
            name = named_params[param]
            if grad is None:
                unused_val_params.append(name)
                continue
            self._val_grads[name] = grad

        # 缓存未使用的参数名称,便于调试或后续分析(可选)
        self._unused_val_params = unused_val_params

        # [Backward #2: Real Pass for Train]
        # 这会触发我们修改后的后向钩子
        loss_train.backward()

        # 此时,后向钩子已经计算并填充了 param.train_grad 和 param.grad_dot_prod
        # 接下来聚合 dot_prod 结果
        self._aggregate_and_log_dot_products()

    def detach(self):
        """
        Detaches the engine from the optimizer, restoring its original state and
        cleaning up hooks and custom attributes.
        """
        optimizer = self.optimizer

        if hasattr(optimizer, "grad_dot_prod_engine"):
            del optimizer.grad_dot_prod_engine

        # Remove the hooks from the model
        # 延迟导入以避免循环导入
        from . import autograd_grad_sample_dotprod
        autograd_grad_sample_dotprod.remove_hooks(self.module)
        self.module.zero_grad()

        # Clean up custom attributes from all parameters
        for param in self.module.parameters():
            if hasattr(param, 'train_grad'):
                del param.train_grad
            if hasattr(param, 'grad_dot_prod'):
                del param.grad_dot_prod
            # 注意;激活值和验证梯度存储在engine中,不在parameter中
            # 这些已经在clear_hooks_data()中处理了

    def set_batch_info(self, sample_indices: List[int], sample_classes: List[int], epoch: int):
        """
        设置当前 batch 的样本信息(全局ID语义)
        
        注意;sample_indices 应为“全局稳定样本ID”,跨任务/跨epoch不变;
        切勿传入当前 DataLoader 的局部下标,以免与回放样本混合时发生索引/标签错配。

        Args:
            sample_indices: 全局样本ID(可为 int/np.int/torch scalar 或 "sample_123"/"123" 字符串)
            sample_classes: 训练样本对应的类别
            epoch: 当前 epoch
        """
        # 只保留训练样本的信息(排除验证样本)
        train_batch_size = len(sample_indices)
        if train_batch_size > 0:
            train_indices = sample_indices[:train_batch_size]
            train_classes = sample_classes[:train_batch_size]
            
            self.sample_tracker.register_batch_samples(train_indices, train_classes)
            self.current_epoch = epoch

    def get_analyzer(self) -> GradDotProdAnalyzer:
        """获取数据分析器"""
        return GradDotProdAnalyzer(self.sample_tracker)

    def export_sample_data(self, save_path: str):
        """导出样本追踪数据"""
        analyzer = self.get_analyzer()
        analyzer.export_statistics(save_path, current_epoch=self.current_epoch)

    def _aggregate_and_log_dot_products(self):
        """
        Calculates the total dot product for the current iteration by summing
        across all layers, and records sample-wise information.
        """
        # 收集所有有效的 grad_dot_prod 张量
        grad_dot_prods = []
        params_to_clean = []
        
        # 只遍历被 hook 的、可能产生 grad_dot_prod 的参数,避免扫描整个模型参数
        params_source = getattr(self, "_dotprod_params", None) or self.module.parameters()

        for param in params_source:
            if hasattr(param, 'grad_dot_prod'):
                grad_dot_prod = param.grad_dot_prod
                # 添加有效性检查
                if grad_dot_prod.numel() > 0:
                    grad_dot_prods.append(grad_dot_prod.detach())
                # 记录需要清理的参数
                params_to_clean.append(param)
        
        # 立即清理所有参数的 grad_dot_prod 属性以节省内存
        for param in params_to_clean:
            delattr(param, 'grad_dot_prod')
        
        if grad_dot_prods:
            # 优化:避免 torch.stack,使用 in-place 累加
            first = grad_dot_prods[0]
            total_dot_product_iter = first.clone()

            for t in grad_dot_prods[1:]:
                if t.device != total_dot_product_iter.device:
                    t = t.to(total_dot_product_iter.device)
                total_dot_product_iter.add_(t)

            # 可选:在 debug 环境下检查 NaN,避免频繁逐参数检查
            if torch.isnan(total_dot_product_iter).any():
                warnings.warn("NaNs found in accumulated gradient dot products; skipping this batch.")
                return

            # 记录每个样本的 grad_dot_prod 值
            self.sample_tracker.record_grad_dot_prod(
                total_dot_product_iter,
                self.current_epoch,
            )
        else:
            # If no dot products were computed, log a warning
            warnings.warn("No gradient dot products computed for this iteration.")

    def _cache_params(self):
        """缓存需要梯度的参数列表"""
        self.cached_params = [p for p in self.module.parameters() if p.requires_grad]
        self.cached_named_params = {p: name for name, p in self.module.named_parameters() if p.requires_grad}

    def refresh_params_cache(self):
        """如果模型结构或requires_grad状态发生变化,调用此方法刷新缓存"""
        self._cache_params()
