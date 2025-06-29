# HRCF Recommendation System Example

这个示例演示了如何使用OpenEvolve框架来进化HRCF（Hyperbolic Regularized Collaborative Filtering）推荐算法。

## 概述

HRCF是一种先进的协同过滤算法，它结合了：
- **双曲空间嵌入**: 在双曲空间（庞加莱球模型）中学习用户和物品的表示
- **几何正则化**: 使用双曲几何的性质来改善嵌入质量
- **排序学习**: 通过对比正负样本来优化推荐性能

## 文件结构

- `initial_program.py`: 包含基础的HRCF算法实现，其中EVOLVE-BLOCK部分可以被LLM进化
- `evaluator.py`: 评估进化后算法性能的评估器
- `config.yaml`: 针对推荐系统优化的配置文件
- `README.md`: 本说明文件

## 核心算法思想

### 双曲空间嵌入
```python
# 将嵌入投影到庞加莱球 (||x|| < 1)
def project_to_poincare_ball(x):
    norms = torch.norm(x, dim=-1, keepdim=True)
    return x / norms * torch.clamp(norms, max=1.0 - eps)

# 计算双曲距离
def hyperbolic_distance(x, y):
    # 使用庞加莱距离公式
    return torch.acosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
```

### 几何正则化
```python
# 鼓励物品嵌入具有多样化的范数
item_norms = torch.norm(item_embeddings, dim=1)
geo_reg = 1.0 / (item_norms.mean() + 1e-8)
```

### 排序损失
```python
# 偏好正样本的距离小于负样本
ranking_loss = F.relu(pos_distances - neg_distances + margin).mean()
```

## 运行示例

### 基础运行
```bash
cd /root/openevlove
python openevolve-run.py examples/hrcf_recsys/initial_program.py examples/hrcf_recsys/evaluator.py --config examples/hrcf_recsys/config.yaml --iterations 100
```

### GPU加速运行
示例会自动检测并使用GPU（如果可用）：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 自定义参数
可以在`initial_program.py`中调整以下参数：
- `num_epochs`: 训练轮数
- `embedding_dim`: 嵌入维度
- `lr`: 学习率
- `margin`: 排序损失的边界值

## 进化目标

OpenEvolve将尝试进化EVOLVE-BLOCK中的算法，可能的改进方向包括：

1. **优化器改进**: 从基础梯度下降进化到更复杂的优化策略
2. **正则化技术**: 发现新的几何正则化方法
3. **负采样策略**: 改进负样本采样的方法
4. **损失函数**: 发现更好的损失函数组合
5. **双曲几何**: 探索不同的双曲空间操作

## 评估指标

- `recommendation_score`: 推荐质量（0-1，越高越好）
- `execution_time`: 执行时间
- `memory_efficiency`: 内存效率
- `convergence_speed`: 收敛稳定性
- `combined_score`: 综合得分

## GPU要求

- 推荐使用CUDA兼容的GPU
- 至少4GB显存
- 示例会自动适配可用的硬件资源

## 预期结果

经过进化后，算法可能会：
- 提高推荐准确率
- 减少训练时间
- 增强收敛稳定性
- 发现新的双曲几何技巧

## 相关论文

- Yang, M., et al. "HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization." WWW 2022.
- 原始实现：https://github.com/marlin-codes/HRCF 