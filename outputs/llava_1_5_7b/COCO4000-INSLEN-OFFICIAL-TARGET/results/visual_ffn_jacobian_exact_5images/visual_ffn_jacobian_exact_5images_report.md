# LLaVA-1.5：576-token exact visual FFN JVP（5 图）

每层将全部 576 个视觉 token 方向放入同一个 `vmap(jvp)`；目标位置仍为 InsLen 首 subtoken 的因果预测行。

## 覆盖

- 图片：5；唯一目标位置：14；image-layer 批次：160。
- 主 JVP 累计时间：0.922 秒。
- 模型加载后 allocated/reserved：13.157/13.176 GiB。
- 失败：0。

## 显存和数值

- 完整多模态 forward 峰值 allocated：13.813 GiB；相对 forward 前最大增量：0.648 GiB。
- JVP 峰值 allocated：14.037 GiB；相对调用前最大增量：0.319 GiB。
- Attention 重建 relative error 最大值：1.582e-03；cosine 最小值：0.99999887。
- `sum_j J a_j = J sum_j a_j` relative error 最大值：4.360e-03。
- eta=0.05 中心 finite difference：relative error 中位数 0.1516，cosine 中位数 0.988438。

### 全 576 并行 vs 64-token 分块

| Layer | 全并行时间 (s) | 64 分块时间 (s) | 全并行增量 (MiB) | 64 分块增量 (MiB) | relative L2 delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.11289 | 0.01598 | 326.20 | 119.99 | 2.459e-03 |
| 16 | 0.00758 | 0.01556 | 326.20 | 119.85 | 8.517e-04 |
| 32 | 0.00756 | 0.01563 | 326.20 | 119.85 | 5.661e-04 |

Layer 1 的全并行时间包含 `vmap/jvp` 首次启动开销；预热后全 576 并行约为 7.6 ms/层，64-token 分块约为 15.6 ms/层。小的输出差异来自 FP16 下不同 batch/kernel 的舍入顺序。

## 5 图描述性结果

| 标签 | I | R | S | D | P entropy | P Top-32 mass |
|---|---:|---:|---:|---:|---:|---:|
| real | 2.7255 | 1.0399 | 0.4687 | -0.1968 | 0.7887 | 0.5120 |
| hall | 3.4883 | 1.2228 | 0.4871 | -0.1933 | 0.7845 | 0.5175 |

这里只是数值/显存 smoke；5 张图的 Real/Hall 均值不能作为统计结论。
