# All-attention、B1、B2 双机实验

**2026-09-13全量复核更正：运行均已结束，但B2 K4数值验收失败（四模型合计121076/1622248 target-layers纯积分误差>1%）。原50-case审计遗漏问题层，B2检测/机制结论暂不能当作已验证结果。见[正式分析报告](ALL_SOURCE_PATHS_RESULTS_20260913.md)。All-attention和B1全量积分检查通过；本次仅分析，未重训。**

## 固定协议

本机GPU0/1分别运行Qwen2.5-VL-7B、LLaVA-1.5-7B；32678 GPU0/1分别运行Qwen3-VL-8B、InternVL2.5-8B。两端共享项目、数据、模型和vicr环境，每模型独占输出锁。运行环境为 `/opt/conda/private/envs/vicr/bin/python`。

一次原生精度clean capture，局部Norm/FFN提升FP32且禁用TF32。All-attention以 `z-sum(a_m)` 为基线；Visual-only用相同视觉write计算组响应。B1为真实 `b_O+alpha*(r+A_P+A_V+A_G)` 的Norm+FFN路径；B2冻结真实端点RMS缩放，只积分纯FFN的 `alpha*n`。B1、B2均进入正式500/4000图，分别对照，不联合拼接。

原生成文本、目标、全部decoder层、共享500图400/100划分、原4000图3200/800划分保持不变。机制主曲线及同图配对均排除标签冲突目标；训练保留全部mentions。prompt含BOS/模板/特殊token；generation只含可见前缀，排除目标和未来位置。当前N_G与response_index相等，长度控制保留两列但不是两个独立变量。

每模型按标签和生成位置五分位选10目标，每目标5层，共50case，种子20260912。All-attention、Visual-only、B2审计K4/8/16/32/64并选择四模型共同通过的最小K。JVP后端一致性比较每条D维方向向量，rtol1e-4/atol1e-6。纯求积闭合及相对K64响应误差≤1%，有效方向余弦差≤0.01；源重建误差单列，不能靠提高K或补齐虚构来源掩盖。

B1另保留普通GL K4/8/16/32/64/128对照。正式积分用alpha=s*sinh(t)，s=sqrt(epsilon)/RMS(S)，包含变量替换权重；每段GL16/32自适应二分，rtol1e-4、atol1e-7、最大深度12。组响应及独立径向响应共同验收，FP64累加。零bias使用严格等价稳定公式 `J_RMS(alpha*S)a = w*r*(a_perp + epsilon*r^2*a_parallel)`，避免径向导数中的大数相减；保留epsilon，不使用近似尺度不变性。非零bias仍使用通用真实路径。未收敛报错停止，不删除B1。

正式逐图文件仅保存token级范数/余弦/位置/来源类型、组级标量和诊断；审计文件可包含向量。逐图原子保存，协议及层数校验后续跑；500图结果直接复用于4000图。当前为显存稳健起见使用1目标块（不超过计划4），不额外做整模型前向。

## 检测和统计

固定15组：F0、F1_raw、F1_log1p、F2；B1/B2各自F3_raw、F3_log1p、F4、F5、F6；length。F3为四来源贡献范数和D_R，F4为7个残差几何量，F5=F2+F4，F6=F0+F5。log1p仅作用于非负norm/gross/net，gain/share/几何量原值保留。NaN在机制统计中报告有效数量，在检测矩阵中固定置零，不新增缺失特征。

固定MLP [128,64,32]、BatchNorm、dropout0.3、Adam、lr0.001、weight_decay1e-5、batch256、至多100epoch、seeds43/44/45；不标准化，沿用训练损失调度/早停/checkpoint，不以测试集选模型或特征。F0及旧source-strength raw/log1p基线须与原mentions、标签、划分严格对齐后复用。

报告AUROC、HALL-AUPR的逐seed和三seed均值/总体标准差。固定21个配对比较，每个指标10000次原800测试图片簇重采样，重复图片保留全部mentions，种子20260912；先计算各seed指标再取均值，不用集成概率替代。机制分析仅描述，不据此挑选检测特征。

## 入口与进度

主入口 `scripts/run_ffn_all_source_paths.py` 提供 audit/select/extract/pipeline/mechanism/train/bootstrap/summarize。每模型pipeline自动经历审计、等待全部四模型审计、500图机制、4000图提取、训练、bootstrap。结果在 `outputs/ffn_all_source_paths_v1`。

```bash
/opt/conda/private/envs/vicr/bin/python -u scripts/run_ffn_all_source_paths.py --stage pipeline --model qwen2_5_vl_7b --device cuda:0
/opt/conda/private/envs/vicr/bin/python scripts/watch_all_source_progress.py --machine local
```

另三模型使用相应模型名与设备；远程通过ssh32678进入同一目录。任务在screen中运行，退出SSH不会中止。重复启动同模型会被锁拒绝。

根目录 `EXPERIMENT_PROGRESS_LOCAL.md` 与 `EXPERIMENT_PROGRESS_32678.md` 每10秒原子覆写，仅含机器/UTC更新时间和两个模型的进度。监控显示当前阶段、已完整保存图片数或已训练heads数、epoch、状态与简短错误。心跳超过120秒显示无更新；所有任务终态后保留最终文本。日志独立保留在outputs，进度文件不追加历史。

## 验证与失败记录

- 新核心数学/恢复7项、分析6项、进度5项测试通过；旧composition4项回归通过。检测epoch callback为可选参数，未传入时旧训练行为不变。
- 首次真实audit在B1后端逐坐标近零值比较中失败：仅2–3个坐标的绝对误差约1–4e-6。改为与归因定义一致的逐方向向量范数容差检验，容差数值未放宽。原数据保留于 `outputs/ffn_all_source_paths_attempt01_coordinate_parity`。
- 第二次真实audit发现零bias径向响应的FP32消减误差，自适应细分到depth12仍失败。引入上面的等价稳定径向公式，未放宽积分容差。原结果保留于 `outputs/ffn_all_source_paths_attempt02_radial_cancellation`。
- 完成状态以进度文件、各模型audit.json/逐图shards/detection.json/bootstrap.json为准。该文档描述协议，不将已启动任务视为已完成实验。

第三次四模型各50case真实审计全部完成，正式共同积分点数为All-attention K32、Visual-only K4、B2 K4；B1冻结上述稳定自适应算法。数值汇总位于outputs/ffn_all_source_paths_v1/audit_summary.json。下表均为50case最大相对闭合误差，不是全4000图结论。

| 模型 | B1稳定自适应 | B1普通GL K64 | 后端向量相对误差最大值 |
|---|---:|---:|---:|
| Qwen2.5 | 2.82e-6 | 0.879 | 1.81e-6 |
| LLaVA | 2.04e-6 | 0.0368 | 1.60e-6 |
| Qwen3 | 3.59e-6 | 0.941 | 1.98e-6 |
| InternVL | 2.22e-6 | 0.346 | 7.71e-6 |

稳定自适应最多使用2个接受区间（包括估误时最多144个节点评估）；纯径向求积闭合最大值不超过7.84e-7。四模型已进入500图正式提取。当前共有30项定向测试通过，尚无正式检测结论。

全部完成后 `--stage summarize` 输出跨模型summary.md、检测/逐seed/bootstrap CSV和机制汇总CSV；各模型机制图为PNG/PDF。

本机screen `allsource_final_summary` 已运行分析入口的 `--stage summarize --wait`，等待四模型bootstrap完成后自动汇总；详细日志 `outputs/ffn_all_source_final_summary.log`。bootstrap阶段另写精简计数，由进度监控显示，不影响worker心跳。
