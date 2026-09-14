# raw_AE_top32 幻觉检测补充实验

## 做了什么

用户要求验证全分解汇总曲线中的 `raw_AE_top32` 是否能用于幻觉检测。直接读取四模型原4000图缓存，构造全层Top32重合率，训练单独信号和拼接B两组，每组三种子，共24个新检测头。B引用已有三种子结果，没有重跑模型前向或路径积分。

本次使用原3200训练/800测试划分和全部mentions，与此前P_E的JS/OT检测对照保持一致。它是旧82协议的补充，不采用后续811实验的3200/400/400划分或单隐藏层搜索设置，不能与811表中的数值直接横比。

## 信号与特征

对每层、每个目标，令r_m为视觉位置m的跨头平均原始attention，P_raw=r/sum_visual(r)，T为已有的语义gate加权后归一化证据分布。

O_l = |Top-k(P_raw) ∩ Top-k(T)| / k，k=min(32,视觉token数)。

O越大，表示加入gate前后选中的高权重视觉位置越一致。它不包含e_m的空间分布，不是attention总量，也不是幻觉概率。它忽略权重差和位置之间的空间距离。排序沿用旧绘图代码的NumPy argsort，权重并列时也保持相同处理。

- `raw_AE_top32`：O的所有层拼接，L维。
- `B_raw_AE_top32`：[AE全层, log1p(I)全层, log1p(S_E)全层, O全层]，4L维。
- `B`：上述前三块，3L维，直接引用旧检测结果。

Qwen2.5/LLaVA/Qwen3/InternVL分别为28/32/36/32层。O不取log、不标准化。S_E=sum_visual ||e_m||，来自此前conditional路径；不替换为完整来源分解的S_C。

## 训练与核对

复用原Torch MLP：隐藏层128/64/32，BN、dropout 0.3，Adam学习率0.001、weight decay 1e-5，batch256，最多100epoch。调度、早停和checkpoint沿用训练loss；没有新增验证集或参数搜索。seeds43/44/45，先逐seed计算指标再报告均值±总体标准差。HALL-F1使用各seed训练集REAL-F1阈值，同时保存固定0.5结果。

合成数据核对Top32全部相同、完全不重合、重合一半三种情况，并与旧分布比较函数一致。全量按旧mention顺序对齐，核对原图片划分；提取的逐层REAL/HALL均值与旧curves.csv对照。只做这些直接影响实验正确性的检查。

## 结果与运行位置

24个新头均已完成。四模型单独O的AUROC均值依次为64.14%、74.53%、68.19%、68.97%，明显低于各自B。B+O相对B的AUROC/HALL-AUPR均值变化（百分点）：Qwen2.5 −2.98/−4.54，LLaVA −0.54/−1.30，Qwen3 −0.73/−0.13，InternVL +0.06/+0.58。前三模型没有增益，InternVL仅小幅变化；本设置下不支持将它作为B的通用增强特征。Qwen3的HALL-F1增加2.53点，但AUROC/AP下降，属于指标间取舍。

合成公式核对、四模型全量曲线均值核对均通过，均值差均为0。四模型训练/测试mentions分别为6985/1732、12317/3146、11846/3027、9378/2381，共50,812。保存24个新头、12行含B的汇总、72行逐seed双阈值结果及12行配对差；两训练进程和汇总进程正常退出，脚本编译与git diff --check通过。

完整均值、标准差、配对增量见[实验结果表](../outputs/ffn_source_composition_v1/raw_ae_top32_detection/summary.md)。逐seed与双阈值见同目录seed_metrics.csv，拼接相对B的逐seed差见paired_deltas.csv。各模型目录保存matrices.pt、protocol.json、heads及detection.json；后者保留旧工具的ensemble字段，但本报告不以ensemble为主结果。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/conda/private/envs/vicr/bin/python scripts/train_raw_ae_top32.py --models qwen2_5_vl_7b qwen3_vl_8b --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/conda/private/envs/vicr/bin/python scripts/train_raw_ae_top32.py --models llava_1_5_7b internvl_2_5_8b --device cuda:1
/opt/conda/private/envs/vicr/bin/python scripts/train_raw_ae_top32.py --summarize
```

原800测试图已经用于多轮探索。这次结果描述既有数据上的检测效果，不作显著性或新数据泛化声明。曲线分离与检测增益分别报告。
