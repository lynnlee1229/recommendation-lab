# WSDM Cup 2022 XMRec：本机复现说明

这轮用真实推荐竞赛和公开冠军代码学习“模型分数如何成为特征，再由排序模型融合”。入口是 [11_XMRec_Start.ipynb](../notebooks/11_XMRec_Start.ipynb)，不是新增的展示平台。所有训练、特征和指标都有可修改的 Python 实现。

## 来源与任务

- [官方任务、数据协议及排行榜](https://xmrec.github.io/wsdmcup/)；[官方 starter](https://github.com/hamedrab/wsdm22_cup_xmrec) 固定版本 `f5b1f087a4054537cd8f46bf346f978094b8467e`。
- [OPDAI 冠军源码](https://github.com/opdai/wsdm2022-xmrec-top1-solution) 固定版本 `8af3260368b1980a85a555fc470caa84bf6697d6`。
- [作者论文](https://arxiv.org/abs/2203.00897)，重点读 §3.1、Table 4 和 §3.3。官方榜冠军账号 young_simple，公开代码使用团队名 OPDAI，隐藏测试 nDCG@10=0.6773。
- 原始文件逐项 SHA-256 和原代码校验值见 [xmrec_sources.json](xmrec_sources.json)。22 个文件从官方固定版本下载，并与冠军仓库中同名数据逐项核对一致。
- 数据只用于本地研究学习；原数据及原源码保存在忽略目录，不上传或重新发布。官方数据说明允许研究用途；不替原作者重新授予许可。

源市场 s1/s2/s3 与目标市场 t1/t2 共享商品，用户跨市场相互独立。每个目标用户有官方给定的 100 个候选，其中验证集已知一个正例。模型任务是给这些候选排序。这里的前置模型做的是打分特征生成，不能把候选命中率说成全库召回率。

没有时间戳、商品文本、国家语义标签或曝光日志，不能构造真实“近 7 日热度”、解释匿名市场为某个国家，或把输出当线上 CTR。5core 正反馈与原评分合并时，将其评分设为 5，按 user-item 去重，先保留 train 的记录，与冠军的处理顺序一致。原 train 确有重复 user-item，原始统计保留重复，构图语料去重。

## 可以复现到哪一层

| 内容 | 本轮实现与证据 | 与冠军完整配方的差异 |
|---|---|---|
| 数据 | 五市场完整文件；未对子集采样；主语料去重后 1,176,863 条 | 特征语料不含 t1/t2 的任何 qrel，冠军部分市场组合加入另一目标市场 qrel |
| ItemCF | icf_v2 IUF 与 icf_v3 评分余弦的稀疏移植；无邻居剪枝、无历史截断 | 选取 local、s1+target、s3+target、all5 等组合，未穷举全部组合 |
| 数值对照 | 直接提取校验过的原类；真实训练历史逐对分数核对，误差 <2e-5 | 对照是公式/实现一致性，不代表整个原始脚本环境被重建 |
| 图模型 | 原 LightGCN 的归一化传播、逐层平均、BPR、ego-L2 和梯度对照 | 本机 32维/2层/12epoch×3更新/32768批次/lr=.01/无dropout；原论文 2048维/4层/lr=.001/dropout=.4 |
| 对照模型 | 同预算的 0 层 MF；可切换均匀或 popularity^.75 负采样 | 小尺寸模型与短训练不代表图模型最佳能力；正式首轮统一使用均匀负采样 |
| 特征 | 21 列：五市场 train/core 的商品计数、目标市场用户/商品评分统计、5路CF分数、MF与图分数 | 未复现原文 206/147 列、Word2Vec/Node2Vec/NCF/Swing/LLR/BiGraph 的完整组合 |
| 融合 | 每个目标市场一个 binary LightGBM，保留冠军的分类目标 | 用固定用户分组代替原 10折行级 bagging；不声称 LambdaRank |
| 选型 | 用户稳定哈希约60%fit、20%select、20%audit；在select上冻结模型选择后才计算audit | 缺少公开隐藏测试标签；没有线上提交或复现0.6773的声明 |
| 特征分析 | 同数据消融、相关矩阵、gain与单预测TreeSHAP | 未做原完整 covariate-shift/null-importance 多轮筛选 |

所有候选、验证正例和数据规模都保留。神经模型的维度和更新预算缩小；“完整数据”不等于“完整冠军方案”。

## 首轮真实运行结果

运行：`20261029T091104737318-champion-core`。固定 CPU：图训练最多4线程；LightGBM **1线程**。实际耗时 77.38 秒，主进程峰值 RSS 1,422.92 MiB（约1.39 GiB；不是整台机器峰值）。先前两次失败尝试保留在 artifacts 中，未写入 latest，不作为成功证据。

目标市场完整官方验证用户共 8,179：t1=2,697、t2=5,482。split用户数：t1 fit/select/audit=1,616/538/543；t2=3,267/1,153/1,062。

ItemCF 不使用任何目标 qrel 训练，可直接评估完整官方验证集：

| 单目标市场 ItemCF-IUF | 本地实际 nDCG@10 | 论文 Table 4 报告 |
|---|---:|---:|
| t1 | 0.677494 | 0.6776 |
| t2 | 0.558946 | 0.5589 |
| 合并所有用户 | 0.598037 | 0.5980 |

合并分数对齐到论文报告精度；t1有约0.000106的残差，未宣称逐位相等。本地相同分数时用 item ID 确定顺序；原方案与官方的同分处理不同。数值精度/同分次序可能产生小差异，未把原因当已证实结论。

以下是**同一批1,605个audit用户**的对照，不能与冠军隐藏测试直接比较：

| 模型 | nDCG@10 | HR@10 |
|---|---:|---:|
| 目标商品交互次数 | 0.340036 | 0.537695 |
| 目标市场 ItemCF | 0.610977 | 0.694081 |
| 全市场 IUF | 0.626545 | 0.717134 |
| 全市场评分余弦 | 0.628597 | 0.720249 |
| 小尺寸 MF | 0.103664 | 0.188785 |
| 小尺寸 LightGCN | 0.465419 | 0.603738 |
| LightGBM 仅统计 | 0.394811 | 0.602492 |
| LightGBM 统计+CF（按select选中） | 0.647982 | 0.745794 |
| LightGBM 再加图模型分数 | 0.648567 | 0.743925 |
| 相同完整特征，fit仅留9个负例/用户 | 0.646032 | 0.743925 |

选择依据仅为 select 合并 nDCG：stats=.381625、memory=.635465、all=.634710、neg9=.632895，因此选 memory。audit 上 all 的微小优势没有用于改选。新图模型有可学习信号，但在本轮预算下作为单模型较弱，作为特征也没有明确选型收益。不能仅凭loss下降判定效果。

`metrics.csv` 的 `all_official_valid` 对于 LightGBM **包含训练过的 fit 用户，不能用作泛化分数**；只用于追踪。报告及 Notebook 的模型效果对照使用 select/audit。audit 发布后已成为可见基准，后续多轮调参不再把它称为未见测试。

## 学习顺序与代码修改

1. [11 题目与数据](../notebooks/11_XMRec_Start.ipynb)：检查文件、去重、候选和用户分区。
2. [12 特征](../notebooks/12_XMRec_Features.ipynb)：手算IUF、跑原码对照、拆开真实候选的相似度贡献。
3. [13 采样与图模型](../notebooks/13_XMRec_Graph.ipynb)：看负例分布、传播/梯度，实际重训真实t1的小模型。
4. [14 排序](../notebooks/14_XMRec_Ranking.ipynb)：实际重训LightGBM，改特征列表、叶子数或负例数量，解释预测。
5. [15 评估](../notebooks/15_XMRec_Evaluation.ipynb)：手算指标、市场加权、配对bootstrap、失败案例和后续实验。

模型实现：`reclab/competition/memory.py`、`graph.py`、`ranking.py`；数据和实验：`data.py`、`experiment.py`；指标：`metrics.py`。工作台“06 代码与 Notebook”可查看源码、进入原生 Jupyter 编辑。Notebook 中的 reload 格使 .py 修改在当前实验生效，改模型后需重新训练。

Notebook13/14的沙盒训练保存在 `artifacts/xmrec/experiments/`，主实验是独立带时间戳的目录。旧 Retailrocket 的 `artifacts/runs`、data/processed 和00~10内容保留；00只追加入口。原 `lakehouse-lab` 未启动或修改。两个项目均无需新增Pod或容器。

## 可重复命令

在 `/Users/lizuolin/local/recommendation-lab`：

```bash
uv sync --frozen --group dev --group notebooks
.venv/bin/python -m reclab.competition prepare
.venv/bin/python -m reclab.competition run --tag my-first-ablation --dim 32 --layers 2 --epochs 12
.venv/bin/python -m reclab.competition results
.venv/bin/python scripts/execute_notebooks.py --only 11_XMRec_Start 12_XMRec_Features 13_XMRec_Graph 14_XMRec_Ranking 15_XMRec_Evaluation
```

默认使用完整官方数据并新建run。隐藏测试候选的预测写到主run内 `test_<model>.parquet`，无公开答案，因此不报测试分数，也未向外部竞赛平台上传。

macOS 需要 libomp；这台机器已存在 `/opt/homebrew/opt/libomp/lib/libomp.dylib`。实测混用 PyTorch/LightGBM 多线程运行库会在数据集构造处卡住，因此 LightGBM 固定单线程并用混合运行回归测试约束，不设置绕过运行库检查的环境变量。
