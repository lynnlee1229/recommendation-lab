# 推荐深化：真实数据上的三组对照

复用Retailrocket冻结时间切分、Dataset.examples因果前缀、retrieval.topk/metrics、现有Jupyter与FastAPI工作台。实验写入artifacts/experiments/recommendation，完成结果不覆盖。共享receipt协议由reclab/experiments.py定义，domain为recommendation/search/ads；任务结果保留独立指标含义，不能跨任务直接混比。

## 实验协议

- 固定12000个训练样本、history20、dim24、4轮、Adam lr=.003，种子42/2026，CPU4线程。统一样本索引、每轮批次顺序；批次顺序随机源与负采样随机源分离。
- 5臂：mean+uniform/popularity^.75/hard，以及GRU+uniform、带位置注意力池化+uniform。历史编码后与同一物品embedding表归一化点积，BPR损失。没有用户ID特征。
- 每个负例排除当前目标和因果前缀，hard从16个提议候选中取模型最高分。未观测行为不是未点击曝光，假负例仍可能存在。popular提议使用训练窗口总体频率，属于离线训练分布，不宣称逐时刻在线频率。
- GRU移除左padding再pack；attention屏蔽padding。没有声称完整复现SASRec/DIN。结构间参数量不同，不能把等维度说成等参数量。
- 只用原完整可评估339位验证用户，全库排序，过滤训练已见，报告Recall/NDCG@20、覆盖率、推荐商品训练频率。未读取保留test窗口；已公开验证集属于开发基准，非新盲测。
- 同一模型打乱非padding历史顺序，观察敏感性；不把打乱干预当作因果效应估计。
- 在mean+uniform上复用12簇粗索引，probes1/4/12，报告精确Top20重合率、扫描量、真实标签指标和实测时间。12簇全扫应回到精确检索。小词表上不预设ANN加速。

## 使用

`python -m reclab.deep_recommendation`产生新实验；16章只读已完成记录、绘图并核验。工作台“共享实验记录”展示结果。失败/未收敛/负结果同样保留。每次只改一个变量，再新建实验，不覆写原双塔或竞赛快照。

2种子、短训练只是学习对照，不支持统计显著或通用模型优劣结论。后续可以扩大样本/多随机种子、匹配参数量和收敛预算，但不能为了“更好看”重复查看并调优audit或test。
