# 同一系统的真实搜索迭代

搜索复用 `reclab/experiments.py` 的 append-only begin/finish/catalog/result、`reclab/retrieval.py` 的稳定 topk 与二值 qrels 指标、`reclab/learning/retrieval.py` 的粗索引、公开下载校验器、FastAPI、本机工作台07和 Notebook 目录。任务 adapter 为 `reclab/search.py`，没有新增仓库或服务器。

## 来源、资源与可复现性

- 原始任务：[SciFact / AllenAI](https://github.com/allenai/scifact)，科学声明查找论文证据；使用发布者托管的 [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) 导出与 [qrels](https://huggingface.co/datasets/BeIR/scifact-qrels)。发布页许可 CC BY-SA 4.0，原始文章可能有自己的条款。仓库不再分发全文、查询或逐查询预测。
- 编码器：[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，Apache-2.0，6层、384维，safetensors 90,868,376 字节。
- 重排器：[ms-marco-TinyBERT-L-2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2)，Apache-2.0，2层、128维，safetensors 17,552,980 字节。训练领域是 MS MARCO，不保证适合生物医学。
- `docs/search_sources.json` 冻结4个发布仓库的40位提交、20个文件的原始HTTPS地址、大小、许可与SHA-256；下载后建立本地校验基准，SHA不是独立的来源真实性证明。总输入114,658,540字节（109.35MiB），模型与数据分别在忽略的 artifacts/search 和 data/search。
- 只加载本地 safetensors，不下载远程代码。依赖在 pyproject.toml/uv.lock。CPU最多4线程，编码批16、最多256 wordpieces；无GPU要求。神经模型冻结，不声称大模型训练或完整论文复现。

```bash
.venv/bin/python -m reclab.search --download
.venv/bin/python scripts/execute_notebooks.py --only 17_Search_Experiments
```

每次运行创建新实验；完成记录不覆盖，失败记录保持failed。下载校验失败即停止。默认读取最新未标记superseded的完成实验。首次运行构建语料向量，重跑也实际重新编码，以保存独立可检查产物。已有推荐模型不覆盖。

## 固定协议

完整5,183篇文档，以标题+摘要作为文本；全300测试查询、339条正qrels。训练集809查询/919正qrels，按查询ID字典序取前200个训练查询，从各查询融合前20候选得到4,000训练对。线性基线真实训练，特征为BM25、余弦、RRF、查询词覆盖和log篇幅；StandardScaler+LogisticRegression，C=1、seed42、不按测试选参。训练候选未判断即作弱负例，可能有假负例。训练/测试ID不重叠，但相关声明、相同论文及预训练语料重叠不能排除。

- BM25：k1=1.2、b=.75，lowercase单词tokenization，二值查询词频，无停用词/词干，语料ID稳定排序。复用topk时内部0保留、真实文档从1编号，避免漏掉第一篇。
- Dense：MiniLM attention-mask mean pooling+L2归一化，精确余弦排序。
- Hybrid：两个通道各前100，RRF常数60，取融合前100。未进候选并集的文档不得补位。
- Linear / TinyBERT：同一融合前20内重排；候选集外不可找回。两种重排的Recall@100实际是最多20结果的召回，因此与其他方法前100预算不同；请结合候选Recall@20解读。
- Coarse：同一384维向量，用共享教学kmeans分12簇，探测2/12簇。额外报告与精确前10交集比例及候选数。全簇必须与精确排序一致。

指标使用原始正qrels的二值相关性，按查询宏平均 MRR/NDCG/Recall/Hit@10、Recall@100。所有正qrels文档均在完整语料内，不缩减候选库。未判断文档按不相关评估；本导出没有穷尽负判断，known_positive_fraction@10只表示已知正例比例，不代表完整判断覆盖。服务任意新查询没有qrels，不能展示虚构质量指标。

延迟实测逐查询CPU墙钟P50/P95，含tokenization、查询编码、检索与排序、对应融合和重排；不含离线语料编码/模型加载、文件读取、HTTP和网络。混合方法顺序执行、warm模型，不是生产吞吐或严格微基准。服务返回的计时口径略不同，且首次HTTP会额外加载模型，不能直接与离线表格横比。

## 使用与教学

- 同一工作台 `http://127.0.0.1:18580` 的07查看两领域独立结果表，可输入英文声明进行五种方式的实际检索。`POST /api/search` 接受query、method、limit，返回真实文档、排序分数、当前run_id；分数不代表科学声明成立概率。
- 统一 `/api/experiments/search/{run_id}`，源码白名单 `/api/code/search`；18本Notebook统一目录。第17章依次预测、查看真实对照、只改融合深度机制、检查失败，重跑训练需显式调用。
- 本机逐查询结果、完整延迟列表、线性系数和向量在忽略的实验目录；精简聚合结果提交到results。失败案例按TinyBERT相对BM25的NDCG降幅选8个，保留负结果，解释只作为假设。

本任务只代表小型英文科学证据检索；有限标注、文本截断、预训练重叠及领域不匹配限制结论，不提供医学判断或线上收益证明。


## 深入课程追加

用户要求拆开讲细，已保留17总览并追加25～30六章，共32格。详见[深入课程](SEARCH_COURSE.md)。本轮新增教学模型按config.workshop隔离，不替换正式搜索服务；原7组固定实验仍为历史正式结果。
