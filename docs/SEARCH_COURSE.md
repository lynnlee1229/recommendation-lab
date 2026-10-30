# 搜索完整核心学习路径（S01–S18）

先修公用31–34；本轮新增35–46十二本，不修改旧35本。搜索现在19本，全Lab47本。六模块顺序为需求/相关性 → 查询处理 → 召回 → 相关性推理与训练/排序 → 查询推荐 → 系统复盘。17综合章后置参考；编号不是教学先后。

|阶段|单元|运行章节|
|---|---|---|
|任务、满意度、分级判断与真实qrels|S01–02|35 → 25|
|规范化、中文切分、字段词权重、类目/意图与改写|S03–06|36 → 37 → 38 → 39|
|位置/布尔、BM25、语义、近似预算、缓存|S07–11|40 → 26 → 27 → 28 → 41|
|BERT输入/推理/微调、排序loss、融合与多目标|S12–15|42 → 43 → 29 → 44|
|前缀与相关查询、端到端追踪和正式误差|S16–18|45 → 46 → 30；17综合参考|

工作台“搜索 → 学习路径”按此排列；“讲义与路线”逐单元显示来源、入口、证据类型与局限。来源复核见search_completion_sources.json；历史缺口审计保留在SEARCH_LEARNING_ROADMAP.md和COURSE_AUDIT_SOURCES.json，完成证据以SEARCH_CURRICULUM_COMPLETION.json为准。

35–41、43–46以明确人工构造讲机制，36使用中文词例；42复用真实英文SciFact训练分区做TinyBERT全参数微调。各章有预测、公式/契约、单变量对照、可执行断言、解释与练习。规则意图/改写、粗簇索引、合成查询日志分别保留边界，不称生产系统或完整论文复现。

42章每次Run All创建独立workshop run，详见SEARCH_TRAINING_PROTOCOL.md。冻结与微调在固定32/8训练/选择query和BM25候选上比较；选择指标没有改善，保留负结果。300已见测试不参与新配方选择。教学checkpoint与预测只留本机，同一registry记录，正式latest不变。

46章使用同一FastAPI的POST /api/search/workshop/trace；响应明确constructed mechanism、四文档、规则语义替身、每请求独立缓存。跨请求TTL/版本实验在41及46章Notebook里运行；不把该替身说成原SciFact链路。正式/api/search五方法保持原样，30章保留真实离线/HTTP和计时检查。

```bash
.venv/bin/python scripts/execute_notebooks.py --only 35_Search_Needs_Judgments 36_Search_Chinese_Tokenization 37_Search_Term_Fields 38_Search_Intent 39_Search_Rewrite 40_Search_Positional_Index 41_Search_Candidate_Cache 42_Search_BERT_Finetune 43_Search_Ranking_Objectives 44_Search_Quality_Diversity 45_Search_Query_Suggestions 46_Search_Trace_Review
```

## 旧六章实现说明（历史保留）

# 深入搜索：六章学习路径

保留17综合章与既有正式实验，新增25～30六章；不重建用户旧Notebook。与广告19～24相同，先预测、检查输入/形状、只改一个变量、实际运行、解释局限并留练习。统一课程目录提供31本入口。

|章|主要问题|实际执行内容|范围|
|---|---|---|---|
|25 任务与qrels|标签契约、宏平均与不完整判断|跨表引用检查、手算NDCG、多正例宏/汇总对照、构造补判断|完整真实语料与300查询记录；反事实标签仅本地副本|
|26 倒排与BM25|词频饱和、IDF、长度归一化、OOV|稀疏倒排检查、饱和曲线、b=0/.75/1真实检索、重复/OOV边界|前30训练查询，完整5183文档|
|27 语义编码|token形状、mask pooling、归一化、截断|真实编码与手工池化一致性、64/256截断、BM25/dense逐查询对照|80文档长度样本、8文档截断；40测试查询诊断|
|28 近似索引|候选预算、近邻一致与任务召回|12簇probes=1/2/4/12、实测索引延迟、全簇逐文档等价|40测试查询诊断，完整文档向量|
|29 融合与训练|分数标度、候选上限、查询级切分|RRF深度、160/40训练/选择、BM25单特征/五特征真实拟合、TinyBERT候选重排|30训练查询融合；正式200训练查询内分组；40测试诊断，3查询交叉编码示例|
|30 误差与服务|均值后的输赢、召回与降位、计时边界|全300查询差值、8失败候选核查、5次HTTP耗时、五方法前10逐文档比对|既有正式run，单查询服务一致性|

## 阅读与执行

按25→26→27→28→29→30，先看每节预测问题再执行代码。每本5～6个短代码格，共32格；所有代码都使用同一`reclab/search.py`和共享指标/粗索引，教学辅助集中在`reclab/search_lessons.py`，源码可从统一工作台白名单查看。没有新增服务器、数据副本或Lab。

先按README配置环境与搜索数据/正式实验，再启动已有工作台：

```bash
.venv/bin/python -m reclab.search --download  # 仅没有正式实验时需要；每次显式调用新建run
./lab.sh up
.venv/bin/python scripts/execute_notebooks.py --only 25_Search_Task_Qrels 26_Search_BM25 27_Search_Semantic 28_Search_ANN 29_Search_Fusion_Ranking 30_Search_Errors_Service
```

30章直接调用18580已有服务；不启动第二台服务器、不停止Jupyter。29章每次Run All实际训练两组模型，创建新的search教学run，按config.workshop隔离；latest服务只读取正式run，共享实验记录分别渲染select与正式测试。语义查询向量缓存放本机artifacts/search/lessons，按父实验ID和查询文本摘要隔离。练习不会覆写已完成正式实验、qrels或服务权重。

## 明确的评估边界

新章节是原固定实验的教学展开。原300查询已经被查看，新40查询对照属于已见测试诊断，不能冒充全新盲测或拿来选择生产参数。BM25/RRF的参数扫面使用训练查询；29章scaler只fit训练查询的候选，不能按候选对随机拆分查询。训练弱负例来自未判断文档，不是真实负判断；sigmoid分数不是校准相关概率，也不是CTR。

模型、文档、标签与原论文边界沿用[搜索协议](SEARCH_EXPERIMENTS.md)。27章神经模型冻结；截断实验先只观察表示变化，不直接宣称效果改善。28章计时只测索引，30章HTTP墙钟与服务内部计时分别记录，不能与原端到端warm延迟混用。新fit与诊断保留正负结果，不替换原正式排名或宣称线上收益。

广告对话在234bb4a提交并明确停止写入后，本搜索对话才接手；两轮修改串行，共享同一checkout与Git历史。旧17/18综合章与广告19～24均保留。
