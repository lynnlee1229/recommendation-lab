# 同一系统的真实广告点击与模拟决策

广告 adapter 为 `reclab/ads.py`，共用 `experiments.begin/finish/catalog/result`、`download_public_assets.fetch`、FastAPI、工作台“共享实验记录”、源码白名单及 Notebook 目录。新增 `reclab/probability.py` 供二分类概率任务复用；AUC/logloss/Brier与检索NDCG分开。独立生成市场的 `reclab/auction.py` 可供其他预算分配练习复用，没有新增服务器或仓库。

## 真实来源与资源

原发布者 [Criteo AI Lab](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) 明确每行是一次真实展示及其点击标记，包含13个匿名数值与26个匿名类别特征；正负样本经过**不同比例抽样**。缺失值和未披露字段语义是数据本身的限制，未把推荐未交互当作负曝光。

采用发布者 [criteo/CriteoClickLogs](https://huggingface.co/datasets/criteo/CriteoClickLogs) 的官方 Parquet。发布页标注 CC BY-NC-SA 4.0；仅用于本地非商业学习，不在Git再分发数据，使用者仍需遵守原条款。固定提交 `e11d69ae913b16cdd7387706a2133def0fdc6ced`，选2015-02-15的part-00079，47,872,279字节（45.65MiB），766,864行。只读取前200,000物理行（约该分片26.08%，不代表全量43.7亿行）。分片选择在训练前固定，不按效果选数据。

`docs/ads_sources.json` 记录固定URL、许可、字节数、SHA-256；SHA采用官方API返回的LFS对象摘要并实际核对。原始日文件有时间顺序，但不假定转换后的Parquet分片保留完整全局时间顺序。无账户登录、额外依赖或GPU；沿用uv.lock和已有环境，最多4线程。

```bash
.venv/bin/python -m reclab.ads --download
.venv/bin/python scripts/execute_notebooks.py --only 18_Ads_Experiments
.venv/bin/python scripts/verify_ads.py
```

首次下载显式可选，原setup预算不增加。已有输入仍核对SHA；每次训练创建新run，完成记录不覆盖；异常保留failed receipt。

## 固定训练与评估协议

先对39个特征做不含label的pandas uint64行哈希，取模10000：0~5499 train，5500~6999 select，7000~8499 calibration，8500~9999 audit。重复特征向量进入同一分区。不是按用户或时间切分；用户ID没有披露，相关样本和哈希碰撞不能完全排除。保存行分区供本地复核，不用audit调参。

本轮各区109,935 / 29,984 / 30,153 / 29,928行，点击3,614 / 1,016 / 1,023 / 992。全样本仅发现1条重复特征行，无冲突标签。编码、缺失填补、缩放、词表均仅拟合train。

- 常量先验：train点击率。
- 可解释线性模型：13数值保符号log1p、train中位数填补与缺失指示、标准化；26类别每字段最多128个onehot类别，稀有与未知按sklearn规则处理。LogisticRegression，C=1，最多300轮；完整系数CSV只在本机。
- 特征树模型：原数值及每字段train最常见255类别，其他/未见=0。LightGBM，15叶、学习率.05、每叶最少100条、L2=1，80/160棵候选；仅select logloss选树数。**这只是树家族内部选型，并未宣称胜过线性模型**。
- 固定选中的树上，另用calibration拟合logit→sigmoid逻辑回归（C=1e6）与isotonic。audit完整报告原始、sigmoid、isotonic；不按audit选校准器。select中的校准指标仅作诊断，因为base已在该区选过。

AUC、二分类logloss（概率裁剪1e-7）、Brier、10等宽箱ECE及各箱曝光数/均值/真实点击率。单类别分段AUC返回null；ECE不是proper scoring rule，不能独立选模型。缺失阈值取train中位数；分段为高/低数值缺失、任一类别在train中未见/全部已见，不能解释成人群属性或公平性。按sigmoid单行logloss保存8个失败曝光，仅留在本地artifact，不上传逐条预测或输入。

配对校准差异用300次IID行bootstrap；未涵盖训练不确定性或样本相关性。计时为实际CPU墙钟；拟合秒不含数据预处理，端到端含读取、特征、评估、产物和模拟，不含下载。结果在 `results/2026-10-30-ads`，完整run `20261030T125738-f82b6582`。

## 实际结果与负结果

| 方法 | audit AUC | logloss | Brier | ECE |
|---|---:|---:|---:|---:|
| train prior | .500000 | .145515 | .032048 | .000272 |
| linear onehot | .721468 | .135423 | .031193 | .004382 |
| GBDT80 | .706329 | .136029 | .031221 | .001738 |
| GBDT160 | .703035 | .137404 | .031358 | .005350 |
| GBDT80 + sigmoid | .706329 | .135968 | .031219 | .001149 |
| GBDT80 + isotonic | .703064 | .136437 | .031302 | .001163 |

线性优于两个树候选，160棵没有改善；更复杂不自动更好。Sigmoid相对原始logloss差-0.000061，95%行bootstrap区间[-0.000170, +0.000063]，跨零，不能确认改善。Isotonic在audit恶化。常量先验ECE最低但无区分能力。高概率箱很稀疏，必须看计数，不能根据连线外推。

本机训练与评估端到端11.72秒；线性拟合4.10秒，树80/160分别.56/.93秒；这不是生产服务吞吐基准。最初一次因sklearn特征名导出接口失败的run保留，修复后新建成功run。

## 独立模拟市场

每seed生成6000请求、3虚构广告主，真实点击率为Beta分布随请求时间上升，模型估计为乘性噪声；点击价值1.5/2/2.5，外部报价均匀.025~.12。按估计点击率×点击价值报价，**单位为每曝光货币，而非CPC或eCPM**。单赢家支付第二高合格报价与外部报价的最大值；外部报价更高则无本系统成交。所有潜在点击一次生成，策略之间共享，实现成对对照。

每广告主预算60，保守要求完整bid可负担才参与；二价支付不能超过bid。pacing额外限制累计花费≤预算×当前请求比例。逐请求断言支付与预算边界；未成交、预算不足、pacing限制都可能导致弃权。均匀pacing不使用未来真实概率或点击。

固定seed41/42/43比较固定曝光报价.12、价值报价、价值报价+pacing、同pacing但出价倍数.5。每个seed的前三组与第四组都报告，不根据模拟结果调参。后半程高价值、所有内部竞价者共同换策略、外部价格固定，均为人为假设。该市场下pacing改善期望价值，半价报价改变成本与曝光分配；不声称truthful最优、真实经济均衡或线上增收。所有收入/点击价值/盈余均为模拟货币。它不是Criteo反事实评估，没有propensity或真实竞价数据。

## 统一工作台与教学

工作台“共享实验记录”按recommendation/search/ads独立渲染，不能跨任务混比指标。广告提供：

- `POST /api/ads/predict`：13数值+26匿名类别，实际加载本机训练GBDT与校准映射；示例来自audit。修改匿名特征不具因果意义。
- `GET /api/ads/example`：本机公开审计输入，不含真实个人标识。
- `POST /api/ads/simulate`：受限预算、出价倍数、pacing开关和固定seed；独立生成市场。
- 共享 `/api/experiments`、`/api/experiments/ads/{run_id}`、源码与Notebook目录。

第18章只读真实结果并做单变量校准/出价对照；训练需显式CLI触发。新增章干净内核执行、格式验证、两图渲染检查；旧章和用户内核保持。GBDT在线预测与离线逐行对照，所有原测试及原推荐/搜索接口回归。

本轮是小预算真实点击任务和明确构造的决策教学，不是全量Criteo benchmark、论文完整复现、线上CTR校准或线上收益证明。


### 本机在线推理兼容性

实际浏览器验收发现，PyTorch工作台进程原生加载LightGBM模型在此Mac上发生崩溃（两个独立诊断进程退出139）；原始训练CLI可以正常训练和导出。服务改为读取本机 `ctr_trees.json`，按原树结构、类别集合、缺失路由与binary sigmoid求值，复用同一训练模型，不另训替代模型。训练器同时保存native text与JSON树导出，二者均不上传。已有完成run仅追加派生JSON导出，原receipt/result不覆盖。30条真实audit输入的raw/sigmoid/isotonic与保存的原生批预测误差均小于1e-10；缺失与类别路由另有数值测试。此兼容层只支持本项目的binary sigmoid:1单树迭代模型，其他目标明确拒绝。


## 19～24深入课程（用户追加的细分迭代）

保留18总览，增加六本独立章节，每本5个短代码格，均包含输入形状、先预测、单变量运行、结果解释和练习。25本课程入口仍在同一目录/API。

- 19：真实40k曝光的标签/缺失统计、label翻转不影响特征分区、构造复制压力测试、train-only词表覆盖。
- 20：真实40k同分区，实际训练数值线性基线与加入类别的线性模型；检查缺失指示shape、权重聚合、固定概率收缩压力测试。
- 21：构造XOR有无交叉与加性目标对照；真实40k固定树预算只改7/15叶，保留容量增加恶化的结果。不是FM/DCNv2论文完整复现。
- 22：概率评分手算、独立calibration上重拟合预定sigmoid并核对保存audit预测；固定概率只改5/10/20分箱，显示可靠性箱计数与分段。
- 23：独立模拟二价手算、只改外部价、保守预算资格、固定潜在市场只改出价倍数及三种子。
- 24：相同潜在市场仅开关pacing、逐请求预算检查、紧预算与预测偏差压力测试，整合数据/预测/决策的不同证据边界。

20/21调用 `reclab/ads_teaching.py`，同样使用begin/finish创建ads实验，config.workshop标明教学性质；只评估select，服务latest明确跳过这些run，避免教学实验替代正式200k模型。工作台对教学结果使用独立select表，不套用正式audit schema。教学聚合在workshops.json，权重/逐条预测仍忽略。

本轮六章共30代码格、6张图全部干净内核执行与图像渲染检视。执行脚本只启动自己的临时内核，不触碰原Jupyter用户内核。后续搜索深化接手时应使用25及之后编号，保留18～24与本轮完成记录。

最终验收：80项pytest通过；25本Notebook目录可用；正式概率保存指标重算一致、500条树导出与原生离线预测匹配、30条API概率匹配；实际18580上五种原搜索前10结果仍与离线一致。原四组双塔、ItemCF、68项输入SHA与45份讲义API回归通过。工作台已实际显示19～24入口、教学select表与正式audit表，并成功调用正式预测。
