# Recommendation Lab · 搜推学习实验室

以 Notebook 为主线的本地推荐学习项目：真实数据、PyTorch 双塔、ItemCF、可追溯推荐过程，以及 WSDM Cup 2022 XMRec 冠军核心方法的教学复现。共17本 Notebook，配套45份公开讲义。

## 克隆后，一条命令配置

实测环境：Apple M1 Pro、16GB、macOS15。macOS要求Apple Silicon和14以上；其他系统尚未做完整验收。建议预留至少5GB磁盘。无需Docker、Podman或GPU。

```bash
git clone https://github.com/lynnlee1229/recommendation-lab.git
cd recommendation-lab
# macOS缺少OpenMP运行库时先执行：brew install libomp
./setup.sh
```

需要已有uv或Python3。脚本使用uv>=0.13.6；缺少合适版本时从官方PyPI将固定版uv安装到项目.setup-tools，不替换全局工具。随后：

1. 根据uv.lock安装项目Python3.12环境和依赖。
2. 从固定原始公开地址下载68项数据/讲义，逐项核对字节数和SHA-256；另校验5份冠军参考源码。
3. 准备Retailrocket，训练热门/ItemCF与四组双塔；准备XMRec并训练缺失的竞赛实验。
4. 注册项目内核、执行pytest和教学API检查，顺序执行11~15五本竞赛Notebook。

首次需要联网下载约376MiB公开输入，以及Python和依赖；耗时受网络影响。下载支持有限重试、断点续传及冻结哈希校验。完成文件会跳过，已有成功模型会保留。脚本不会自动启动服务，不要求数据账号或浏览器凭据；若原站要求认证或哈希变化，会失败并报告，不跳过校验。

可选：`./setup.sh --download-only`只安装环境并校验输入；`./setup.sh --no-notebooks`完成训练测试但不执行Notebook。发生下载中断后重新运行同一命令。失败训练目录会保留供检查；不会自动覆盖不完整实验。

## 打开学习入口

```bash
./lab.sh up
./lab.sh notebook
```

打开 http://127.0.0.1:18580 ，点击“打开 Notebook 课程”。从00课程导航开始，01/02学习真实数据与双塔，03~10为推荐系统机制实验，11~15为XMRec数据、特征、图模型、排序和评估。

工作台18580、Jupyter18581均仅监听127.0.0.1。Jupyter保留随机token认证，同源入口会完成本机登录。已有其他程序占用端口时请先核查，脚本不会代替你停止它们。

```bash
./lab.sh status
./lab.sh notebook-status
./lab.sh notebook-stop  # 先保存Notebook
./lab.sh stop
```

Notebook代码和项目源码可编辑，使用Recommendation Lab (Python 3.12)内核。02的Run All会新建训练实验；其他章节分别标注真实数据、构造张量或模拟流量。学习约定见[NOTEBOOK_FIRST.md](docs/NOTEBOOK_FIRST.md)。

## 仓库包含什么

- Python源码、前端、16本带教学输出的Notebook、锁文件、测试、公开下载清单。
- [M1 Pro实测结果](results/2026-10-30-m1-pro/README.md)：聚合指标、训练曲线、配置和校验来源。
- 数据集、讲义PDF、模型权重、逐用户预测、日志、环境、缓存和认证信息均不上传。`data/`和`artifacts/`由配置脚本在本机生成。

已有本机训练产物不会因上传被删除。克隆者会自行下载数据并重训，而不是把静态结果文件当作可用模型。

## 验证与新实验

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_restored_lab.py
.venv/bin/python scripts/execute_notebooks.py --only 11_XMRec_Start 12_XMRec_Features 13_XMRec_Graph 14_XMRec_Ranking 15_XMRec_Evaluation
.venv/bin/python -m reclab.competition run --tag my-experiment
```

图模型最多4线程，LightGBM单线程。模型按select选型，audit独立报告；不使用保留测试集调参，不声称隐藏测试成绩。当前方法是教学核心复现，非完整冠军集成、生产实时链路或线上收益证明。

## 数据与方法来源

- [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [XMRec官方数据](https://github.com/hamedrab/wsdm22_cup_xmrec)
- [OPDAI冠军方案](https://github.com/opdai/wsdm2022-xmrec-top1-solution)
- [王树森推荐系统讲义](https://github.com/wangshusen/RecommenderSystem)

固定版本、下载地址、哈希分别记录在DATA_DOWNLOADS.json和docs/xmrec_sources.json。数据与讲义仍受原发布方条款约束，仓库不再分发它们。课程目录覆盖不等于完整论文复现；更多方法边界见[XMREC_REPRODUCTION.md](docs/XMREC_REPRODUCTION.md)。

## 新增：同一系统中的深入实验

```bash
.venv/bin/python -m reclab.deep_recommendation
.venv/bin/python scripts/execute_notebooks.py --only 16_Recommendation_Experiments
```

16章对比均匀/热门/难负例、mean/GRU/带位置注意力、精确/12簇粗索引；默认12000条真实因果样本、4轮、2个种子。正式旧双塔不覆盖。结果进入同一工作台“07 实验对照”，共用实验记录/API，后续搜索和广告沿用该接口。详细协议见[推荐深化](docs/RECOMMENDATION_DEPTH.md)。
