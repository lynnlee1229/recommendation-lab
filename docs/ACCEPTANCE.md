# 本地验收记录

日期：2026-10-27（北京时间）。范围：独立离线推荐学习工作台；未连接生产系统。

## PASS：数据与训练

- 原发布入口的 Retailrocket ZIP 已实际下载、解析；SHA-256：`5dc06173eaa4a4d3b0a5f6afc4daffb0218d95e409aaacba5fdaa0214eacf2a2`。
- 原始 2,756,101 条事件；删除 460 条完全重复记录。训练期选定 6,734 用户 / 5,000 商品；每个双塔 run 使用 54,392 个因果新物品目标样本。
- 所有模型使用同一冻结数据版本 `2772cab0aca8cb81`。训练用户/物品选择不使用未来窗口；历史输入严格早于目标。
- 已完成热门、ItemCF 以及四组 8-epoch 双塔训练，均保存真实权重、训练曲线和验证结果。
- 另从页面提交并完成一次 2-epoch 验收训练：`20261027T114624347212`；它是功能验收，不作为首轮模型比较结论。

## PASS：验证集（339 个可评估用户）

| 模型 | Recall@20 | NDCG@20 |
|---|---:|---:|
| popular | 0.020754 | 0.017910 |
| itemcf | 0.145167 | 0.079874 |
| baseline-history32 | 0.060756 | 0.028300 |
| ablation-no-history | 0.031739 | 0.020035 |
| ablation-logq | 0.066995 | 0.040203 |
| ablation-history64 | 0.069979 | 0.039398 |

ItemCF 当前强于所有双塔对照。双塔没有被预设为必须胜出的模型。模型选择、论文收益和业务收益是不同问题。

## PASS：一次保留测试集验收

四组双塔中依据验证 NDCG@20 选择 `ablation-logq`，然后执行一次：

```bash
.venv/bin/python -m reclab.cli final-test ablation-logq
```

236 位测试用户：热门 Recall@20=0.030241；ItemCF=0.125913；所选双塔=0.056869（精确值见 `artifacts/final-test.json`）。保留测试窗口没有用于训练或该次选择，后续学习调参仍应使用验证集。

## PASS：编译、检查与测试

在项目目录执行：

```bash
.venv/bin/python -m compileall -q reclab scripts tests
node --check web/app.js
bash -n lab.sh
uvx ruff check --select F reclab scripts tests
.venv/bin/python -m pytest -q
```

全部 exit 0。pytest：15 passed，0 skipped，11.22s；另有依赖库 Starlette/httpx 的一条弃用提示，不影响本次验收。测试包含手算排名指标、重复目标屏蔽、有限梯度、实际小批次优化、源数据与样本时间边界、在线/离线打分一致、API 参数和跨来源写保护。

## PASS：真实页面与生命周期

- 用隐藏内置浏览器操作：推荐、逐商品解释、特征快照、真实样本、softmax 与 dL/dlogit 诊断、页面启动训练、完成后结果回显。
- 390 / 768 / 1440 像素宽度检查；表格在容器内滚动，页面无水平溢出。平板发现并修复指标表挤压；最终复查有数据的实际页面。
- 最终页面 console error/warn 为空。
- `./lab.sh setup` 在已有快照上重入成功，没有重建或覆盖已有实验；ZIP 与 manifest 哈希一致。
- `./lab.sh stop` 后 `./lab.sh up` 成功，模型与数据保留；`/api/health` 返回当前项目绝对路径。
- 服务监听 `127.0.0.1:18580`，留给用户继续使用；没有注册开机启动或远程服务。
- 资源抽样：data 310 MiB、artifacts 17 MiB、独立环境约 982 MiB；一次服务 RSS 抽样约 132 MiB，非峰值承诺。训练 CPU 4 线程，未使用 GPU。
- 原 lakehouse-lab 保持 `feature/learning-studio` 干净工作树；`podman ps` 仍无运行容器，未启动或修改它。

## UNKNOWN / 尚未实现

学习型粗排/精排、DIN/SIM/MMoE/DCN 等后续模型、真实曝光 CTR、在线 A/B 收益、冷启动端到端效果与实时链路均未验收。课程地图完整映射 45 份 / 902 页讲义，覆盖目录不等于全部论文复现完成。LogQ 只复现校正机制，Q 为固定子集的经验频率，未复现原论文流式估计器与工业规模。

## 2026-10-28：Notebook 与源码入口增量验收

PASS：网页源码白名单读取实际 Python 文件；可在“代码与 Notebook”切换双塔、数据、召回、训练与服务模块。新增 HTML cache revalidation 与资源版本参数，修复浏览器继续显示旧页面的问题。

PASS：两本 Notebook 用项目 Python 3.12 内核从头执行，保存输出并转换 HTML 检查预览。01 包含 8 个代码单元格，02 包含 10 个；无错误。已检查 4 张实际输出图表和 JupyterLab 中的保存内容。02 使用共享训练器新建 run `20261028T023407346105`，未使用保留测试集，也未覆盖原模型。

```bash
.venv/bin/python scripts/execute_notebooks.py
.venv/bin/python -m compileall -q reclab scripts tests
node --check web/app.js
bash -n lab.sh
.venv/bin/python -m pytest -q
```

全部 exit 0；pytest 为 21 passed、0 skipped（4.91s），保留原依赖弃用提示。新增检查覆盖源码白名单、固定登录目标、无效目标不启动进程、跨站启动拒绝、Notebook 格式/语法/内核，以及状态接口不泄漏 token。

PASS：真实 JupyterLab UI 编辑并执行 Python 得到 `UI_KERNEL_OK 42`，随后恢复教程原文、重执行并保存。最终文件无此临时标记；模型源文件未被 UI 验收改动。JupyterLab 的 Python 编辑器可直接打开 `reclab/models.py`。

PASS：JupyterLab 仅监听 127.0.0.1:18581，随机 token 与签名 cookie 认证保留；未认证 API 被拒绝，认证 API 正常，kernel argv 为本项目 `.venv/bin/python`。runtime 目录权限 0700、server-info 权限 0600。认证信息只留在忽略的本地私有运行文件中。

PASS：`./lab.sh notebook-stop` → `notebook-status` 确认停止 → `./lab.sh notebook` 确认重启。网页服务也已重启并读回新 HTML/cache header。Jupyter 与网页均保持运行供用户使用。新增源码页在窄屏/平板没有页面级横向溢出；源码本身在容器内滚动。

## 2026-10-28：Notebook 优先课程配套

PASS：保留原有01/02 Notebook，新建00课程导航与03~10八本专题，共11本Notebook、58个保存执行结果的代码格。新增9本的40个代码格均用干净内核顺序执行成功，保存输出和HTML预览；未重新训练原先两本或修改它们。45个讲义主题逐条关联已执行的配套Notebook，目录覆盖不表示所有论文或业务系统已完整复现。

专题：基础/分桶、召回与采样、排序/MMoE/粗排、FM/DCN/LHUC/Bilinear、DIN/SIM风格机制、多样性/DPP、冷启动/保量、工程/漂移/蒸馏。真实快照、受控张量、模拟流量分别标注。Deep Retrieval仅路径搜索；DIN是归一化注意力教学变体；时长、作者、社交示例为受控机制。均没有声称线上收益。

PASS：FM显式/向量化数值和梯度一致，DPP增量贪心与慢速贪心一致且不超过穷举全局最优；负采样校准逆公式、Gate归一化、padding全空、缓存/检索近似、Bloom不漏已加入元素、配额守恒等已核验。完整pytest为33 passed、0 skipped（4.49s），保留依赖弃用提示。

```bash
.venv/bin/python scripts/execute_notebooks.py --only 00_Course_Map 03_Foundations_AB 04_Retrieval_Methods 05_Ranking_Multitask 06_Feature_Crossing 07_Sequence_DIN_SIM 08_Diversity_DPP 09_ColdStart_Traffic 10_Engineering_Improvement
.venv/bin/python -m compileall -q reclab scripts tests
node --check web/app.js
uvx ruff check --select F reclab scripts tests
.venv/bin/python -m pytest -q
```

均exit 0。已查看新增4张实际输出图。原生JupyterLab展示课程导航，并实际点击导航打开特征交叉Notebook，确认非下载链接或平台示意页；最后返回课程导航供用户学习。左侧文件栏收起，遮挡提示已隐藏。真实交付页保留，用户原有页面未关闭。

PASS：实时课程API读回45个已执行配套；工作台新增醒目的Notebook课程入口，默认/notebooks进入00导航。只重启网页服务以加载入口，未停止用户Jupyter内核、未启动或修改原流lab、未覆盖正式模型。学习约定保存于docs/NOTEBOOK_FIRST.md。

## 2026-10-28：06 页完整 Notebook 目录修复

原页面写死01/02入口，遗漏后续专题。已改为读取 /api/notebooks，其数据复用本地登录目标白名单与课程元数据；页面按文件排序显示00至10共11个Notebook。新增课程只需登记统一目标目录，页面无需另写入口。未改Notebook文件、正式模型或Jupyter内核。

验证命令：.venv/bin/python -m compileall -q reclab tests；node --check web/app.js；uvx ruff check --select F reclab/service.py tests/test_notebook.py；.venv/bin/python -m pytest -q。均exit 0，pytest34 passed、0 skipped（10.99s），保留依赖弃用提示。新增回归核对目录与全部已登记Notebook一致、文件存在、顺序和专题标题正确。

已重启网页服务并在实际浏览器读取页面，确认11个可点击入口及对应目标全部出现；测试页已关闭，用户工作台与Notebook页保留。页面资源版本更新为catalog1以避开旧缓存。

## 2026-10-29 · WSDM Cup 2022 XMRec 冠军核心复现

- PASS：官方 starter 固定版本的22个原始文件逐项 SHA-256 校验，与冠军仓库数据一致。完整五市场训练语料1,176,863条，保留官方8,179用户×100候选；不采样评估列表。
- PASS：原版 icf_v2/icf_v3 在真实8用户历史上的384对分数对照；最大误差IUF1.64e-7、rating8.10e-7。原版 LightGCN 的BPR误差0、梯度最大差9.31e-10；nDCG对照测试包含第1/2/5/10/11名边界。
- PASS：实际完整主流程 `.venv/bin/python -m reclab.competition run`，run `20261029T091104737318-champion-core`，77.38s、主进程峰值RSS1422.92MiB。结果、来源和差异记录在XMREC_REPRODUCTION.md与xmrec_result.json。
- PASS：本地单目标ItemCF完整验证合并nDCG@10=.598037，对齐论文Table4报告的.5980精度；t1存在约.000106残差，未宣称逐位一致。冻结select选型为lgb_memory；audit1605用户nDCG=.647982。没有隐藏测试分数。
- PASS：目标qrel不进入特征训练语料；排序器fit/select/audit用户不交叉；负采样仅用于训练，候选评估保持100项。LightGBM全验证集的训练内混合分数不作为泛化结果。
- PASS：macOS多libomp混用导致LightGBM多线程阻塞，改为n_jobs=1；原生Booster解释/预测也显式num_threads=1。回归子进程验证先训练Torch、再连续训练LightGBM/TreeSHAP，超时45s约束。未设置KMP_DUPLICATE_LIB_OK或改全局运行库。
- PASS：`.venv/bin/python -m compileall -q reclab tests scripts`，exit0。
- PASS：`.venv/bin/python -m pytest -q`，45 passed，最终8.05s；仅既有Starlette/httpx弃用提示。
- PASS：`uvx ruff check --select F reclab/competition tests/test_competition.py scripts/create_competition_notebooks.py`、`node --check web/app.js`、`git diff --check`。
- PASS：`.venv/bin/python scripts/execute_notebooks.py --only 11_XMRec_Start 12_XMRec_Features 13_XMRec_Graph 14_XMRec_Ranking 15_XMRec_Evaluation`，5本30格从清洁项目内核执行，4.4/3.0/4.1/7.8/3.4秒；保存输出及HTML预览。6张实际图表已逐图检查，原生Jupyter14学习文本/代码/项目内核已显示。
- PASS：工作台实时目录16本，11～15齐全；竞赛graph.py源码和xmrec_code_graph编辑链接正确。390/768/1440尺寸检查新增入口和目录，无新增布局溢出；窄屏原有导航为水平滚动。临时viewport已恢复，关闭本轮2个QA页，保留用户原工作台。
- 范围：数据/模型在data/xmrec和artifacts/xmrec；未修改旧Retailrocket数据或模型、未重写01～10 Notebook，00只追加入口。只重启推荐工作台以加载白名单；保留Jupyter进程与用户已有内核，未操作lakehouse-lab。
- 未完成范围明确：完整冠军的全部特征/模型组合、2048维图训练配方、10折bagging和隐藏测试成绩均未宣称复现。当前交付为可运行、有源码数值对照和论文基线分数对照的本机核心复现。

## 2026-10-29 · M1 16GiB 恢复包

- 用户目标：另一台M1 16GiB Mac，不向外部Git写入。目标要求macOS14+，由锁文件中torch2.14.1的macosx_14_0_arm64 wheel确定。未安装/升级任何系统软件或修改原流lab。
- 本地源码提交abd4066冻结恢复包，路径artifacts/portable/study-1029-A1。276项有效载荷，16本Notebook、45份讲义、两套数据与成功模型；8个包总785.27MiB，7个独立ZIP分别<100MiB，原始Retailrocket另放约291MiB。排除.venv、运行日志、Jupyter凭据/会话、失败XMRec尝试、Git配置与无关lab。包内讲义路径改为resources/slides。
- PASS：`.venv/bin/python -m compileall -q scripts tests`；`.venv/bin/python -m pytest -q` 56passed（新增11项覆盖损坏、路径越界、符号链接、重复成员、拒绝覆盖和拒绝重定向环境）。ruff F与git diff --check通过。
- PASS：`.venv/bin/python scripts/export_portable.py --output artifacts/portable/study-1029-A1`，全部成员和包SHA-256通过。
- PASS：`python3 artifacts/portable/study-1029-A1/restore.py verify`，8包276文件。
- PASS：`python3 artifacts/portable/study-1029-A1/restore.py restore --destination /private/tmp/phase-restore-EBWXvG/rehearsal`，276文件逐项一致。
- PASS：`env PATH=/usr/bin:/bin:/usr/sbin:/sbin python3 artifacts/portable/study-1029-A1/restore.py setup --destination /private/tmp/phase-restore-EBWXvG/rehearsal`，刻意不提供现有uv；项目内bootstrap实际从PyPI安装uv0.12.5，再按uv.lock安装Python3.12.14与全新依赖环境；reclab内核安装到恢复项目自身prefix，未复用原.venv或全局内核。
- PASS：`python3 artifacts/portable/study-1029-A1/restore.py validate --destination /private/tmp/phase-restore-EBWXvG/rehearsal --notebooks`，编译通过，56passed/64.11秒，11～15共30代码格顺序执行通过；真实沙盒模型写入恢复目录，主模型和原Notebook未覆盖。
- PASS：恢复目录的临时HTTP服务使用空闲端口启动；health确认项目根为新目录；16Notebook入口可用，45PDF逐个HTTP200且PDF文件头正确；32维特征及10条实际推荐通过；竞赛模块/数据由完整测试覆盖。验收后临时HTTP进程已退出。
- PASS：原18580工作台PID20954与18581 Jupyter PID28815仍ready，未重启；原01~15Notebook与业务算法未改。本轮代码仅新增恢复/打包能力和忽略规则，恢复源版本与后续验收文档分开记录。
- UNKNOWN/未实施：目标M1机器尚未连接、未做实机验收；OpenAI项目文件/dots上传、下载入口与大小/类型限制尚未实测，本包未上传；不能把本机新环境验证当作目标机器或云端传输成功。包内VERIFIED_ON_THIS_MAC.json和docs/portable_result.json保留此边界。
