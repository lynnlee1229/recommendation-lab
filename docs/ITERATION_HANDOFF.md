## 广告核心课程已补齐（2026-10-30）

在搜索31c2a8c结束、提交writer_released标记且工作区干净后接手同一checkout。新增47–62十六本，A01–A18按产品/计费/市场→资格候选→响应预测→投放反馈组织；原19–20支持A09、22支持A12，旧18/23/24后置参考。全Lab63本，广告23本；旧47本逐字节保留，正式模型不重训不替换。

48格清洁执行并保存输出，逐本原Jupyter渲染检查，93 pytest、4导航测试、桌面/390px与原三领域API回归通过。真实10k Criteo类别LR/FM workshop `20261030T143314-b0538124`，select logloss .1323/.1783，FM过拟合负结果保留；两模型权重重载预测和重算指标一致。其余新增转化、画像、归因、拍卖与预算均显式构造机制，不称真实CVR、IPS/OPE或线上增收。中文GitHub第8章96页已核验固定blob并读取文字，原件仅链接；不是现代工程全套课。

证据：ADS_CURRICULUM_GUIDE.md、ads_course.json、ADS_CURRICULUM_COMPLETION.json与results/2026-10-30-ads-curriculum。修复共享台账对已有BERT workshop缺少逐方法耗时的兼容，不改原实验记录；未知耗时明确显示未记录。同一18580正常重启，18581与用户内核未停止。用户已授权验收后推送原origin/main；最终网络结果以本聊天最终SHA为准。

## 搜索补课已验收，串行交接（2026-10-30）

同一checkout新增35–46十二本，覆盖S01–S18核心教学目标；全Lab47本，搜索19本。旧35本逐字节核对5943d1e保持，原正式模型latest不变。42章TinyBERT真实全参数微调run `20261030T142455-38553a49`，32/8原train查询、128弱监督pair、两epoch；选择NDCG约0.806与冻结相同，权重更新已验且checkpoint重载预测/指标完全一致。构造中文词例、规则意图/改写、缓存、质量多样性与会话建议均显式标机制。

12本61格清洁内核执行并保存输出/HTML；真实Chrome逐本渲染、桌面与390px路由检查；87 pytest、导航与JS语法、68公开资产/45讲义/四组原推荐+ItemCF、广告30概率/500树预测/预算与五搜索方法回归通过。证据results/2026-10-30-search-completion；课程说明SEARCH_COURSE.md。没有新Lab/服务/数据下载，没有停止Jupyter或用户内核。

最后占用46，广告从47开始检查。SEARCH_CURRICULUM_COMPLETION.json作为最终提交的释放标记；广告仍须确认本聊天已结束、标记已提交且git status干净后才写。用户已明确允许本轮与后续验收后直接推送原main；保留下文旧拒绝作为历史，不再要求重复确认。最终推送SHA见搜索聊天最终答复；本标记不假设尚未执行的网络结果。

## 当前推送授权（2026-10-30 更新）

用户已明确确认“允许推送到原仓库的main，后续搜索和广告补齐并验收通过后也直接推送，还是同一个Lab，别再反复问了。”原自动审批拒绝为历史，现获针对main的新确认。搜索当前唯一writer完成验收后直接推送原origin/main，广告仍须等释放标记与聊天结束。下文未更新的等待审批表述仅为历史。

# 课程主线审计（最新，2026-10-30）

用户要求先对照课程建立搜索和广告完整学习路线，再分别新session补，不迁就现有实验。已完成18+18语义单元审计与源快照：SEARCH_LEARNING_ROADMAP.md、ADS_LEARNING_ROADMAP.md、COURSE_AUDIT_SOURCES.json。本轮只新增路线文档，不声称已实现缺口。GitHub中文广告PPT已由API核实，确为ccrbook/slides/Chapter 8 计算广告.pptx；它是单章，不是全套工程广告课。

公用session已完成35本课程并在本地提交9bfd436、结束写入；其推送被自动审批拒绝，要求用户直接确认更新main。远端仍为42625aa。不要借旧授权绕过拒绝，当前审计与后续补课可本地继续实现/验收/提交；在收到明确的新确认前不推送main。本轮最终答复向用户说明这一具体审批阻塞。

串行交接规则见COURSE_COMPLETION_HANDOFF.md：当前先完成审计本地提交并停止写checkout，再启动搜索和广告新session；广告先只读准备，确认搜索session结束且释放标记已提交、工作区干净后才写同一checkout。保持同一Lab，旧35本和固定正式模型不覆写。

# 公用基础课程已完成（最新）

同一checkout新增31–34四本独立公用课程，总35本：公用4、推荐17、搜索7、广告7。旧31本未改动。公用学习路径、TARGETS、curriculum、API与源码白名单已接通，首页缺口提示替换为课程入口。详见COMMON_FOUNDATIONS.md。下文为历史。

# 模块层级修正（最新，取代下面的第一版产品布局）

用户指出领域、算法、工具混排，明确要求按“公用、搜、广、推”组织，发现缺口后另起session补。现一级仅公用/搜索/广告/推荐。二级为模块自身的课程、交互、记录和源码；双塔仅在推荐内，广告预测与模拟拍卖分开。全局台账归公用；领域内台账和源码自动限制当前领域。旧#experiments转#recommendation/two-tower，旧链接兼容。结构配置web/navigation.js，说明docs/PRODUCT_ARCHITECTURE.md。

课程归属纠正：原00是推荐导览，01–03仍为推荐入门。现31本全部原样保留，推荐17/搜索7/广告7；不再冒充公用基础。确有公用独立基础课程缺口，将按用户要求单独新建session补，参见docs/COMMON_FOUNDATIONS_HANDOFF.md。不要再把工具放回一级，也不要把推荐课直接挪到公用。下文第一版“基础4/推荐13”和旧首页说明仅为历史。

# 产品化迭代（2026-10-30 最新）

统一品牌为“搜推广告学习 Lab / Search Recommendation Ads Lab”；原GitHub仓库已改名search-recommendation-ads-lab，ID仍为R_kgDOVc_wBw，沿用同一checkout和历史。origin已更新。首页并列三领域，推荐/搜索/广告各自工作区，共享实验记录按领域筛选并折叠详情。31本Notebook按基础4/推荐13/搜索7/广告7组织，可筛选与检索；旧Notebook未重建或修改。源码与讲义入口保留，原工作台07现为“共享实验记录”。没有新服务器、数据副本或重训。

本轮83项pytest通过，verify_restored_lab和verify_ads通过；浏览器验证真实推荐、RRF查询、点击概率、模拟拍卖、课程筛选/无匹配与无效JSON恢复。390px首页无横向溢出，截图见docs/screenshots。Jupyter与用户内核未停止，仅正常重启本项目18580。下文为各阶段历史交接，旧页面编号和内核显示名描述以本节为准。

# 同一搜推广告系统：阶段交接

## 当前最终状态：三阶段均已实现

广告阶段已完成实现、真实训练与原功能回归，沿用下述唯一checkout、同一main、环境与服务。以下较早阶段的“下一对话/后续广告”是历史交接，不再是待办。最终广告run `20261030T125738-f82b6582`，精简结果 `results/2026-10-30-ads`；完整来源、分区和局限见 `docs/ADS_EXPERIMENTS.md`。

官方Criteo固定提交与LFS SHA，下载47,872,279字节，取单分片前200,000真实曝光。train/select/calibration/audit为109935/29984/30153/29928，特征哈希组无跨区；线性AUC .721468优于本轮GBDT；sigmoid改善区间跨零，isotonic负结果保留。原始抽样率未知，概率不代表线上CTR。独立二价按曝光模拟市场，3个广告主、6000请求、3种子、4策略；全部预算检查通过，收益仅称模拟。

新增共享probability.py评估、ads adapter、独立可复用auction.py；仍同一registry/FastAPI/工作台07/源码白名单/课程目录，无新Lab。新增第18章综合章及19～24六本深入章（共25本）；综合章四格两图，深入章30格六图，干净内核执行与渲染检视通过。80项pytest通过；68项原始下载哈希、45份讲义API、原四组双塔和ItemCF验收通过；500条树推理与原生离线预测复核、30条API raw/sigmoid/isotonic误差<1e-10，所有保存指标重算一致。

此Mac的PyTorch服务进程原生加载LightGBM时发生退出139；已改为读取原模型JSON树结构推理，含类别与缺失路由，不改系统安全或线程设置。训练仍在原CLI用原生LightGBM完成。原失败run/诊断记录保留。数据、权重、校准映射、词表、明细、日志和环境全留本机，不进Git。Jupyter18581及用户内核保持，仅正常恢复本项目18580工作台。


## 用户目标与授权

用户明确要求当前对话完成推荐深化并推送，再开全新对话完成搜索，搜索完成推送后再开全新对话完成广告；不在乎耗时。唯一核心要求：所有阶段迭代同一系统，复用可复用部分，绝不能最终形成三个独立Lab。用户已明确公开GitHub仓库并授权后续迭代推送。

## 仓库与运行环境

同一checkout：/Users/lynnlee/Documents/Codex/2026-10-30/task/soutui-lab；公开远端https://github.com/lynnlee1229/search-recommendation-ads-lab，main。目标机LYNNZLLI-MB2，M1Pro16GB，macOS15。使用已有.venv/Python3.12，依赖按uv.lock，已有Homebrew libomp。旧uv0.9有问题，项目.bootstrap-uv/bin/uv为0.13.6。公开数据在data，模型/运行结果在artifacts，均Git忽略；任何凭据、环境和数据不上传。

工作台127.0.0.1:18580，Jupyter127.0.0.1:18581，保留token与同源保护。只可重启本项目拥有的工作台，不停止用户Jupyter或其他服务。不要改容器、睡眠或系统安全设置。

## 推荐阶段已完成

旧Retailrocket四组双塔、ItemCF和XMRec流程保持可用。新增reclab/deep_recommendation.py：均匀/热门/16候选hard负采样、mean/GRU/带位置attention、同模型历史打乱、12簇粗索引probes1/4/12。实际12000样本×4epoch×2seed，共10个模型。统一样本、训练批次顺序和优化预算，维度相同但参数量不同。低分/负结果已完整记录，非完整DIN/SASRec复现，也未证实优于旧模型。

已完成run 20261030T123443-578def14，结果results/2026-10-30-recommendation-depth；权重及逐用户结果在artifacts/experiments/recommendation。notebooks/16_Recommendation_Experiments.ipynb干净内核5格成功、3图已检视。共17个Notebook；pytest64通过。API、UI、源码白名单、Notebook目录已接入。

## 必须复用的接口

- reclab/experiments.py：begin(domain, config, provenance) -> 新目录；finish(folder,result)完成后不可覆写。catalog()记录目录；result(domain,run_id)读取结果。domain= recommendation/search/ads。失败记录应保留、不得伪装成功。
- /api/experiments、/api/experiments/{domain}/{run_id}；同一web工作台07实验对照，任务结果不同schema按domain渲染，不能混比任务指标。
- reclab/retrieval.py已有稳定tie的topk、metrics；reclab/learning/retrieval.py已有教学粗索引；可抽象复用，不能复制一套。
- 现有reclab/service.py FastAPI、web/app.js、web/index.html、Notebook TARGETS和docs/curriculum.json统一目录；新增课程后更新目录及验证脚本，不写死17。
- docs/NOTEBOOK_FIRST.md：先预测→改一个变量→执行→解释；真实数据/构造机制/模拟市场分清；不覆盖用户编辑Notebook、不重建旧章节。

## 搜索阶段已完成（本轮）

同一checkout、同一main已实现reclab/search.py；真实BEIR SciFact完整5183文档、300测试查询，200/809训练查询，4000对真实训练LogisticRegression。冻结MiniLM和TinyBERT，BM25/向量/RRF/线性/交叉编码器及共享12簇索引探测2/12，共7组。完成run `20261030T124927-d1467622`；精简结果results/2026-10-30-search，完整本机artifacts/experiments/search。首轮旧命名run有superseded注释，保留原记录。

NDCG@10：BM25 .6617、dense .6484、RRF .6855、linear .7034、TinyBERT .6670；负结果不隐藏。完整来源许可、四个提交版本及20项SHA见docs/search_sources.json（总114658540字节，约109.35MiB）。只用本地safetensors、不下载远程代码。新增transformers4.57.6/safetensors等已锁uv.lock，已有.venv保留。

POST /api/search五种方式已实际比对离线前10完全一致，统一07页面可输入英文声明；来源白名单和notebook TARGETS/curriculum已接入。18本Notebook，第17章4格、1张双面板图干净执行与渲染检查。pytest71通过，原推荐验收通过。原Jupyter18581/用户内核不动，仅正常重启本项目18580。重跑命令 `.venv/bin/python -m reclab.search --download`（新建运行，不覆盖）。具体边界见docs/SEARCH_EXPERIMENTS.md。

后续广告必须接着迭代这里，不重新建Lab。复用刚新增的SHA来源清单+download_public_assets.fetch（支持固定HF来源与curl下载）及原实验/API/Notebook。数据下载是显式搜索命令，原setup默认不额外跑搜索，以控制安装预算。下一对话仅广告阶段，不必重跑搜索或改其固定测试协议。

## 搜索阶段原验收范围（已达成）

在同一系统中建立实际可复现的小型搜索任务：BM25、语义/向量检索、混合融合及可行的小型重排对照；查询—文档—qrels要有真实公开来源、下载固定版本和SHA，不从Retailrocket凭空造查询标签。资源适配M1，允许冻结小模型embedding或小样本明确标注，不承诺大模型训练。指标至少MRR/NDCG、召回覆盖、实测延迟、失败案例；子集限制和未判断文档处理说清。先调研数据/模型许可和大小，再实施。复用已有registry、指标、索引、工作台、依赖、下载和教学形式；数据领域adapter可新增但不可另建服务器或另repo。

用户授权实现、训练、验收、推送，不必再问是否继续。新增依赖只在项目，网络/本机端口按执行审批流程，不绕拒绝。所有代码、真实运行、tests、Notebook渲染/接口/页面证据齐全再commit push。保持数据/模型/凭据忽略。更新本文件与README后，用正常create_thread工具在新对话交接广告阶段。新对话只读本交接与仓库，无需其他聊天。

## 广告阶段期望（由搜索阶段完成后交接）

仍同一系统：真实公开点击预测数据上的可解释基线/特征模型与校准实验；拍卖、出价、预算pacing使用明确模拟市场，收入只称模拟结果。复用registry、数据下载/哈希、评估、Notebook和统一工作台，预测与决策层分清。不得把推荐未交互当真实负曝光。公开数据许可/规模须先核实。完成后同样实际训练验收、push并给用户三领域整合结果与局限。

## 交接机制

当前Codex list_projects未列出这个checkout；不要误用其他llm项目。若新对话为独立协调任务目录，明确cd到上述唯一checkout工作，不创建或clone另一个Lab。若原目录无法访问，报告精确阻塞而非复制项目绕过。用户明确要求逐阶段新对话，阶段不可并发改同一checkout。使用默认模型，不擅自切换。


## 广告深入追加交接（最后更新）

用户追加“拆一下讲细点，广告也一样”，已保留18并完成19～24，编号全部占用；下一阶段从25起避免冲突。六章详见ADS_EXPERIMENTS和README。20/21实际四个模型的教学run使用同一ads registry，config.workshop隔离，latest只取正式run，UI明确select成绩；不覆盖正式audit。所有新增Notebook干净执行、渲染与目录检查通过。搜索深化由原搜索对话在本阶段push并停止写入后单独实施，本对话不接手搜索深化。

本轮最终HTTP复核已通过，教学run分别为 `20261030T130907-22ac6037`（线性）与 `20261030T130911-14b5a3de`（树容量）。所有80项测试通过。六本新增深入章独立执行，无并行写checkout。下一阶段须在收到本对话最终SHA和“已停止写入”后开始。


## 搜索六章深化（最新；串行完成）

广告234bb4a停止写入后才接手，未并发改同一checkout。新增25～30六章，章节/样本边界见docs/SEARCH_COURSE.md；保留17综合章、18广告综合与19～24深入章，当前共31本。新增32代码格和5图干净执行/视觉检查，83测试通过，广告概率/树与推荐、搜索五方法服务回归通过。

重排教学run `20261030T131425-ebe3235e` 在统一search registry记录两次实际拟合，160训练/40选择，select五特征未优于单BM25，负结果保留。config.workshop过滤保证latest只选正式搜索run；工作台分开渲染select。源码选择器改为自动读取白名单，可读新增模块；无新服务器/仓库。正式搜索算法与数据、原权重不改，只增加教学记录隔离和可复用辅助。精简验收results/2026-10-30-search-course。下一新增Notebook编号从31起；原章不重建。
