# 同一搜推广告系统：阶段交接

## 用户目标与授权

用户明确要求当前对话完成推荐深化并推送，再开全新对话完成搜索，搜索完成推送后再开全新对话完成广告；不在乎耗时。唯一核心要求：所有阶段迭代同一系统，复用可复用部分，绝不能最终形成三个独立Lab。用户已明确公开GitHub仓库并授权后续迭代推送。

## 仓库与运行环境

同一checkout：/Users/lynnlee/Documents/Codex/2026-10-30/task/soutui-lab；公开远端https://github.com/lynnlee1229/recommendation-lab，main。目标机LYNNZLLI-MB2，M1Pro16GB，macOS15。使用已有.venv/Python3.12，依赖按uv.lock，已有Homebrew libomp。旧uv0.9有问题，项目.bootstrap-uv/bin/uv为0.13.6。公开数据在data，模型/运行结果在artifacts，均Git忽略；任何凭据、环境和数据不上传。

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

## 搜索阶段期望

在同一系统中建立实际可复现的小型搜索任务：BM25、语义/向量检索、混合融合及可行的小型重排对照；查询—文档—qrels要有真实公开来源、下载固定版本和SHA，不从Retailrocket凭空造查询标签。资源适配M1，允许冻结小模型embedding或小样本明确标注，不承诺大模型训练。指标至少MRR/NDCG、召回覆盖、实测延迟、失败案例；子集限制和未判断文档处理说清。先调研数据/模型许可和大小，再实施。复用已有registry、指标、索引、工作台、依赖、下载和教学形式；数据领域adapter可新增但不可另建服务器或另repo。

用户授权实现、训练、验收、推送，不必再问是否继续。新增依赖只在项目，网络/本机端口按执行审批流程，不绕拒绝。所有代码、真实运行、tests、Notebook渲染/接口/页面证据齐全再commit push。保持数据/模型/凭据忽略。更新本文件与README后，用正常create_thread工具在新对话交接广告阶段。新对话只读本交接与仓库，无需其他聊天。

## 广告阶段期望（由搜索阶段完成后交接）

仍同一系统：真实公开点击预测数据上的可解释基线/特征模型与校准实验；拍卖、出价、预算pacing使用明确模拟市场，收入只称模拟结果。复用registry、数据下载/哈希、评估、Notebook和统一工作台，预测与决策层分清。不得把推荐未交互当真实负曝光。公开数据许可/规模须先核实。完成后同样实际训练验收、push并给用户三领域整合结果与局限。

## 交接机制

当前Codex list_projects未列出这个checkout；不要误用其他llm项目。若新对话为独立协调任务目录，明确cd到上述唯一checkout工作，不创建或clone另一个Lab。若原目录无法访问，报告精确阻塞而非复制项目绕过。用户明确要求逐阶段新对话，阶段不可并发改同一checkout。使用默认模型，不擅自切换。
