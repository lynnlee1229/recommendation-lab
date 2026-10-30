# 搜索与广告：分session补课，串行写同一Lab

用户原话：『先对照这些课程梳理搜索和广告的完整学习路线，看看现有内容缺什么，再分别开新session补齐。还是迭代同一个Lab，不要为了迁就现有实验来组织课程。广告看看有没有github讲义』。

## 共同边界

唯一checkout `/Users/lynnlee/Documents/Codex/2026-10-30/task/soutui-lab`；`lynnlee1229/search-recommendation-ads-lab`，同一main与Git历史。项目外新聊天目录仅协调/暂存，不复制、clone或另起服务。先读本文件、PRODUCT_ARCHITECTURE.md、NOTEBOOK_FIRST.md、两个LEARNING_ROADMAP和COURSE_AUDIT_SOURCES.json。

一级保持公用/搜索/广告/推荐，教材路线决定学习顺序。现在35本Notebook，公用31–34已经完成，领域旧综合与深入课原样保留。18+18单元是验收目标，不强迫恰好新建18本，也不接受几个通用演示覆盖所有目标。新增章节逐项可运行、有解释、先修与来源；对未公开完整讲义的目录，使用已核验补充文献，不假称作者已发布。

使用既有.venv、reclab/learning、registry、下载哈希与同一FastAPI；旧正式模型/数据/评估边界不变。新增真正训练只能写新的教学run，按workshop隔离，latest不指向教学模型。构造机制、真实数据和模拟市场在Notebook和网页显著标识，不推导虚假的线上收益。旧35本不重建；新编号从35起检查分配，广告必须接搜索最终编号。

Jupyter18581/用户内核保持、随机token和同源保护保持；仅正常重启本项目18580，不停其他服务。依赖只装项目；不得改系统睡眠、安全或网络保护；下载/端口/GitHub按审批。大文件、凭据、环境、逐条日志不进Git。界面仍Notebook优先；每个领域补自己的讲义/路线入口，不复制全局侧栏和服务器。

## 搜索session（先写）

按SEARCH_LEARNING_ROADMAP.md的S01–S18实施。特别补查询处理/意图/改写、缓存召回、BERT相关性训练、排序目标与查询推荐。现有26–30作为课程中的合适章节，而非路线边界。逐项建立交付映射、运行证据与局限。新数据先查来源许可；中文人工词例只称机制，不能把SciFact说成中文搜索数据。

完成顺序：代码与Notebook → 清洁内核执行/渲染 → pytest与课程导航、原三领域API/模型回归 → 浏览器桌面和窄屏 → 本地commit。最后一步在`docs/SEARCH_CURRICULUM_COMPLETION.json`写`{"status":"complete","writer_released":true,...}`并与最终成果一同提交；附每单元文件、覆盖状态、验收摘要、最后占用编号与未完成项。若外部阻塞导致无法全部达标，必须用`status:"blocked"`并列出真实未完成项，但仍在停止写入时将writer_released置true。标记提交后不再写checkout，结束本session。

## 广告session（后写）

可以先读公开课程与现有实现，在本聊天自己的work目录做准备。开始写checkout必须同时满足：

1. 搜索session的compact wait_threads快照表明该轮已结束，没有运行中的写入；threadId由创建时prompt提供。
2. SEARCH_CURRICULUM_COMPLETION.json已提交、writer_released为true，status为complete或blocked。
3. `git status --short`为空；重新读最新main、课程与编号，避免覆盖搜索新增共享文件。

不要创建分身Lab或另一个checkout来绕过等待。等待是已授权串行流程，可用有界wait_threads等待；不因等前序就问用户是否继续。搜索若列出未解决项，广告应保留，不掩盖其状态。

按ADS_LEARNING_ROADMAP.md的A01–A18实施：先产品/计费/市场，再资格和候选、预测、拍卖、投放、反馈；不要沿用只有Criteo字段的知识地图。已验证GitHub中文第8章PPT、南京课程表及作者RTB英文教程，不要把社区合集当权威课件或未经许可再分发原件。每个单元有独立证据与边界；缺转化日志可用明确机制例，不能造真实CVR/归因。完成后同样验收本地commit，写ADS_CURRICULUM_COMPLETION.json逐项交付状态并释放checkout。

## 当前推送审批阻塞

公用基础的9bfd436已本地完成，但自动审批拒绝其向默认main的push，理由是需要用户直接确认更新默认分支。远端尚为42625aa。不能重试或换工具/分支来绕过这次拒绝；本轮及两个session继续完成本地实现、验收与commit，在收到针对推送的明确新确认前不要push。不要反复请求确认，更不能把未推送称已上传。父对话将向用户说明并请求具体审批；后续是否放行以可信用户新消息为准。
