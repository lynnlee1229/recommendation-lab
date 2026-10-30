> 已完成：新增31–34四本公用基础课，详细内容与验收见[COMMON_FOUNDATIONS.md](COMMON_FOUNDATIONS.md)。下文保留原交接要求。

# 下一session：独立补齐公用基础课程

用户要求按“公用、搜索、广告、推荐”组织，组织后发现实质缺口再另起session补。此次确认：现有00只覆盖推荐导览，01–03以推荐为例，并无独立公用基础课程。当前一级模块与作用域已修正，下一session只补这一缺口，不再改成平铺工具栏。

唯一checkout：`/Users/lynnlee/Documents/Codex/2026-10-30/task/soutui-lab`；同一GitHub仓库`lynnlee1229/search-recommendation-ads-lab`，沿用main、origin、Git历史与.venv。不要clone、新建Lab、服务器或worktree。先读本文件、PRODUCT_ARCHITECTURE.md、NOTEBOOK_FIRST.md、ITERATION_HANDOFF.md及git status。必须等父session完成commit/push并停止写此checkout，串行修改。

补充一个小而连贯的公用基础学习路径：线性代数与张量形状、计算图与梯度、优化与泛化、评分/概率/排序及评估边界。先审查现有reclab/learning可复用部分，再确定合理章数；不要为了凑数量重复领域课程。新Notebook从31起，不覆盖旧31本或用户内核。用小数组与清楚标注的构造数据理解机制，断言形状与数学关系，先预测→只改一个变量→运行→解释；不把构造样本包装为真实业务实验，不强行新训练三领域固定模型。

公用基础属于公用模块内部的学习路径，更新web/navigation.js、Notebook TARGETS、curriculum与API目录；原课程仍归原领域。需要可查看的教学辅助代码时复用现有工具，合理加入源码白名单。更新navigation.test.cjs关于公用缺口的断言为新增课程归属验证，不删除层级/旧链接/无重复归属检查。原31本需仍全部可达。

在已有.venv中执行与验收；不加不必要依赖，不安装到系统。新章逐本清洁内核执行和渲染验收，界面桌面/窄屏检查，适当pytest与四模块导航回归。实际完成后再移除公用首页的缺口提示。保留所有负结果，固定数据URL、SHA、正式模型与评估含义不改。

已有工作台18580与Jupyter18581只监听本机，保留认证与同源保护；仅可正常重启本项目工作台，不停Jupyter/用户内核/其他服务。数据、权重、日志、逐条预测和凭据不进Git。用户先前已授权实际实现、验收和push；网络/端口/GitHub操作遵循审批，不绕拒绝。完成后commit/push并核对远端SHA。
