请在这台Apple Silicon Mac上恢复这个学习项目，使用同目录restore.py、manifest.json和所有数据包，不需要外部Git仓库。

目标目录：~/local/study-1029（必须是新目录；存在则保留并另选名字，不清空、不覆盖）。先阅读START_HERE.md并检查包校验和。按顺序执行verify、restore、setup、validate --notebooks。使用目标项目自己的Python3.12虚拟环境和reclab内核，不复用原机器路径或虚拟环境。

项目有16本Notebook：00导航，01~10为Retailrocket与推荐课程，11~15为WSDM Cup 2022 XMRec冠军核心复现。保留代码、真实数据、已有模型和讲义；M1 16GiB采用CPU，图训练最多4线程，LightGBM和原生Booster预测保持单线程。不要扩大到2048维完整冠军配置，不要声称复现隐藏测试成绩。Notebook优先，允许我修改代码做实验，继续遵守docs/NOTEBOOK_FIRST.md。

缺少依赖先精确报告：setup可安装项目内uv和锁定依赖；libomp缺失时用已有Homebrew安装。不要为安装而随意修改全局Python或运行不明脚本。没有可用系统安装工具时说明具体缺失，不伪造安装成功。不要迁移任何账户密码、token、浏览器会话、Codex聊天数据库或其他项目。

验收时区分文件校验、编译、测试、Notebook执行和实际服务。检查45份本地讲义能读取、16个Notebook入口齐全、真实推荐/特征API和竞赛快照可用。验证成功后再启动本机18580工作台与18581 Jupyter；已有端口先检查归属，不停止其他服务。将准确命令、结果、机器配置和限制保存在本机恢复验收记录，给我工作台和Notebook入口。原机器上的测试结果只能作参考，不能当本机验收。
