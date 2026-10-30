# 在另一台 Apple Silicon Mac 恢复

本包目标为 M1/M2/M3/M4/M5 等 Apple Silicon Mac，至少16GiB内存且macOS 14或更新版本（锁定的PyTorch wheel要求）。它保存学习项目的源文件、16本Notebook及输出、两套真实数据、成功模型、实验配置和45份讲义。无需向任何外部Git仓库推送。

原始包也可以保存在自己的OpenAI项目文件或dots可用的云文件位置，但具体文件类型、大小限制和下载流程需在你的账号实际验证。**生成本地包不等于已上传，也不证明另一台M1已通过验收。** 当前环境没有直接向dots上传任意项目包的接口。

## 文件要齐全

将整个恢复包目录的文件放到新机器的同一个目录，包括所有 `content-*.zip`、`input-*.zip`、`manifest.json`、`restore.py`、`SHA256SUMS`和两份说明。content文件是独立ZIP，不是需要手工合并的分卷；input是原始数据文件，由恢复器放到正确位置，不要自己解压。

源文件、数据和产物逐项记录SHA-256。普通ZIP分组的未压缩内容预算96MiB；原始Retailrocket压缩包另放，约291MiB。实际云端是否接受这些文件尚未验证；不要因为聊天能同步就假定文件也已传输。

## 给目标机器的 Codex

在目标Mac打开本目录，将 `RESTORE_PROMPT.md` 的文字交给Codex。它可以读取目录、恢复项目和运行验收，不需要原聊天全文或账户凭据。主目录建议 `~/local/study-1029`，可以换名字，但必须是一个尚不存在的新目录。

若你亲手执行，从包含restore.py的恢复包目录运行：

```bash
shasum -a 256 -c SHA256SUMS
python3 restore.py verify
python3 restore.py restore --destination "$HOME/local/study-1029"
python3 restore.py setup --destination "$HOME/local/study-1029"
python3 restore.py validate --destination "$HOME/local/study-1029" --notebooks
```

Python3用于读取包，项目自身Python3.12由uv按锁文件安装。若没有uv，setup会在新项目内创建`.restore-tools`并从PyPI安装固定版本uv，不改全局Python。需要联网下载Python及依赖，这不是完全离线环境镜像。

本机库依赖libomp：缺失时，使用你已安装的Homebrew运行 `brew install libomp`，然后重试setup。没有Homebrew时由目标Codex说明缺失项，不承诺无人工步骤，也不自动运行sudo或替你接受系统授权。

`restore`拒绝覆盖现有目录。若setup因下载失败中断，保留已恢复目录，修复原因后直接重试setup；不需要再次restore。校验文件不会执行项目代码；setup和validate才会安装依赖/执行测试。

## 启动与学习

验收通过后，在新项目目录运行：

```bash
./lab.sh up
./lab.sh notebook
```

工作台：[打开](http://127.0.0.1:18580/)。Notebook入口：[课程](http://127.0.0.1:18580/notebooks?target=course) 或 [竞赛](http://127.0.0.1:18580/notebooks?target=xmrec_start)。使用网关地址，避免直接打开18581后看到密码框。新机器会生成新的本地认证信息。

如果18580/18581已被其他项目占用，先让Codex检查归属，不停止归属不明的服务。验收命令不自动启动任何长期服务。

M1 16GiB建议一次打开一个训练Notebook。竞赛图训练最多4线程，LightGBM单线程；不复现2048维完整冠军配方。上一轮主实验峰值约1.39GiB，只是单训练进程，不是整机总内存；M1性能和整机内存使用必须在目标机实测。

## 恢复边界

- 保留完整官方数据和现有成功模型；不自动重训旧Retailrocket模型，也不重跑已保留的隐藏测试评估。
- validate会跑完整测试集、再执行11～15的30格竞赛Notebook。13/14包含真实沙盒训练，会生成新的沙盒结果，不覆盖主实验。
- 不搬`.venv`、运行日志、Jupyter会话/凭据、Codex登录/聊天数据库、无关流lab或未完成的竞赛尝试。
- 讲义路径改成项目内`resources/slides`。其余迁移文件按清单校验；新建的环境、验收记录和实验输出不属于原始快照。
- `python3 restore.py verify`检查包完整性；文件清单校验不是来源数字签名，只恢复你自己可信来源的包。
- 文件可逐字节核对；跨机器重新训练按指标和数值容差验收，不保证浮点结果逐位相同。
- 包不含`.git`和任何远端配置。需要后续版本管理时，可以让Codex在新项目内建立本地Git仓库；不需要推送GitHub。
