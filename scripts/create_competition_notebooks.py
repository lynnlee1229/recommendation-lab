"""Scaffold only NEW notebooks; never overwrite a learner's edited notebook."""
from pathlib import Path
import inspect
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys, json, inspect
ROOT = Path.cwd()
if not (ROOT / "reclab").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Code
from reclab.competition import data
from reclab.competition.teaching import load_run, style, bar_scores
style()
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_rows", 40)
run, rows, features = load_run()
print("实际训练快照:", run.name)'''


def md(text):
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text).strip())


def code(text):
    return nbf.v4.new_code_cell(inspect.cleandoc(text).strip())


def save(stem, cells):
    path = ROOT / 'notebooks' / (stem + '.ipynb')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Search Recommendation Ads Lab (Python 3.12)', 'language': 'python', 'name': 'reclab'},
        'language_info': {'name': 'python', 'version': '3.12'},
        'reclab': {'mode': 'real-competition-core-reproduction', 'source': 'WSDM Cup 2022 XMRec'}})
    nbf.validate(nb)
    nbf.write(nb, path)


def main():
    save('11_XMRec_Start', [
        md('''# 11 · 真实竞赛：从题目走到一条预测
        ## 目标
        为缺少交互的市场 t1、t2 推荐商品，利用 s1、s2、s3 的共享商品信息。
        你将读取**完整官方数据**，从 100 件给定商品中排序。每个用户只有一个验证正例，成功指标是 nDCG@10。

        本轮以 [OPDAI 冠军公开方案](https://github.com/opdai/wsdm2022-xmrec-top1-solution)为依据。
        [官方排行榜](https://xmrec.github.io/wsdmcup/)冠军分数 0.6773 是当年隐藏测试成绩，**本地不会把验证成绩冒充这个分数**。
        这里复现了冠军核心算法与两阶段流程，缩小图模型、特征集合和集成规模。完整差异见[复现说明](../docs/XMREC_REPRODUCTION.md)。

        顺序：[12 特征](12_XMRec_Features.ipynb) → [13 采样与图模型](13_XMRec_Graph.ipynb) → [14 重训排序器](14_XMRec_Ranking.ipynb) → [15 评估](15_XMRec_Evaluation.ipynb)。
        ## 准备
        当前内核使用项目 .venv。下载只需首次执行 `python -m reclab.competition prepare`；本轮已经下载并核对校验和。'''),
        code(SETUP),
        md('''## 步骤 1 · 先明确哪些文件能进训练
        `train.tsv` 是评分交互；`train_5core.tsv` 是经过筛选的正反馈，不是未点击曝光。
        冠军将 5core 的评分改成 5，再按 user-item 去重，优先保留原 train 的记录。
        目标市场的 qrel 不能进入特征训练语料。官方没有公开隐藏测试标签。'''),
        code('''summary = []
        for market in data.MARKETS:
            train = data.interactions(market)
            combined = data.corpus([market])
            summary.append({"market": market, "raw_train_rows": len(train),
                            "duplicate_user_item_pairs": int(train.duplicated(["userId", "itemId"]).sum()),
                            "merged_rows": len(combined), "users": combined.userId.nunique(),
                            "items": combined.itemId.nunique()})
        summary = pd.DataFrame(summary)
        display(summary)
        display(data.interactions("t1").head(5))'''),
        code('''fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(summary.market, summary.merged_rows, color="#006e60")
        ax.set(xlabel="Market", ylabel="Deduplicated interactions", title="Official training corpus by market")
        ax.grid(axis="y", alpha=.15)
        plt.show()'''),
        md('''## 步骤 2 · 跟踪一个真实用户
        每行不是一次曝光；它是竞赛给定的候选商品之一。`label=0` 表示不是本次被留出的那个正例，不能解释成真实业务“不喜欢”。
        用户和商品 ID 都已匿名化，数据没有时间戳、商品标题、价格或可解释的国家名。不能在这里伪造“最近 7 天热度”等特征。'''),
        code('''user = rows.query("market == 't1'").userId.iloc[0]
        example = rows[(rows.market == "t1") & (rows.userId == user)]
        known = data.corpus(["t1"])
        display(known[known.userId.eq(user)].head(10))
        display(example.head(8))
        print("候选数:", len(example), "正例数:", int(example.label.sum()))
        print("正例 ID:", example.loc[example.label.eq(1), "itemId"].iloc[0])'''),
        md('''## 步骤 3 · 查看实际运行结果的来源
        全数据特征生成后，官方验证用户按稳定哈希分为约 60% fit、20% select、20% audit。
        fit 训练排序器，select 选模型，选定后才查看 audit。所有候选行跟着用户一起分组。
        **原方案是 10 折行级 bagging；此处改成用户隔离的本地实验流程。**'''),
        code('''receipt = json.loads((run / "receipt.json").read_text())
        display(pd.Series(receipt))
        display(rows.groupby(["market", "partition"]).userId.nunique().unstack())
        assert rows.groupby(["market", "userId"]).partition.nunique().eq(1).all()
        source = json.loads(data.SPEC.read_text())
        print("官方数据版本:", source["official_revision"])
        print("冠军源码版本:", source["winner_revision"])
        print("原始文件校验数:", len(source["files"]))'''),
        md('''## 检查理解
        如果把 qrel 的正例加入 t1 的交互矩阵，再评估这批用户，为什么分数会变好？
        <details><summary>答案路径</summary>模型已经看到评估目标；这不是泛化提升。跨市场共享的是已知训练交互里的商品关系，不是把目标答案加入图。</details>
        ## 下一步
        打开 12。先预测：源市场数据更多，是否一定使 t1、t2 都变好？后面用真实分数验证。''')
    ])
    save('12_XMRec_Features', [
        md('''# 12 · 冠军核心：把协同过滤分数变成特征
        ## 目标
        手算一个 ItemCF 相似度，核对原版代码，解释一条真实推荐。
        “特征工程”在这里是可查看的数字：市场商品计数、评分统计、不同市场语料训练的模型分数。
        ## 准备
        [可编辑实现](../reclab/competition/memory.py) · [冠军 icf_v2 原文件](../data/xmrec/upstream/winner/src/retrieval/icf_v2.py)。'''),
        code(SETUP),
        md('''## 步骤 1 · 手算 IUF
        二值历史矩阵 B，用户 u 的历史长度为 n_u：
        `S_ij = sum_u[B_ui B_uj / log(1+n_u)] / sqrt(count_i count_j)`，令对角线为 0。
        活跃用户连接很多商品，因此共现权重更小。用户候选分数为其历史商品到候选商品的相似度之和。'''),
        code('''B = np.array([[1., 1., 0.], [0., 1., 1.], [1., 1., 1.]])
        weights = 1 / np.log1p(B.sum(axis=1))
        cooccur = B.T @ (weights[:, None] * B)
        counts = B.sum(axis=0)
        similarity = cooccur / np.sqrt(counts[:, None] * counts[None, :])
        np.fill_diagonal(similarity, 0)
        manual_01 = (1 / np.log(3) + 1 / np.log(4)) / np.sqrt(2 * 3)
        assert np.isclose(similarity[0, 1], manual_01)
        display(pd.DataFrame(similarity, index=["item A", "item B", "item C"], columns=["item A", "item B", "item C"]))
        print("用户历史[A,B] -> C:", similarity[0, 2] + similarity[1, 2])'''),
        md('''## 步骤 2 · 与固定版本的冠军代码数值对照
        只提取经过校验的原类用于对照，不执行它的脚本入口、安装器或路径配置。
        IUF 和 rating-cosine 是两种分数；原方案 rating-cosine 查询时也对历史做**无评分权重求和**。'''),
        code('''from reclab.competition.reference import verify_references
        parity = verify_references()
        display(parity)
        from reclab.competition.memory import ItemCF
        display(Code(inspect.getsource(ItemCF.fit), language="python"))'''),
        md('''## 步骤 3 · 拆开真实候选分数
        载入全数据实际训练产物，不重新训练现有模型。相似度贡献之和必须等于候选分数。
        可以直接修改 `USER` 为同市场其他用户。'''),
        code('''from reclab.competition.experiment import load_memory
        model = load_memory(run / "memory/t1_local")
        USER = rows.query("market == 't1'").userId.iloc[0]
        subset = rows[(rows.market == "t1") & rows.userId.eq(USER)].copy()
        subset["score"] = model.score(subset)
        display(subset.sort_values("score", ascending=False).head(10))
        ITEM = subset.sort_values(["score", "itemId"], ascending=[False, True]).itemId.iloc[0]
        explanation = pd.DataFrame(model.explain(USER, ITEM))
        display(explanation.head(10))
        assert np.isclose(explanation.similarity.sum(), subset.loc[subset.itemId.eq(ITEM), "score"].iloc[0])'''),
        md('''## 步骤 4 · 同一 user-item 到底有哪些特征
        一行同时包含多个市场的商品计数、t1 自身统计和多个模型分数。ID 和 label 不进入排序特征。
        没有真实属性表，这套竞赛复现不会假造类目或用户画像。'''),
        code('''position = subset.index[0]
        display(features.loc[position].rename("value").to_frame())
        assert not {"label", "userId", "itemId", "partition"} & set(features)
        table = pd.read_csv(run / "metrics.csv")
        selected = table[(table.partition == "select") & (table.market.isin(["t1", "t2"])) & table.model.isin(["icf_local", "icf_s1", "icf_s3", "icf_all_iuf"])]
        display(selected.pivot(index="model", columns="market", values="ndcg10"))'''),
        code('''columns = ["icf_local", "icf_s1", "icf_s3", "icf_all_iuf", "icf_all_rating"]
        correlation = features[columns].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="BrBG")
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
        ax.set_yticks(range(len(columns)), columns)
        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(j, i, f"{correlation.iloc[i,j]:.2f}", ha="center", va="center", color="white" if abs(correlation.iloc[i,j]) > .65 else "black")
        ax.set_title("Pre-rank feature correlation · all official validation candidates")
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        fig.tight_layout()
        plt.show()'''),
        md("## 对照论文的实际分数\n\n使用完整官方验证集对照论文 Table 4；本地确定性同分规则不同，允许报告小残差。"),
        code('''full = table[(table.partition == "all_official_valid") & (table.model == "icf_local")][["market", "users", "ndcg10"]].copy()
        full["paper_table4"] = full.market.map({"t1": .6776, "t2": .5589, "combined": .5980})
        full["local_minus_paper"] = full.ndcg10 - full.paper_table4
        display(full)'''),
        md('''## 检查与下一步
        相关系数接近 1 的两个特征是否只能删掉一个？
        <details><summary>答案路径</summary>不能仅凭相关性决定。先在 select 用户上做固定配置的消融；它们可能在某个市场或长尾用户上有互补信息。不要盯着 audit 反复挑特征。</details>
        修改 `ItemCF.fit` 的权重、设置 `top_k` 都会改变分数，需要重新生成特征。进入 13 看可训练的表示。''')
    ])
    save('13_XMRec_Graph', [
        md('''# 13 · 负采样、BPR 与 LightGCN
        ## 目标
        亲自抽三元组 `(user, positive, unknown)`，看图传播的张量，再运行一个小训练。
        这不是把人工张量包装成竞赛效果：机制演示会明确标识，训练使用真实 t1 历史。
        ## 准备
        [模型和训练循环](../reclab/competition/graph.py) · [冠军原模型](../data/xmrec/upstream/winner/src/retrieval/lgcn_v2/model.py)。'''),
        code(SETUP),
        md('''## 步骤 1 · 抽样规则就是模型假设
        先均匀选用户，再从其已知历史中均匀选正例，从未见商品里拒绝采样负例。
        这些“负例”只是未知项，可能含未来喜欢的商品。`negative_power=.75` 改为按热度的 .75 次幂抽样；它与均匀抽样优化的分布不同。'''),
        code('''from reclab.competition.data import encode
        from reclab.competition.graph import NegativeSampler, normalized_graph, LightGCN, GraphConfig, train_graph, graph_score
        import torch
        real = data.corpus(["t1"])
        users, items, history = encode(real)
        history.data[:] = 1
        draws = {}
        for power in [0., .75]:
            sampler = NegativeSampler(history, power=power, seed=2022)
            u, p, n = sampler.sample(20000)
            assert sampler.is_known(u, p).all() and not sampler.is_known(u, n).any()
            popularity = np.asarray(history.sum(axis=0)).ravel()
            draws[power] = popularity[n]
            print("power:", power, "负例平均训练热度:", round(float(popularity[n].mean()), 2))
        display(pd.DataFrame({"user": users[u[:5]], "positive": items[p[:5]], "sampled_unknown": items[n[:5]]}))'''),
        code('''fig, ax = plt.subplots(figsize=(8, 3))
        for power, values in draws.items():
            ax.hist(np.log1p(values), bins=25, histtype="step", linewidth=2, density=True, label=f"sampling power={power}")
        ax.set(xlabel="log(1 + training item degree)", ylabel="Density", title="Which unknown items become training negatives?")
        ax.legend(); plt.show()'''),
        md('''## 步骤 2 · 构造张量核对传播
        下面只有 3 个用户、4 个商品，是**构造的机制演示**。
        邻接矩阵 A 在两侧分别连接 user 与 item；归一化 `D^-1/2 A D^-1/2` 后传播。
        最终表示是第 0 到 L 层的平均，不含 MLP、非线性和额外特征变换。'''),
        code('''from scipy import sparse
        toy = sparse.csr_matrix([[1,1,0,0], [0,1,1,0], [0,0,1,1]], dtype=np.float32)
        graph = normalized_graph(toy)
        torch.manual_seed(2022)
        net = LightGCN(3, 4, dim=8, layers=2)
        user_vectors, item_vectors = net.propagate(graph)
        print("邻接矩阵", tuple(graph.shape), "用户向量", tuple(user_vectors.shape), "商品向量", tuple(item_vectors.shape))
        u0, p0, n0 = torch.tensor([0,1]), torch.tensor([0,1]), torch.tensor([3,0])
        loss, reg = net.bpr_loss(graph, u0, p0, n0)
        loss.backward()
        print("BPR:", float(loss.detach()), "ego L2:", float(reg.detach()))
        print("Embedding gradient norm:", float(net.item_embedding.weight.grad.norm()))
        assert torch.isfinite(net.item_embedding.weight.grad).all()'''),
        md('''## 步骤 3 · 改代码并运行真实训练
        这格会实际训练，不覆盖主实验。第一次用 16 维、4 个 epoch，允许你在几十秒内看完整过程。
        修改 `layers=0` 可做相同训练预算的 MF 对照；修改 `negative_power=.75` 看采样影响。
        BPR loss 下降不保证候选排名上升。修改 .py 后，先运行下格的 reload，再重跑训练格。'''),
        code('''import importlib
        import reclab.competition.graph as graph_module
        importlib.reload(graph_module)
        CONFIG = graph_module.GraphConfig(dim=16, layers=2, epochs=4, steps_per_epoch=2, negative_power=0.)
        sandbox = graph_module.train_graph(real, CONFIG)
        target = data.candidates("t1")
        sandbox_scores = graph_module.graph_score(sandbox, target)
        from reclab.competition.metrics import evaluate
        display(evaluate(target, sandbox_scores))
        # 保存到本轮 Notebook 沙盒文件，和主实验分开。
        from datetime import datetime, timezone
        sandbox_path = data.OUTPUT / "experiments" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-graph")
        sandbox_path.mkdir(parents=True)
        torch.save({"state_dict": sandbox["model"].state_dict(), "config": sandbox["config"]}, sandbox_path / "model.pt")
        (sandbox_path / "trace.json").write_text(json.dumps(sandbox["trace"], indent=2))
        print("沙盒保存:", sandbox_path)'''),
        md('''## 步骤 4 · 回看主实验与资源差异
        主实验用完整五市场语料；下图是实际运行保存的训练曲线。
        冠军论文配置是 2,048 维、4 层、学习率 .001、dropout .4；本机版是 32 维、2 层、学习率 .01、无 dropout，且只做少量大批次更新。
        数值对照证明核心公式一致，不证明训练配方或最终分数相同。'''),
        code('''trace = json.loads((run / "graph_trace.json").read_text())
        fig, ax = plt.subplots(figsize=(8, 3))
        for layers, values in trace.items():
            t = pd.DataFrame(values)
            ax.plot(t.epoch, t.bpr, marker="o", label=f"layers={layers}")
        ax.set(xlabel="Epoch", ylabel="Sampled BPR loss", title="Actual full-data training · same update budget")
        ax.legend(); plt.show()
        display(pd.Series(json.loads((run / "config.json").read_text())["graph"]))'''),
        md('''## 检查与下一步
        为什么 loss 可以很低，官方 100 候选的 nDCG 仍然差？
        <details><summary>答案路径</summary>训练负例来自全商品空间，可能太容易；评估候选的分布、训练预算、正例稀疏性和模型容量都会影响排名。要查看相同候选上的指标，不能用 loss 替代。</details>
        进入 14，把多个分数交给 LightGBM。''')
    ])
    save('14_XMRec_Ranking', [
        md('''# 14 · 亲手重训融合模型
        ## 目标
        查看一行排序样本，训练 LightGBM，改变特征或负样本数量，再解释一条预测。
        冠军使用 **binary classifier** 给候选打分，不是 LambdaRank；这里保留这个目标函数。
        ## 准备
        [排序器代码](../reclab/competition/ranking.py) · [完整实验入口](../reclab/competition/experiment.py)。
        以下训练格使用真实 t1 的全部官方候选与固定用户分区；只展示 select 成绩，方便你做实验。'''),
        code(SETUP),
        md('''## 步骤 1 · label 与特征必须分开
        每个用户 1 个正例、99 个候选负例。候选是官方提前给定的，不是在知道答案后“补进召回池”。
        特征矩阵中没有用户 ID、商品 ID、label 或分区。fit、select、audit 均按完整用户分组。'''),
        code('''first_user = rows.query("market == 't1' and partition == 'fit'").userId.iloc[0]
        mask = rows.market.eq("t1") & rows.userId.eq(first_user)
        graph_columns = [c for c in features if c.startswith("graph_layers") and c != "graph_layers0"]
        inspect_columns = ["target_user_rating_count", "target_item_rating_count", "icf_local", "icf_s1", "icf_all_iuf"] + graph_columns[:1]
        sample = rows.loc[mask, ["userId", "itemId", "label", "partition"]].join(features.loc[mask, inspect_columns])
        display(sample.sort_values("label", ascending=False).head(8))
        print("样本正负比:", rows.loc[mask].label.value_counts().to_dict())'''),
        md('''## 步骤 2 · 重训一个真实模型
        修改三个变量之一，再运行本格和下一格：`COLUMNS`、`NEGATIVE_KEEP`、`LEAVES`。
        此处保留正例，只对 fit 的负例下采样；select/audit 的 100 候选保持原样。
        负采样后的概率不再能直接当原分布概率，更不能当线上 CTR。'''),
        code('''import importlib
        import reclab.competition.ranking as ranking_module
        import reclab.competition.teaching as teaching_module
        importlib.reload(ranking_module)
        importlib.reload(teaching_module)
        COLUMNS = list(features.columns)
        NEGATIVE_KEEP = 99
        LEAVES = 15
        model, selection, experiment_path = teaching_module.sandbox_ranker(
            columns=COLUMNS, negative_keep=NEGATIVE_KEEP, leaves=LEAVES, tag="manual-baseline")
        display(selection)
        print("本次配置/模型保存:", experiment_path)'''),
        md('''## 步骤 3 · 只改一个变量：99 个负例 → 9 个
        这格会再训练一次，标签、用户划分、特征和叶子数不变。
        留下不涨分的结果；不要因为某次消融不符合预期就删掉记录。'''),
        code('''sampled_model, sampled_selection, sampled_path = teaching_module.sandbox_ranker(
            columns=COLUMNS, negative_keep=9, leaves=LEAVES, tag="negative-sampling-9")
        comparison = pd.concat([selection.assign(experiment="99 negatives"), sampled_selection.assign(experiment="9 negatives")])
        display(comparison[comparison.market.eq("t1")][["experiment", "users", "ndcg10", "hr10"]])'''),
        md('''## 步骤 4 · 树在用什么信息
        gain 是训练过程的分裂增益，不能解释成因果贡献。下面再看同一条预测的 TreeSHAP log-odds 分解。
        正负贡献加上偏置等于模型 raw score，不等于概率。'''),
        code('''importance = pd.DataFrame({"feature": COLUMNS, "gain": model.booster_.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False)
        display(importance.head(10))
        fig, ax = plt.subplots(figsize=(9,4))
        top = importance.head(10).sort_values("gain")
        ax.barh(top.feature, top.gain, color="#006e60")
        ax.set(xlabel="LightGBM split gain (training)", title="Feature importance · t1 notebook experiment")
        fig.tight_layout(); plt.show()'''),
        code('''candidate = rows.query("market == 't1' and partition == 'select'").index[0]
        x = features.loc[[candidate], COLUMNS]
        contributions = model.booster_.predict(x, pred_contrib=True, num_threads=1)[0]
        raw = model.booster_.predict(x, raw_score=True, num_threads=1)[0]
        assert np.isclose(contributions.sum(), raw)
        display(pd.DataFrame({"feature": COLUMNS + ["bias"], "log_odds_contribution": contributions}).sort_values("log_odds_contribution", key=abs, ascending=False).head(12))
        print("raw score:", round(float(raw), 5), "probability:", round(float(model.predict_proba(x)[0,1]), 5))'''),
        md('''## 检查与下一步
        如果把 `icf_s1` 从 COLUMNS 删除，t1 的 select 成绩一定会下降吗？先预测，再重跑步骤 2。
        <details><summary>答案路径</summary>不一定。它可能冗余、有噪声，或者帮助的只是另一个市场。固定用户划分和训练配置做对照；单次小差异不能直接宣称稳定收益。</details>
        进入 15 查看主实验已经冻结的选择与 audit 对照。''')
    ])
    save('15_XMRec_Evaluation', [
        md('''# 15 · 评估是否可信，下一次该改哪里
        ## 目标
        手算官方指标，核对两市场的正确加权方式，查看主实验的固定审计结果和误差案例。
        **公开 audit 后，这批数据已成为可见基准。继续反复优化它，就不能再声称它是未触碰的测试集。**
        ## 准备
        [官方协议](https://xmrec.github.io/wsdmcup/) · [冠军论文](https://arxiv.org/abs/2203.00897) · [指标实现](../reclab/competition/metrics.py)。'''),
        code(SETUP),
        md('''## 步骤 1 · 一个正例的 nDCG
        正例在第 r 名且 r≤10 时为 `1/log2(r+1)`，否则为 0。所有用户取平均。
        官方合并 t1、t2 的用户后平均，因此不能简单把两个市场分数相加除以 2。'''),
        code('''from reclab.competition.metrics import evaluate, per_user
        display(pd.DataFrame({"positive_rank": [1,2,3,10,11], "ndcg10": [1/np.log2(r+1) if r <=10 else 0 for r in [1,2,3,10,11]]}))
        predictions = pd.read_parquet(run / "valid_predictions.parquet")
        selection = json.loads((run / "selection.json").read_text())
        chosen = selection["selected"]
        audit = rows.partition.eq("audit").to_numpy()
        report = evaluate(rows.loc[audit], predictions.loc[audit, chosen])
        display(report)
        by_market = report[report.market.ne("combined")]
        weighted = np.average(by_market.ndcg10, weights=by_market.users)
        assert np.isclose(weighted, report.iloc[-1].ndcg10)
        print("仅按 select 冻结的主模型:", chosen)'''),
        md('''## 步骤 2 · 所有对照用同一批 audit 用户
        `lgb_stats` 只有统计特征；`lgb_memory` 增加协同过滤；`lgb_all` 再加小图模型；`lgb_neg9` 只减少训练负例。
        全部配置在查看 audit 前固定。原冠军隐藏测试 0.6773 与本地 audit 不同，不画成同一条排行榜。'''),
        code('''metrics = pd.read_csv(run / "metrics.csv")
        table = metrics[(metrics.partition == "audit") & (metrics.market == "combined")].copy()
        display(table[["model", "users", "ndcg10", "hr10"]].sort_values("ndcg10", ascending=False))
        bar_scores(table, "Fixed audit users · local champion-core reproduction")'''),
        md('''## 步骤 3 · 给提升一个可检查的尺度
        对同一用户的分数差，按市场分层 bootstrap。区间反映这批用户抽样的不确定性，不能补偿反复看 audit 的选型偏差。
        这里比较的模型在 selection.json 中已冻结。'''),
        code('''from reclab.competition.teaching import paired_bootstrap
        interval = paired_bootstrap(rows.loc[audit], predictions.loc[audit, "icf_local"], predictions.loc[audit, chosen])
        display(interval)'''),
        md('''## 步骤 4 · 失败用户是什么样
        查看正例名次靠后的用户和它们的已知历史长度。错误案例用于形成下一次假设，不能据此直接改答案。
        例如：冷门商品缺少共现边、源市场口味不匹配、图模型训练不足。'''),
        code('''errors = per_user(rows.loc[audit], predictions.loc[audit, chosen])
        target_counts = rows[["market", "userId"]].join(features[["target_user_rating_count"]]).drop_duplicates(["market", "userId"])
        errors = errors.merge(target_counts, on=["market", "userId"], validate="one_to_one")
        display(errors.sort_values("rank", ascending=False).head(10))
        bins = pd.cut(errors.target_user_rating_count, bins=[-1,2,5,10,np.inf], labels=["0-2", "3-5", "6-10", "11+"])
        display(errors.assign(history_band=bins).groupby(["market", "history_band"], observed=True).agg(users=("userId","size"), ndcg10=("ndcg10","mean")))'''),
        md('''## 检查 · 这轮复现了什么
        - **直接对照通过**：原版 ItemCF 评分公式，原版 LightGCN 的传播/BPR/梯度；完整官方数据、100 候选与 nDCG@10。
        - **本机缩小或调整**：图模型维度/深度/训练更新数、特征数；用户隔离 fit/select/audit 替代原 10 折行级 bagging；不把另一目标市场 qrel 加进特征图。
        - **没有复现**：完整 Word2Vec/Node2Vec/NCF/Swing/LLR/BiGraph 多组合集成、null importance 大规模筛选、最终隐藏测试成绩。原源码保留供回查。

        ## 下一步 · 自己做一次完整实验
        建议先从 14 去掉一个市场特征，只看 select 成绩；再从 13 改采样或层数。
        改模型实现后需要重训；修改的图模型要接入主流程时运行下面命令，会创建新 run，不覆盖旧产物。'''),
        code('''print("cd", ROOT)
        print(".venv/bin/python -m reclab.competition run --tag my-first-ablation --dim 32 --layers 2 --epochs 12")
        print("当前 run 的配置:", run / "config.json")
        print("指标:", run / "metrics.csv")
        print("隐藏测试预测（无公开答案，不报分）:", run / f"test_{chosen}.parquet")''')
    ])


if __name__ == '__main__':
    main()
