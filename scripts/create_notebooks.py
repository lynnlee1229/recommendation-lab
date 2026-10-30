"""Author the two project tutorials with nbformat; execution is a separate step."""

from pathlib import Path
from inspect import cleandoc
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(cleandoc(text))


def code(text):
    return nbf.v4.new_code_cell(cleandoc(text))


SETUP = r"""
from pathlib import Path
import sys
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Code

PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents]
               if (p / "reclab/models.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from reclab.data import Dataset
from reclab.paths import ARTIFACTS, PROCESSED, read_json

plt.rcParams.update({"figure.figsize": (9, 3.8), "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#182b2b", "text.color": "#182b2b",
    "axes.facecolor": "white", "figure.facecolor": "white",
    "grid.color": "#d4ddda", "axes.axisbelow": True})
dataset = Dataset()
print("Project:", PROJECT)
print("Dataset version:", dataset.manifest["dataset_version"])
print("Python:", sys.executable)
assert Path(sys.executable).parent == PROJECT / ".venv/bin", "请选择 Recommendation Lab 内核"
"""


def first():
    return [
        md("""# 01 · 真实数据、特征与 ItemCF
        ## Goal
        沿着真实 Python 代码看一次推荐：数据文件 → 因果历史 → 用户特征 → ItemCF 得分 → 已看过滤。

        本 Notebook 默认只读已有数据和基线，不重新下载、不改原流计算 lab，也不更新模型。用 **Shift+Enter** 逐格执行；左侧文件浏览器可以打开并编辑 `reclab/*.py`。
        """),
        md("""## Setup
        内核选择 **Recommendation Lab (Python 3.12)**。输入来自本项目的 `data/processed/events.parquet`、`manifest.json` 和 `artifacts/baselines/`；数据源为 [Retailrocket](https://www.kaggle.com/retailrocket/ecommerce-dataset/home)。

        当前任务是训练截止之后的**暖商品、新物品交互召回**。浏览是行为，不是推荐曝光；未交互不等于曝光未点击。"""),
        code(SETUP),
        md("""## Steps
        ### 1. 先看文件和实际时间边界
        这些数字从冻结 manifest 读取，和网页是同一份数据。"""),
        code("""m = dataset.manifest
        display(pd.Series({k: m[k] for k in ["raw_events", "cohort_users", "catalog_items", "cohort_events", "feature_version"]}, name="value").to_frame())
        boundaries = pd.Series({"start": m["start_ms"], "train_end": m["train_end_ms"], "valid_end": m["valid_end_ms"], "end": m["end_ms"]})
        display(pd.to_datetime(boundaries, unit="ms", utc=True).to_frame("UTC boundary"))
        display(dataset.events.head(5))"""),
        md("""### 2. 用 SQL 检查 Parquet
        DuckDB 只在内存里查询，不启动任何容器。`item_idx=0` 表示不在训练词表中的物品。"""),
        code('''import duckdb
        with duckdb.connect(":memory:") as con:
            split_summary = con.execute("""
                SELECT split, count(*) AS events,
                       count(DISTINCT visitorid) AS users,
                       count(*) FILTER (WHERE item_idx = 0) AS oov_events
                FROM read_parquet(?) GROUP BY split ORDER BY split
            """, [str(PROCESSED / "events.parquet")]).df()
        display(split_summary)'''),
        md("""### 3. 选一个用户，查看真正送入模型的历史
        修改 `USER_POSITION` 或 `HISTORY_WINDOW` 后，重跑以下几格。类目是用于解释的元数据，当前双塔没有把它作为输入。"""),
        code("""USER_POSITION = 0
        HISTORY_WINDOW = 20
        TOP_K = 10
        eligible, audit = dataset.evaluation("valid", 1000)
        user_idx = list(eligible)[USER_POSITION]
        history = dataset.histories[user_idx]
        user_events = dataset.train[dataset.train.user_idx == user_idx].sort_values("timestamp", kind="stable")
        print("User ID:", dataset.users[user_idx], "User index:", user_idx)
        print("Training-only history:", history[-HISTORY_WINDOW:])
        preview = user_events.tail(HISTORY_WINDOW)[["timestamp", "event", "itemid", "item_idx"]].copy()
        preview["timestamp"] = pd.to_datetime(preview.timestamp, unit="ms", utc=True)
        display(preview)
        assert user_events.timestamp.max() < m["train_end_ms"]"""),
        md("""### 4. 从真实样本倒查历史
        正目标是首次交互的新商品，历史必须严格早于它。先读函数源码，再检查一条实际样本。"""),
        code("""display(Code(inspect.getsource(Dataset.examples), language="python"))
        sample_users, sample_history, sample_targets, sample_times = dataset.examples(HISTORY_WINDOW)
        selected = np.flatnonzero(sample_users == user_idx)
        assert len(selected), "当前用户没有可用训练样本，请换 USER_POSITION"
        j = int(selected[0])
        context = sample_history[j][sample_history[j] != 0].tolist()
        target = int(sample_targets[j])
        prior = user_events[user_events.timestamp < sample_times[j]]
        display(pd.DataFrame([{"user_idx": user_idx, "history": context,
            "target_item_idx": target, "target_time_utc": pd.to_datetime(sample_times[j], unit="ms", utc=True)}]))
        assert context == prior.item_idx.astype(int).tail(HISTORY_WINDOW).tolist()
        assert target not in set(prior.item_idx)"""),
        md("""### 5. 看 ItemCF 源码并计算推荐
        这里使用实际保存的邻居矩阵。当前变体加入了活跃用户惩罚，保留每个商品 Top100 邻居；打分使用最近 50 次行为的衰减权重。"""),
        code("""from scipy import sparse
        from reclab.retrieval import itemcf, cf_scores, topk
        display(Code(inspect.getsource(itemcf), language="python"))
        similarity = sparse.load_npz(ARTIFACTS / "baselines/itemcf.npz")
        scores = cf_scores(similarity, history, dataset.n_items)
        recommendations = topk(scores, TOP_K, dataset.seen[user_idx])
        display(pd.DataFrame([{**dataset.items[i-1], "itemcf_score": float(scores[i])}
                              for i in recommendations]))
        assert not set(recommendations) & dataset.seen[user_idx]"""),
        md("""### 6. 一个分数是怎样累加出来的？
        对第一个候选，把每条历史的贡献展开，并核对它们的和。"""),
        code("""chosen_item = recommendations[0]
        recent = history[-50:]
        recency_weights = np.geomspace(0.2, 1.0, len(recent))
        contributions = pd.DataFrame([{"history_item_idx": i,
            "history_item_id": dataset.items[i-1]["item_id"],
            "similarity": float(similarity[i, chosen_item]), "weight": float(w),
            "contribution": float(similarity[i, chosen_item]) * w}
            for i, w in zip(recent, recency_weights)])
        display(contributions.sort_values("contribution", ascending=False).head(10))
        print("Sum of contributions:", contributions.contribution.sum())
        print("Saved scorer output:", float(scores[chosen_item]))
        assert np.isclose(contributions.contribution.sum(), scores[chosen_item], atol=1e-6)"""),
        md("""### 7. 对照已有基线结果
        同一验证窗口、全训练词表、训练期已看过滤。分数只覆盖当前可评估的活跃暖启动用户。"""),
        code("""baseline = read_json(ARTIFACTS / "baselines/metrics.json")
        result = pd.DataFrame([{"model": name, "Recall@20": row["metrics"]["recall@20"],
            "NDCG@20": row["metrics"]["ndcg@20"], "users": row["audit"]["evaluated_users"]}
            for name, row in baseline["models"].items()]).set_index("model")
        display(result)
        ax = (result["Recall@20"] * 100).plot.barh(color="#006e60", figsize=(8, 3))
        ax.set(xlabel="Recall@20 (%)", ylabel="Model",
               title=f"Frozen validation / {int(result.users.iloc[0])} eligible users")
        ax.set_xlim(left=0)
        for y, value in enumerate(result["Recall@20"] * 100):
            ax.text(value + 0.2, y, f"{value:.2f}%", va="center")
        ax.set_xlim(0, max(result["Recall@20"] * 100) * 1.2)
        plt.tight_layout(); plt.show()"""),
        md("""## Checks
        上面的断言验证了时间边界、因果样本、已看过滤，以及 ItemCF 分数的逐条贡献。原始精确数值仍保留在表格里。

        ## Next Steps
        先预测：把历史截成最近 3 个商品，哪些候选应该改变？将 `cf_scores(similarity, history[-3:], ...)` 作为单用户反事实对照；它不是重新训练后的离线指标。

        接着打开 `02_TwoTower_HandsOn.ipynb`。讲义对应：ItemCF、离散特征处理、双塔样本。"""),
    ]


def second():
    return [
        md("""# 02 · 双塔：源码、张量、梯度与真实训练
        ## Goal
        从可编辑的 Python 网络走到真实训练。你可以改维度、历史窗口、学习率，也可以在左侧文件浏览器中打开 `reclab/models.py` 修改正式网络。

        **Run All 会在最后创建一个新模型版本**，保留原来的数据和模型。本教程先做小 Batch 数学检查，再运行项目的共享训练器；不使用测试集。"""),
        md("""## Setup
        使用项目的 **Recommendation Lab (Python 3.12)** 内核。下面的用户和物品来自现有真实训练集；小 Batch 实验只是检查梯度，不能证明推荐效果提升。"""),
        code(SETUP),
        code("""import torch
        from torch import nn
        from torch.nn import functional as F
        import reclab.models as model_module
        from reclab.models import TwoTower
        torch.set_num_threads(4)
        torch.manual_seed(42)
        DIM = 32
        HISTORY_LEN = 20
        BATCH = 8
        TEMPERATURE = 0.1"""),
        md("""## Steps
        ### 1. 直接看当前网络源码
        这就是网页服务使用的类。要修改它，在左侧打开 `reclab/models.py`；保存后重启内核再从头执行，或使用最后训练格中的模块 reload。"""),
        code("""print(inspect.getfile(TwoTower))
        display(Code(inspect.getsource(TwoTower), language="python"))"""),
        md("""### 2. 用真实样本追踪张量形状
        用户 `[B]`、历史 `[B,N]`、目标 `[B]`。两侧编码后都是 `[B,D]`；用户向量乘目标向量转置得到 `[B,B]`。"""),
        code("""users_np, histories_np, targets_np, times_np = dataset.examples(HISTORY_LEN)
        users = torch.from_numpy(users_np[:BATCH])
        histories = torch.from_numpy(histories_np[:BATCH])
        targets = torch.from_numpy(targets_np[:BATCH])
        model = TwoTower(dataset.n_users, dataset.n_items, dim=DIM, use_history=True, dropout=0.0)
        model.eval()  # 固定本次数学诊断；eval 不等于关闭 autograd
        user_vectors = model.encode_user(users, histories)
        item_vectors = model.encode_item(targets)
        display(pd.DataFrame({"tensor": ["users", "histories", "targets", "user_vectors", "item_vectors"],
            "shape": [list(x.shape) for x in [users, histories, targets, user_vectors, item_vectors]]}))
        print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
        assert torch.allclose(user_vectors.norm(dim=1), torch.ones(BATCH), atol=1e-6)"""),
        md("""### 3. 把 sampled softmax 损失展开
        先预测：降低温度会让分布更尖锐，还是更平滑？

        每行的正确类别在对角线。其他目标是 batch 内采样负例，重复目标会屏蔽。这里不启用 LogQ，便于先核对基本数学；它们并不是真实“曝光未点击”。"""),
        code("""cosine = user_vectors @ item_vectors.T
        duplicate = targets[:, None].eq(targets[None, :])
        duplicate.fill_diagonal_(False)
        logits = (cosine / TEMPERATURE).masked_fill(duplicate, -1e9)
        logits.retain_grad()
        labels = torch.arange(BATCH)
        probabilities = logits.softmax(dim=1)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        expected_gradient = (probabilities.detach() - F.one_hot(labels, BATCH)) / BATCH
        assert torch.allclose(logits.grad, expected_gradient, atol=1e-6)
        display(pd.DataFrame(probabilities.detach().numpy(), index=users.tolist(), columns=targets.tolist()))
        print("Cross-entropy:", float(loss.detach()))
        print("dL/dlogit for row 0:", logits.grad[0].tolist())"""),
        code("""fig, ax = plt.subplots(figsize=(7.5, 5.5))
        image = ax.imshow(probabilities.detach().numpy(), cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(BATCH), [str(i) for i in targets.tolist()])
        ax.set_yticks(range(BATCH), [str(i) for i in users.tolist()])
        ax.set(xlabel="Target item index (column)", ylabel="User index (row)",
               title="Reconstructed training batch / softmax probability")
        for row in range(BATCH):
            for col in range(BATCH):
                value = float(probabilities[row, col].detach())
                text = "x" if duplicate[row, col] else f"{value:.2f}"
                ax.text(col, row, text, ha="center", va="center", color="white" if value > .55 else "#182b2b")
        fig.colorbar(image, ax=ax, label="Probability")
        plt.tight_layout(); plt.show()"""),
        md("""### 4. 先让一个小 Batch 过拟合
        这是训练代码的 sanity check。只更新 Notebook 内存中的新模型，不覆盖已保存的权重；小 Batch loss 下降不代表验证指标提高。"""),
        code("""optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        tiny_losses = []
        for step in range(30):
            optimizer.zero_grad(set_to_none=True)
            tiny_loss = model.loss(users, histories, targets, TEMPERATURE)
            tiny_losses.append(float(tiny_loss.detach()))
            tiny_loss.backward()
            optimizer.step()
        assert tiny_losses[-1] < tiny_losses[0]
        print("Tiny-batch loss:", tiny_losses[0], "->", tiny_losses[-1])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(range(1,31), tiny_losses, color="#006e60", marker=".")
        ax.set(xlabel="Optimization step", ylabel="Cross-entropy", title="Same 8 examples / overfit sanity check", ylim=(0, max(tiny_losses)*1.1))
        ax.grid(axis="y"); plt.tight_layout(); plt.show()"""),
        md("""### 5. 在本格里试着改网络
        `MyTwoTower` 是一个独立学习分支，只检查前向传播。把 `ReLU` 换成 `Tanh`，或改 `HIDDEN`，观察参数量和输出形状。

        这一格没有替换正式训练器。希望工作台使用新结构时，编辑真实的 `reclab/models.py`，然后运行下面的完整训练格。"""),
        code("""HIDDEN = 16
        class MyTwoTower(TwoTower):
            def __init__(self, n_users, n_items, dim):
                super().__init__(n_users, n_items, dim, use_history=True, dropout=0.0)
                self.user_mlp = nn.Sequential(nn.Linear(2*dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, dim))

        sandbox = MyTwoTower(dataset.n_users, dataset.n_items, DIM)
        print("Edited user tower:", sandbox.user_mlp)
        print("Output shape:", tuple(sandbox.encode_user(users, histories).shape))
        print("Parameters:", sum(p.numel() for p in sandbox.parameters()))"""),
        md("""### 6. 跑项目真实训练器，保存新版本
        先只改 `EXPERIMENT` 中一个变量。保存后的模型、配置、曲线会出现在网页“训练与实验”里（刷新页面）。它使用完整训练子集和固定验证口径，不使用上面的 8 条样本作为评估。

        reload 会读取你已经保存的 `models.py` / `train.py`。修改结构后从新模型开始训练；不要用不兼容的新结构加载旧 checkpoint。"""),
        code("""import importlib
        import reclab.train as train_module
        importlib.reload(model_module)
        train_module = importlib.reload(train_module)

        EXPERIMENT = train_module.TrainConfig(
            name="notebook / first-full-run", dim=32, history_len=20,
            epochs=4, batch_size=256, learning_rate=0.003,
            use_history=True, logq_correction=False, seed=42,
        )
        new_run = train_module.train(EXPERIMENT)
        print("Saved run:", new_run["run_id"])
        display(pd.Series(new_run["validation"]["metrics"], name="validation score").to_frame())"""),
        code("""curve = pd.DataFrame(new_run["curve"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(curve.epoch, curve.loss, color="#006e60", marker="o")
        axes[0].set(title="Training sampled-softmax", xlabel="Epoch", ylabel="Cross-entropy", ylim=(0, curve.loss.max()*1.1))
        axes[1].plot(curve.epoch, curve["validation_ndcg@20"], color="#006e60", marker="o")
        axes[1].set(title=f"Validation / {new_run['validation']['audit']['evaluated_users']} users", xlabel="Epoch", ylabel="NDCG@20 score", ylim=(0, max(curve["validation_ndcg@20"].max()*1.2, .001)))
        for ax in axes:
            ax.set_xticks(curve.epoch)
            ax.grid(axis="y")
        plt.tight_layout(); plt.show()"""),
        md("""## Checks
        本 Notebook 用断言检查了向量归一化、交叉熵梯度公式，以及真实小 Batch 能否优化。最终模型由验证 NDCG@20 选择 checkpoint，结果保存到 `artifacts/runs/<run_id>/`。

        ## Next Steps
        先写下预测，再仅改变 `history_len` 或 `learning_rate` 重跑完整训练格。训练 loss 和验证指标是否同向变化？用 `configs/no-history.json` 对照历史特征的作用。

        当前机制参考双塔模型、LastN 与 batch 内负样本讲义；这不是原论文数据与工业规模的完整复现。测试集保留给最终验收，继续实验不要围绕它调参。"""),
    ]


def main():
    OUT.mkdir(exist_ok=True)
    for filename, cells in [
        ("01_Data_Features_ItemCF.ipynb", first()),
        ("02_TwoTower_HandsOn.ipynb", second()),
    ]:
        path = OUT / filename
        if path.exists():
            raise FileExistsError(f"Preserve existing notebook: {path}")
        nb = nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "Recommendation Lab (Python 3.12)",
                    "language": "python",
                    "name": "reclab",
                },
                "language_info": {"name": "python", "version": "3.12"},
            },
        )
        nbf.validate(nb)
        nbf.write(nb, path)
        print(path)


if __name__ == "__main__":
    main()
