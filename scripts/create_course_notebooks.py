"""Create notebook-first course companions without overwriting user notebooks."""

from pathlib import Path
import json
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_notebooks import md, code, SETUP

CHAPTERS = [
    (
        "basics",
        "03_Foundations_AB",
        "基础、指标与 A/B",
        "01_Basics",
        3,
        "真实数据 + 模拟分桶",
    ),
    (
        "retrieval_course",
        "04_Retrieval_Methods",
        "召回、样本与检索",
        "02_Retrieval",
        12,
        "真实快照 + 机制练习",
    ),
    (
        "ranking",
        "05_Ranking_Multitask",
        "多目标、MMoE 与粗排",
        "03_Rank",
        6,
        "受控模拟，不是曝光 CTR",
    ),
    (
        "cross",
        "06_Feature_Crossing",
        "FM、DCN、LHUC 与 FiBiNET",
        "04_Cross",
        4,
        "张量与梯度机制",
    ),
    (
        "sequence",
        "07_Sequence_DIN_SIM",
        "LastN、DIN 与 SIM",
        "05_LastN",
        3,
        "真实历史 + 注意力教学变体",
    ),
    (
        "diversity",
        "08_Diversity_DPP",
        "多样性、规则与 DPP",
        "06_Rerank",
        5,
        "真实向量 + 数值验收",
    ),
    (
        "cold",
        "09_ColdStart_Traffic",
        "冷启动与流量调控",
        "07_ColdStart",
        6,
        "真实属性 + 模拟流量",
    ),
    (
        "engineering",
        "10_Engineering_Improvement",
        "系统改进与工程实验",
        "08_Improvement",
        6,
        "真实快照 + 受控漂移/社交模拟",
    ),
]
EXPERIMENTS = {}


def add(chapter, title, scope, source, question):
    EXPERIMENTS.setdefault(chapter, []).extend(
        [
            md(f"### {title}\n{scope}\n\n**先预测：** {question}"),
            code(source),
        ]
    )


add(
    "basics",
    "指标的分母与推荐漏斗",
    "模拟一份有曝光的日志。真实 Retailrocket 没有完整曝光分母。",
    """
impressions, clicks, carts, purchases = 10000, 800, 120, 36
rates = {"CTR (simulated)": clicks/impressions,
    "cart given click": carts/clicks, "purchase given cart": purchases/carts,
    "purchase per impression": purchases/impressions}
display(pd.Series(rates).to_frame("rate"))
assert np.isclose(rates["CTR (simulated)"]*rates["cart given click"]*rates["purchase given cart"], rates["purchase per impression"])
print("Real event types:", dataset.events.event.unique().tolist())
""",
    "把购买数除以浏览数和除以曝光数，分别回答什么问题？",
)
add(
    "basics",
    "全词表与采样评估",
    "固定分数的构造案例。候选变容易，不是模型变好。",
    """
scores = np.arange(100, dtype=float); positive = 50
full_rank = 1 + int((scores > scores[positive]).sum())
sampled = np.r_[np.arange(10), positive]
sampled_rank = 1 + int((scores[sampled] > scores[positive]).sum())
display(pd.DataFrame({"candidates":[100,len(sampled)],"rank":[full_rank,sampled_rank]},index=["full","sampled"]))
assert sampled_rank < full_rank
""",
    "删掉竞争者之后，Recall@10 会怎样变化？",
)
add(
    "basics",
    "同层互斥、跨层散列与 Holdout",
    "20,000 个模拟标识。有限样本不保证列联表绝对相等；这里不做真实因果实验。",
    """
from reclab.learning.systems import bucket
display(Code(inspect.getsource(bucket), language="python"))
ids = np.arange(20000)
a = np.array([bucket(int(i),"retrieval") for i in ids])
b = np.array([bucket(int(i),"ranking") for i in ids])
groups = pd.DataFrame({"retrieval":np.select([a<20,a<40],["A","B"],default="control"),
    "ranking":np.where(b<50,"C","control"),
    "holdout":[bucket(int(i),"holdout")<10 for i in ids]})
display(pd.crosstab(groups.retrieval, groups.ranking, normalize="all"))
assert not np.any((a<20) & ((a>=20)&(a<40)))
assert bucket(123,"retrieval") == bucket(123,"retrieval")
""",
    "两个层使用相同 salt，会发生什么？反转实验为什么需要等待长期指标？",
)

add(
    "retrieval_course",
    "UserCF 与 Swing",
    "构造的可手算矩阵。Swing 是不加用户权重、每个无序用户对只计一次的教学变体。真实 ItemCF 见第 01 本。",
    """
from reclab.learning.retrieval import (binary_matrix, usercf, swing,
    MatrixFactorization, objective_losses, kmeans, partition_retrieval,
    beam_search, SideInfoEncoder, mask_fields)
import torch
from torch.nn import functional as F
torch.set_num_threads(4); torch.manual_seed(42)
matrix = binary_matrix([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(3,2),(3,3)],4,4)
display(pd.DataFrame(matrix))
display(Code(inspect.getsource(swing),language="python"))
display(pd.DataFrame(swing(matrix)))
score, neighbors, similarity = usercf(matrix,0,2)
display(pd.DataFrame({"neighbor":neighbors,"cosine":similarity}))
assert np.allclose(swing(matrix),swing(matrix).T)
""",
    "让两个用户额外共同喜欢十件物品，Swing 的单对贡献会变大还是变小？",
)
add(
    "retrieval_course",
    "真实训练交互上的矩阵分解 / BPR",
    "只使用训练期。固定抽样负例可能包含潜在兴趣，训练 loss 下降不代表验证提升。",
    """
pairs = dataset.train[["user_idx","item_idx"]].drop_duplicates().head(1500).to_numpy()
u = torch.tensor(pairs[:,0],dtype=torch.long); p = torch.tensor(pairs[:,1],dtype=torch.long)
n = torch.randint(1,dataset.n_items,(len(pairs),))
n = torch.where(n==p, n%(dataset.n_items-1)+1, n)
mf = MatrixFactorization(dataset.n_users,dataset.n_items,dim=12)
opt = torch.optim.Adam(mf.parameters(),lr=.03); curve=[]
for step in range(25):
    opt.zero_grad(); loss=mf.bpr_loss(u,p,n)
    loss.backward(); opt.step(); curve.append(float(loss.detach()))
assert curve[-1] < curve[0]
display(Code(inspect.getsource(MatrixFactorization),language="python"))
plt.plot(curve,color="#006e60"); plt.xlabel("Step"); plt.ylabel("Fixed-pair BPR loss"); plt.ylim(bottom=0); plt.show()
""",
    "把向量维度从 12 改成 4，参数量和拟合能力会怎样变化？",
)
add(
    "retrieval_course",
    "三种损失、采样幂次与 LogQ",
    "loss 尺度不可跨定义比较。讲义的 50/50 困难负例需要排序淘汰日志，现有数据没有该日志。",
    """
p=torch.tensor([1.,2.,.5],requires_grad=True); n=torch.tensor([0.,1.,1.],requires_grad=True)
display(pd.Series({k:float(v.detach()) for k,v in objective_losses(p,n).items()}).to_frame("loss"))
counts=np.array([1.,10.,100.,1000.]); exponent=.75
sampling=counts**exponent; sampling/=sampling.sum()
display(pd.DataFrame({"count":counts,"uniform":.25,"power_sampling":sampling,"in_batch_prior":counts/counts.sum()}))
logits=torch.ones(1,4)
display(pd.DataFrame({"raw":logits.softmax(-1)[0].numpy(),
    "logq_corrected":(logits-torch.log(torch.tensor(sampling))[None,:]).softmax(-1)[0].numpy()}))
""",
    "指数改成 0 或 1，热门物品成为负例的概率会怎样变化？",
)
add(
    "retrieval_course",
    "真实物品向量缓存与分区近似检索",
    "和精确 Top20 的重合率只衡量索引误差，不是用户偏好 Recall。未部署大型 ANN 系统。",
    """
vectors=np.load(ARTIFACTS/"runs/baseline-history32/item_vectors.npy")[1:]
query=vectors[0]; exact=np.argsort(-(vectors@query))[:20]
centers, labels=kmeans(vectors,clusters=12,iterations=5)
rows=[]
for probes in [1,3,6,12]:
    found,count=partition_retrieval(vectors,query,centers,labels,probes=probes,k=20)
    rows.append({"probes":probes,"scored":count,"exact_top20_overlap":len(set(found)&set(exact))/20})
display(pd.DataFrame(rows))
assert rows[-1]["exact_top20_overlap"]==1.
""",
    "为什么增加 probes 通常会增加代价？全量和增量更新时缓存版本应如何配对？",
)
add(
    "retrieval_course",
    "Deep Retrieval 的路径搜索环节",
    "仅复现条件路径概率与 beam search，未实现神经路径学习和交替优化，不能称作完整 Deep Retrieval。",
    """
transition={():{"A":.6,"B":.4},("A",):{"x":.5,"y":.5},("B",):{"x":.99,"y":.01}}
display(Code(inspect.getsource(beam_search),language="python"))
display(pd.DataFrame([{"width":w,"best":beam_search(transition,width=w)[0][0],
    "probability":np.exp(beam_search(transition,width=w)[0][1])} for w in [1,2,4]]))
assert beam_search(transition,width=1)[0][0] != beam_search(transition,width=4)[0][0]
""",
    "局部概率最大，为什么不等于整条路径概率最大？",
)
add(
    "retrieval_course",
    "字段自监督与 Bloom 过滤",
    "使用真实训练截止属性做 SSL 梯度诊断；没有将辅助任务接入正式双塔。Bloom 使用构造的整数 ID，不等同于真实曝光日志。",
    """
cats={v:i+1 for i,v in enumerate(sorted({x["category"] for x in dataset.items}))}
ids=torch.arange(1,65); categories=torch.tensor([cats[x["category"]] for x in dataset.items[:64]])
encoder=SideInfoEncoder(dataset.n_items,len(cats)+1)
a=encoder(*mask_fields(ids,categories,.5)); b=encoder(*mask_fields(ids,categories,.5))
ssl=F.cross_entropy(a@b.T/.2,torch.arange(64)); ssl.backward()
assert torch.isfinite(ssl)
print("SSL mechanism loss:",float(ssl.detach()))
from reclab.learning.systems import BloomFilter
bloom=BloomFilter(bits=512,hashes=4)
for i in range(100):bloom.add(i)
assert all(bloom.contains(i) for i in range(100))
print("False positives on 5000 unseen IDs:",np.mean([bloom.contains(i) for i in range(100,5100)]))
print("Author/geo/social retrieval requires matching fields; see explicitly simulated social exercise in chapter 10.")
""",
    "Bloom 存储翻倍会怎样？字段 Mask 和隐藏层 dropout 有什么区别？",
)

add(
    "ranking",
    "条件概率、融分与负采样校准",
    "以下都是构造的概率。校准依赖已知采样机制，不可直接用于缺少曝光分母的日志。",
    """
from reclab.learning.networks import (calibrate_negative_sampling,
    fit_multitask, MMoE, ThreeTower)
import torch
from torch.nn import functional as F
torch.manual_seed(42); torch.set_num_threads(4)
pc=np.array([.2,.4,.15]); cart=np.array([.3,.1,.5]); pay=np.array([.4,.6,.3]); price=np.array([20.,100.,50.])
display(pd.DataFrame({"p_pay":pc*cart*pay,"expected_revenue":pc*cart*pay*price,"weighted_sum_proxy":pc+cart+pay}))
true=np.array([.01,.1,.5]); alpha=.1
sampled=true/(true+alpha*(1-true))
recovered=calibrate_negative_sampling(sampled,alpha)
assert np.allclose(true,recovered)
display(pd.DataFrame({"true":true,"sampled":sampled,"recovered":recovered}))
""",
    "条件概率连乘和任意预测分数相加分别表达什么？",
)
add(
    "ranking",
    "MMoE 的任务与 Gate",
    "受控合成标签；训练拟合不能解释为真实推荐收益。这里可直接修改任务相关度。",
    """
TASK_CORRELATION=.7
x=torch.randn(512,6); signal=x[:,0]+x[:,1]*x[:,2]
other=TASK_CORRELATION*signal+(1-TASK_CORRELATION)*(x[:,3]-x[:,4])
y=torch.stack([(signal>0).float(),(other>0).float()],-1)
model,curve=fit_multitask(x,y,steps=80)
logits,gates=model(x)
display(Code(inspect.getsource(MMoE),language="python"))
display(pd.DataFrame(gates.mean(0).detach().numpy(),index=["task1","task2"],columns=["expert1","expert2","expert3"]))
assert torch.allclose(gates.sum(-1),torch.ones_like(gates.sum(-1)),atol=1e-6)
assert curve[-1]<curve[0]
plt.plot(curve,color="#006e60");plt.xlabel("Step");plt.ylabel("Training multitask BCE");plt.ylim(bottom=0);plt.show()
""",
    "把任务相关度改为 0，Gate 一定分开、效果一定提升吗？",
)
add(
    "ranking",
    "播放时长加权 Logistic",
    "没有真实时长，因此用构造时长核对常数模型的数学解。最优 odds 是正例权重和 / 负例个数，不是普通 CTR。",
    """
durations=torch.tensor([10.,20.,30.]); negative_count=12
z=torch.nn.Parameter(torch.tensor(0.)); opt=torch.optim.Adam([z],lr=.1)
for _ in range(220):
    opt.zero_grad()
    loss=-(durations.sum()*F.logsigmoid(z)+negative_count*F.logsigmoid(-z))/15
    loss.backward();opt.step()
expected=float(durations.sum())/negative_count
print("Learned odds:",float(torch.exp(z.detach())),"closed form:",expected)
assert abs(float(torch.exp(z.detach()))-expected)<.05
""",
    "所有正例时长翻倍，最优 odds 会怎样变化？",
)
add(
    "ranking",
    "用户、物品与交叉三塔",
    "讲义粗排机制的小模型，非完整 COLD。随机特征只用于检查缓存等价性。",
    """
ranker=ThreeTower(dim=8)
u=torch.randn(1,8); items=torch.randn(40,8); cross=torch.randn(40,8)
cached=ranker.cached_score(u,items,cross)
looped=torch.stack([ranker.cached_score(u,items[i:i+1],cross[i:i+1])[0] for i in range(40)])
assert torch.allclose(cached,looped,atol=1e-6)
display(Code(inspect.getsource(ThreeTower),language="python"))
print("User encoding: cached=1, naive=40. Cross features still run per candidate.")
""",
    "哪些特征能缓存，哪些随候选或请求变化？",
)

add(
    "cross",
    "FM 的等价公式与梯度",
    "直接核对枚举交互和向量化表达式。",
    """
import torch
from reclab.learning.networks import fm_interaction,fm_explicit,CrossV2,LHUC,SEBilinear
torch.manual_seed(42);torch.set_num_threads(4)
fields=torch.randn(4,5,8,requires_grad=True)
a=fm_interaction(fields);b=fm_explicit(fields)
assert torch.allclose(a,b,atol=1e-5)
ga=torch.autograd.grad(a.sum(),fields,retain_graph=True)[0]
gb=torch.autograd.grad(b.sum(),fields)[0]
assert torch.allclose(ga,gb,atol=1e-5)
display(Code(inspect.getsource(fm_interaction),language="python"))
display(pd.DataFrame({"fast":a.detach().numpy(),"explicit":b.detach().numpy()}))
""",
    "字段数翻倍后，枚举交互数量怎样变化？",
)
add(
    "cross",
    "DCN V2 单层",
    "full-rank cross 的算子实验，不含原论文完整低秩 MoE 配方。",
    """
x0=torch.randn(4,6);layer=CrossV2(6)
out=layer(x0,x0)
assert torch.allclose(out,x0*(x0@layer.affine.weight.T+layer.affine.bias)+x0,atol=1e-6)
display(Code(inspect.getsource(CrossV2),language="python"))
print("Shape:",out.shape,"parameters:",sum(p.numel() for p in layer.parameters()))
""",
    "将 affine 的权重和偏置设成零，输出应是什么？",
)
add(
    "cross",
    "LHUC 与 SE / Bilinear",
    "LHUC 缩放隐藏单元；SE-style sigmoid gate 与共享 W 的 bilinear 是教学变体，不是完整 FiBiNET/PPNet。",
    """
lhuc=LHUC(3,12); hidden=torch.randn(4,12);context=torch.randn(4,3)
adjusted,scale=lhuc(hidden,context)
assert bool(((scale>0)&(scale<2)).all())
display(pd.DataFrame(scale.detach().numpy()).round(3))
se=SEBilinear(5,8);interaction,weights=se(fields.detach())
assert interaction.shape==(4,80)
display(Code(inspect.getsource(SEBilinear),language="python"))
display(pd.DataFrame(weights.detach().numpy(),columns=[f"field{i}" for i in range(5)]))
""",
    "门控是在选择字段、隐藏单元，还是候选物品？三者如何区分？",
)

add(
    "sequence",
    "真实 LastN 与均值池化",
    "使用训练截止已保存商品向量做表示诊断，不评估早期样本预测收益。",
    """
import torch
from reclab.learning.networks import TargetAttention
torch.manual_seed(42);torch.set_num_threads(4)
vectors=torch.tensor(np.load(ARTIFACTS/"runs/baseline-history32/item_vectors.npy"))
user=max(dataset.histories,key=lambda u:len(dataset.histories[u]))
ids=dataset.histories[user][-min(1000,len(dataset.histories[user])):]
history=vectors[ids][None,:,:];mask=torch.ones(1,len(ids),dtype=torch.bool)
assert torch.allclose(history.mean(1),history.flip(1).mean(1),atol=1e-6)
print("Real history length:",len(ids),"user:",dataset.users[user])
""",
    "逆序后均值为什么不变？改变窗口截断长度呢？",
)
add(
    "sequence",
    "DIN 风格的候选相关激活",
    "输入 h、q、h-q、h*q。这里是 softmax 教学变体，未实现 Dice/原论文正则；随机权重不代表已学好的兴趣。",
    """
attention=TargetAttention(vectors.shape[1])
short=history[:,-40:];short_mask=mask[:,-40:]
qa=vectors[100][None,:];qb=vectors[2000][None,:]
pa,wa=attention(short,qa,short_mask);pb,wb=attention(short,qb,short_mask)
assert torch.allclose(wa.sum(-1),torch.ones(1))
display(Code(inspect.getsource(TargetAttention),language="python"))
plt.plot(wa[0].detach(),label="candidate100",color="#006e60")
plt.plot(wb[0].detach(),label="candidate2000",color="#855519",linestyle="--")
plt.xlabel("History position");plt.ylabel("Attention weight");plt.title("Untrained activation / candidate sensitivity");plt.legend();plt.show()
""",
    "同一用户的表示为什么会随候选改变？这还能缓存成独立用户塔吗？",
)
add(
    "sequence",
    "让激活单元真正反向传播",
    "人为向量拟合任务，只检查可学习性，不是 CTR 标签。",
    """
optimizer=torch.optim.Adam(attention.parameters(),lr=.02);curve=[]
for _ in range(40):
    optimizer.zero_grad();pooled,_=attention(short,qa,short_mask)
    loss=torch.nn.functional.mse_loss(pooled,qa)
    loss.backward();optimizer.step();curve.append(float(loss.detach()))
assert curve[-1]<curve[0]
print("Artificial vector-fit loss:",curve[0],"->",curve[-1])
""",
    "权重变化是否意味着真实用户更感兴趣？还缺什么证据？",
)
add(
    "sequence",
    "SIM 风格的先检索后激活",
    "从真实长历史按类目和相似度筛 TopK。展示检索/激活解耦，未复现 SIM 完整训练与论文性能。",
    """
K=20;candidate_id=100;candidate=vectors[candidate_id][None,:]
category=dataset.items[candidate_id-1]["category"]
pool=[i for i in ids if dataset.items[i-1]["category"]==category] or ids
pool_vectors=vectors[pool]
selected=torch.topk(pool_vectors@candidate[0],min(K,len(pool))).indices
filtered=pool_vectors[selected][None,:,:]
with torch.no_grad():
    full,_=attention(history,candidate,mask)
    reduced,_=attention(filtered,candidate,torch.ones(1,filtered.shape[1],dtype=torch.bool))
print("Full/filtered tokens:",len(ids),filtered.shape[1])
print("Representation distance (not accuracy):",float(torch.norm(full-reduced)))
assert filtered.shape[1]<=K
""",
    "K 增加时，计算量与表示误差一定单调变化吗？",
)

add(
    "diversity",
    "真实向量上的 ILD、类目熵与 MMR",
    "ILD 衡量协同向量距离，不等同于独立图文语义。列表变化不是用户反馈。",
    """
from reclab.learning.systems import ild,category_entropy,constrained_ranking,greedy_dpp,slow_greedy_dpp,exhaustive_dpp
from reclab.retrieval import mmr
vectors=np.load(ARTIFACTS/"runs/baseline-history32/item_vectors.npy")
scores=vectors@vectors[100];pool=np.argsort(-scores[1:])[:80]+1
rows=[]
for lam in [1.,.8,.5]:
    selected,_=mmr(pool.tolist(),scores,vectors,k=10,relevance=lam)
    rows.append({"lambda":lam,"mean_similarity":float(scores[selected].mean()),
        "ILD":ild(vectors,selected),"category_entropy_bits":category_entropy([dataset.items[i-1]["category"] for i in selected])})
display(pd.DataFrame(rows))
""",
    "同类目多不一定意味着向量相似吗？",
)
add(
    "diversity",
    "规则与不可行约束",
    "若没有可行候选，本实现明确返回部分结果，不伪造满足规则的完整列表。",
    """
categories={i:dataset.items[i-1]["category"] for i in pool}
selected=constrained_ranking(pool.tolist(),categories,k=10,max_consecutive=2)
display(pd.DataFrame({"item":selected,"category":[categories[i] for i in selected]}))
for pos in range(2,len(selected)):
    assert not (categories[selected[pos]]==categories[selected[pos-1]]==categories[selected[pos-2]])
""",
    "候选全部来自一个类目时，算法应该放松规则还是少返回？",
)
add(
    "diversity",
    "DPP：体积、慢速贪心、快速 Cholesky 和穷举",
    "受控随机向量，便于数值验收。贪心 MAP 不是全局最优保证。",
    """
rng=np.random.default_rng(42)
v=rng.normal(size=(8,8));v/=np.linalg.norm(v,axis=1,keepdims=True)
quality=np.exp(np.linspace(0,1,8));features=v*quality[:,None];kernel=features@features.T
assert np.linalg.eigvalsh(kernel).min()>-1e-8
orthogonal=np.eye(2);duplicate=np.array([[1.,0.],[1.,0.]])
print("Orthogonal/duplicate volume squared:",np.linalg.det(orthogonal@orthogonal.T),np.linalg.det(duplicate@duplicate.T))
display(Code(inspect.getsource(greedy_dpp),language="python"))
fast=greedy_dpp(kernel,3);slow=slow_greedy_dpp(kernel,3)
best_logdet,best_ids=exhaustive_dpp(kernel,3)
assert fast==slow
greedy_logdet=np.linalg.slogdet(kernel[np.ix_(fast,fast)])[1]
assert greedy_logdet<=best_logdet+1e-8
display(pd.DataFrame([{"method":"greedy","items":fast,"logdet":greedy_logdet},{"method":"global exhaustive","items":best_ids,"logdet":best_logdet}]))
""",
    "把两个候选向量设成相同，Gram 行列式会怎样变化？",
)

add(
    "cold",
    "隐藏 ID 后只用类目召回",
    "训练目录内人为隐藏 ID 的压力练习，不是真实新发布商品效果。这里不用隐藏物品的已训练 ID 向量宣称冷启成功。",
    """
categories=sorted({x["category"] for x in dataset.items});lookup={c:i for i,c in enumerate(categories)}
side=np.eye(len(categories),dtype=np.float32)[[lookup[x["category"]] for x in dataset.items]]
user=next(iter(dataset.histories));history=dataset.histories[user]
profile=side[np.array(history[-20:])-1].mean(0);scores=side@profile
hidden_ids=set(range(200))
chosen=[i for i in np.argsort(-scores,kind="stable") if i in hidden_ids][:10]
display(pd.DataFrame([{"item":dataset.items[i]["item_id"],"category":dataset.items[i]["category"],"score":scores[i]} for i in chosen]))
""",
    "类目缺失或全部是 unknown 时，这条通道会怎样？",
)
add(
    "cold",
    "聚类与 Look-Alike 人群扩散",
    "真实类目与训练历史的机制实验。没有真实图文或作者字段，不复现 CNN/BERT 内容相似度训练。",
    """
from reclab.learning.retrieval import kmeans
centers,labels=kmeans(side[:600],clusters=8,iterations=5)
user_ids=list(dataset.histories)[:500]
profiles=np.array([side[np.array(dataset.histories[u][-20:])-1].mean(0) for u in user_ids])
seed_category=dataset.items[0]["category"]
seeds=[i for i,u in enumerate(user_ids) if any(dataset.items[j-1]["category"]==seed_category for j in dataset.histories[u])] or [0]
audience=profiles[seeds].mean(0)
audience=audience/max(float(np.linalg.norm(audience)),1e-8)
normalized=profiles/np.maximum(np.linalg.norm(profiles,axis=1,keepdims=True),1e-8)
expanded=np.argsort(-(normalized@audience))[:10]
display(pd.DataFrame({"user":[dataset.users[user_ids[i]] for i in expanded],"similarity":(normalized@audience)[expanded]}))
print("Cluster counts:",np.unique(labels,return_counts=True))
""",
    "如果种子群体偏了，扩散会修正偏差还是放大偏差？",
)
add(
    "cold",
    "保量、差异化目标与供需约束",
    "全部是模拟流量。100 次基础曝光只是讲义示例。供给不足不可能全保量；即使流量足够，固定 boost 也未必成功。",
    """
from reclab.learning.systems import allocate_quotas,bucket
quality=np.linspace(.1,1.,20);QUOTA=100;rows=[]
for requests in [500,2000,4000]:
    exposure=allocate_quotas(quality,requests,quota=QUOTA,boost=3.)
    rows.append({"requests":requests,"required_supply":len(quality)*QUOTA,
        "items_reaching_quota":int((exposure>=QUOTA).sum()),"allocated":int(exposure.sum())})
    assert exposure.sum()==requests
display(pd.DataFrame(rows))
assert rows[0]["items_reaching_quota"]<len(quality)
""",
    "保证数量、内容质量与消费体验能否同时最大化？",
)
add(
    "cold",
    "作者侧 A/B 的流量干扰",
    "构造两个作者池；共享预算下 A 增量会挤占 B。没有真实作者发布和用户留存的因果结论。",
    """
budget=10000;base=np.array([1.,1.]);boost=np.array([1.5,1.])
before=budget*base/base.sum();after=budget*boost/boost.sum()
display(pd.DataFrame({"before":before,"after":after,"delta":after-before},index=["authorA","authorB"]))
assert np.isclose((after-before).sum(),0)
print("Reproducible author buckets:",[bucket(i,"author",2) for i in range(6)])
""",
    "如何区分业务净增与实验组之间搬运流量？",
)

add(
    "engineering",
    "召回配额与去重",
    "真实训练快照中的热门、ItemCF 和双塔候选；候选数相加不等于去重后的数量。",
    """
from scipy import sparse
from reclab.retrieval import cf_scores,topk
from reclab.train import load_model,model_scorer
user=next(iter(dataset.histories));sim=sparse.load_npz(ARTIFACTS/"baselines/itemcf.npz")
model,report=load_model(dataset,"baseline-history32");scorer,_=model_scorer(dataset,model)
channels={"popular":topk(dataset.popularity,100,dataset.seen[user]),
    "itemcf":topk(cf_scores(sim,dataset.histories[user],dataset.n_items),100,dataset.seen[user]),
    "tower":topk(scorer(user),100,dataset.seen[user])}
display(pd.DataFrame([{"per_channel":n,"sum_before":3*n,
    "unique_after":len(set().union(*(set(ids[:n]) for ids in channels.values())))} for n in [10,30,100]]))
""",
    "固定总预算，调整配额时应同时观察哪些质量、覆盖与时延指标？",
)
add(
    "engineering",
    "粗精排蒸馏",
    "受控 teacher/student 分数；检查 KL 的方向和梯度，不宣称已有完整精排模型。",
    """
import torch
from reclab.learning.systems import distillation_loss
torch.manual_seed(42);torch.set_num_threads(4)
teacher=torch.randn(8,20);student=torch.nn.Parameter(torch.zeros(8,20))
opt=torch.optim.Adam([student],lr=.15);curve=[]
for _ in range(70):
    opt.zero_grad();loss=distillation_loss(student,teacher,2.)
    loss.backward();opt.step();curve.append(float(loss.detach()))
assert curve[-1]<curve[0]
plt.plot(curve,color="#006e60");plt.xlabel("Step");plt.ylabel("Listwise KL distillation loss");plt.ylim(bottom=0);plt.show()
""",
    "teacher 很差时，更好地拟合 teacher 是否一定更好？",
)
add(
    "engineering",
    "全量、增量与分布漂移",
    "受控二分类规律改变，展示继续训练与从头训练。训练准确率不等于泛化或线上老汤模型收益。",
    """
torch.manual_seed(42)
x=torch.randn(512,4);old_y=(x[:,0]+x[:,1]>0).float();new_y=(-x[:,0]+x[:,1]>0).float()
def fit(net,label,steps):
    optimizer=torch.optim.SGD(net.parameters(),lr=.2)
    for _ in range(steps):
        optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1),label)
        loss.backward();optimizer.step()
    return net
old=fit(torch.nn.Linear(4,1),old_y,80)
import copy
incremental=fit(copy.deepcopy(old),new_y,20);fresh=fit(torch.nn.Linear(4,1),new_y,20)
display(pd.DataFrame([{"model":name,"new_distribution_training_accuracy":float(((net(x).squeeze(-1)>0)==new_y.bool()).float().mean())}
    for name,net in [("old",old),("incremental20",incremental),("fresh20",fresh)]]))
""",
    "旧数据继续保留多少，才能兼顾遗忘与新分布适应？",
)
add(
    "engineering",
    "特殊人群、多样性、探索与社交",
    "按训练期活跃度分层；关注/作者/站外回流缺失，社交关系明确使用构造字典。探索比例不能由静态日志验证长期收益。",
    """
activity=pd.Series({u:len(h) for u,h in dataset.histories.items()})
groups=pd.cut(activity,bins=[0,10,50,np.inf],labels=["low","medium","high"])
display(groups.value_counts(sort=False).to_frame("training-defined users"))
synthetic_follows={"u1":{"authorA","authorB"}}
synthetic_items={"authorA":["i1","i2"],"authorB":["i3"]}
social={i for author in synthetic_follows["u1"] for i in synthetic_items[author]}
assert social=={"i1","i2","i3"}
print("Synthetic U2A2I:",sorted(social))
print("MMR/diversity experiments: chapter 08. Exploration 2% is a hypothesis, not a universal default.")
""",
    "为什么不能用未来一周的活跃度给今天的用户分组？",
)


def save(path, cells, lectures=()):
    if path.exists():
        raise FileExistsError(f"Preserve existing user notebook: {path}")
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Search Recommendation Ads Lab (Python 3.12)",
                "language": "python",
                "name": "reclab",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "course_lectures": list(lectures),
        },
    )
    for i, cell in enumerate(cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:cell{i}", "exec")
    nbf.validate(nb)
    nbf.write(nb, path)


def refine_companions(curriculum):
    overrides = {
        "02_Retrieval_01.pdf": (
            "data",
            "01_Data_Features_ItemCF.ipynb",
            "真实数据与 ItemCF 基线",
        ),
        "02_Retrieval_04.pdf": (
            "tower",
            "02_TwoTower_HandsOn.ipynb",
            "真实 ID / Embedding / 梯度",
        ),
        "02_Retrieval_06.pdf": (
            "tower",
            "02_TwoTower_HandsOn.ipynb",
            "真实数据与共享双塔训练器",
        ),
        "03_Rank_05.pdf": (
            "data",
            "01_Data_Features_ItemCF.ipynb",
            "真实特征快照，不含全部排序特征",
        ),
        "05_LastN_01.pdf": ("tower", "02_TwoTower_HandsOn.ipynb", "真实 LastN 与梯度"),
        "08_Improvement_04.pdf": (
            "diversity",
            "08_Diversity_DPP.ipynb",
            "真实向量多样性；探索收益未验证",
        ),
        "02_Retrieval_11.pdf": (
            "engineering",
            "10_Engineering_Improvement.ipynb",
            "真实多路候选 + 合成社交；缺地理字段",
        ),
    }
    for lesson in curriculum["lessons"]:
        if lesson["file"] in overrides:
            target, filename, mode = overrides[lesson["file"]]
            nb = nbf.read(ROOT / "notebooks" / filename, as_version=4)
            executed = all(
                c.execution_count is not None for c in nb.cells if c.cell_type == "code"
            )
            lesson["notebook"] = {
                "target": target,
                "file": filename,
                "mode": mode,
                "state": "executed" if executed else "pending execution",
            }
    return curriculum


def main():
    folder = ROOT / "notebooks"
    curriculum = json.loads((ROOT / "docs/curriculum.json").read_text())
    mapping = {}
    chapters = []
    for key, stem, title, prefix, count, mode in CHAPTERS:
        lectures = [f"{prefix}_{i:02}.pdf" for i in range(1, count + 1)]
        cells = [
            md(
                f"# {title}\n## Goal\n把概念变成可检查的输入、运算、梯度与对照。\n\n数据类型：**{mode}**。本地教学参数不是生产默认值；机制实验不是原论文完整复现。"
            ),
            md(
                "## Setup\n每本首次从 Setup 起顺序运行，后面的格会复用前面的变量。之后只改一个参数，重跑对应格及下游格。不要把构造数据当作真实业务收益。"
            ),
            code(SETUP),
            md("## Steps\n讲义对应：" + ", ".join(lectures)),
        ]
        cells += EXPERIMENTS[key]
        cells.append(
            md(
                "## Checks\n所有断言和输出都可回溯到代码。它们验证算子、数据快照或机制，不是线上收益。\n\n## Next Steps\n先记录预测，只改一个变量，再运行对应实验格。需要查看辅助函数时，打开 reclab/learning/。"
            )
        )
        save(folder / (stem + ".ipynb"), cells, lectures)
        chapters.append(
            {"target": key, "file": stem + ".ipynb", "title": title, "mode": mode}
        )
        for lecture in lectures:
            mapping[lecture] = {
                "target": key,
                "file": stem + ".ipynb",
                "mode": mode,
                "state": "pending execution",
            }
    for lesson in curriculum["lessons"]:
        lesson["notebook"] = mapping[lesson["file"]]
    refine_companions(curriculum)
    curriculum["notebook_chapters"] = chapters
    curriculum["learning_path"] = (
        "Notebook 优先：00课程导航 → 01真实数据 → 02双塔基础 → 03~10专题；讲义与论文用于回查。"
    )
    (ROOT / "docs/curriculum.json").write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2)
    )
    entries = [
        ("01_Data_Features_ItemCF.ipynb", "01 · 真实数据与 ItemCF", "真实数据"),
        ("02_TwoTower_HandsOn.ipynb", "02 · 双塔、张量与梯度", "真实数据与共享训练器"),
    ] + [(c["file"], c["title"], c["mode"]) for c in chapters]
    table = "\n".join(
        f"| [{title}](./{file}) | {mode} |" for file, title, mode in entries
    )
    cells = [
        md(
            "# 推荐系统 · Notebook 课程导航\n\n**以代码为主线。** 点击章节，先预测，再逐格执行、改一个变量、检查结果。\n\n| Notebook | 数据与范围 |\n|---|---|\n"
            + table
        ),
        md(
            "## 如何学习\n1. 优先01、02，恢复数据、张量、autograd和训练直觉。\n2. 03~10配套全部45份讲义；章节覆盖不等于45篇论文完整复现。\n3. 真实数据、构造张量、模拟流量在每节明确区分。\n4. Shift+Enter运行。02最后会保存新模型；其他专题默认只做内存实验。\n5. 代码在reclab/和reclab/learning/，PPT用于回查。"
        ),
        code(
            "from pathlib import Path\nimport sys\nprint('Python:',sys.executable)\nprint('Current directory:',Path.cwd())"
        ),
        md(
            "## 你的实验记录\n每次写三行：**我预测什么 → 我只改了什么 → 结果支持什么、仍不能说明什么**。\n\n每本首次从头顺序运行；之后可以只改一个参数，重跑对应格及下游格。"
        ),
    ]
    save(folder / "00_Course_Map.ipynb", cells)
    print("Created course map + 8 companions; mapped", len(mapping), "lectures")


if __name__ == "__main__":
    main()
