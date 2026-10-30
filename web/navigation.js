/* Product hierarchy. Domains own their lessons, tools, records and source views. */
const LAB_MODULES = {
  common: {
    title: '公用', code: 'COMMON', description: '先建立共用的实验方法与证据口径，再进入一个领域。',
    pages: [
      ['start', '起步', 'home'], ['path', '学习路径', 'code'], ['method', '实验方法', 'method'],
      ['records', '全局实验台账', 'research'], ['source', '公共源码', 'source'],
    ],
    sources: ['foundations', 'experiments', 'probability', 'service'],
  },
  search: {
    title: '搜索', code: 'SEARCH', description: '从查询与相关性判断开始，学习词法、语义、索引、融合与重排。',
    pages: [
      ['path', '学习路径', 'code'], ['retrieval', '检索实验', 'search'],
      ['records', '搜索实验记录', 'research'], ['source', '搜索源码', 'source'],
    ],
    sources: ['search', 'search_lessons'],
  },
  ads: {
    title: '广告', code: 'ADS', description: '从真实曝光点击学习概率预测，再进入独立模拟的出价与预算决策。',
    pages: [
      ['path', '学习路径', 'code'], ['prediction', '点击预测', 'ads'],
      ['auction', '拍卖与预算', 'auction'], ['records', '广告实验记录', 'research'],
      ['source', '广告源码', 'source'],
    ],
    sources: ['ads', 'ads_teaching', 'auction'],
  },
  recommendation: {
    title: '推荐', code: 'REC', description: '从用户行为与候选物品开始，学习召回、排序、训练和评估。',
    pages: [
      ['path', '学习路径', 'code'], ['data', '数据与基线', 'overview'],
      ['retrieval', '召回与重排', 'trace'], ['features', '特征与样本', 'features'],
      ['two-tower', '双塔训练', 'experiments'], ['records', '推荐实验记录', 'research'],
      ['references', '讲义与论文', 'course'], ['source', '推荐源码', 'source'],
    ],
    sources: ['data', 'models', 'retrieval', 'train', 'deep_recommendation',
      'xmrec_data', 'xmrec_memory', 'xmrec_graph', 'xmrec_ranking', 'xmrec_experiment', 'xmrec_metrics'],
  },
};
const COURSE_PATHS = {
  common: [
    {title: '基础 · 从张量到评估', chapters: [31, 32, 33, 34], description: '张量与线性代数 → 计算图与梯度 → 优化与泛化 → 评分、概率与排序评估。小数组构造机制实验，再进入领域路径。'},
  ],
  recommendation: [
    {title: '入门 · 数据、基线与训练', chapters: [1, 2, 3], description: '以真实推荐日志为例认识数据、双塔、梯度与评估；这是推荐入门，并非独立的公用基础课。'},
    {title: '机制 · 逐个理解算法', chapters: [4, 5, 6, 7, 8, 9, 10], description: '召回 → 多目标 → 特征交叉 → 序列 → 多样性 → 冷启动 → 工程。构造张量和模拟流量均按章节标识。'},
    {title: '实践 · XMRec 竞赛核心', chapters: [11, 12, 13, 14, 15], description: '题目与数据 → 特征 → 图模型 → 排序 → 评估；非完整冠军集成。'},
    {title: '综合 · 推荐对照实验', chapters: [16], description: '把负采样、序列和索引放回同预算对照，保留负结果。'},
    {title: '参考 · 旧版推荐课程导览', chapters: [0], description: '原00章只覆盖推荐课程，保留原内容；三领域导航以当前网页为准。'},
  ],
  search: [
    {title: '综合 · 先看完整搜索实验', chapters: [17], description: '在真实 SciFact 上比较词法、向量、融合与重排，建立全景。'},
    {title: '深入 · 按搜索链路拆解', chapters: [25, 26, 27, 28, 29, 30], description: '任务与qrels → BM25 → 语义编码 → 近似索引 → 融合与训练 → 误差与服务。'},
  ],
  ads: [
    {title: '综合 · 预测与决策的边界', chapters: [18], description: '先看真实点击建模与模拟拍卖如何分层；两者不是同一条真实竞价日志。'},
    {title: '深入 · 点击概率', chapters: [19, 20, 21, 22], description: '数据切分 → 线性特征 → 非线性容量 → 概率校准与审计。'},
    {title: '深入 · 模拟市场决策', chapters: [23, 24], description: '拍卖与支付 → 出价 → 预算pacing → 预测偏差；所有收入均为模拟。'},
  ],
};
const LEGACY_ROUTES = {
  home: 'common/start', research: 'common/records', code: 'recommendation/path',
  course: 'recommendation/references', overview: 'recommendation/data',
  trace: 'recommendation/retrieval', features: 'recommendation/features',
  experiments: 'recommendation/two-tower', search: 'search/retrieval', ads: 'ads/prediction',
};
function resolveRoute(hash) {
  const raw = hash.replace(/^#/, '');
  const [module, requested] = (LEGACY_ROUTES[raw] || raw).split('/');
  const config = LAB_MODULES[module];
  if (!config) return {module: 'common', page: 'start'};
  return {module, page: config.pages.some(p => p[0] === requested) ? requested : config.pages[0][0]};
}
