# M1 Pro 16GB 实际训练结果

2026-10-30在新机运行。这里只保留聚合指标、配置、训练曲线和校验来源；不包含原始数据、逐用户预测、训练样本或模型权重。完整训练产物保留在原项目被Git忽略的artifacts目录。

Retailrocket包含热门、ItemCF和四组8轮双塔；XMRec为冠军核心方法教学复现，8179位官方验证用户，图4线程、LightGBM1线程，模型按select选型，audit独立报告。hidden_test_score为null。不同设备重训可能存在浮点差异。

新机验收：56项pytest通过；11~15五本Notebook的30格顺序执行通过；45份PDF、16个Notebook入口、32维特征及10条推荐API通过。01~10保留既有输出，本次未重新执行。
