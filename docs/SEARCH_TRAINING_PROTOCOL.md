# 搜索教学训练协议

42章只复用已下载且verify_sources逐项校验的SciFact与TinyBERT。来源、固定revision、SHA256、大小、许可见search_sources.json；SciFact CC-BY-SA-4.0，TinyBERT Apache-2.0。无新数据下载、无远程代码、无权重再分发；模型与逐查询预测保留本机忽略目录。新增人工中文词例与会话由Lab构造，不是第三方用户日志。

预先固定原train按query ID字典排序前40条：前32 fit、后8 select，和原300 test不相交。每条fit最多一个judged positive加3个BM25未判断候选（弱负例，不是真负判断）。select直接BM25 top10，绝不插正例。候选上限与漏召单独记录。此前已看过300测试，不用它选择超参或称新盲测。

M1Pro16GB预算：CPU四线程、128 pair以内、batch8、max_length96、两epoch最多32 optimizer steps、AdamW lr2e-5、seed42。模型约4.4M参数；本实验不做超参搜索，不启动其他服务或用户内核。训练所有BERT参数，不只是线性头；记录attention query权重L1变化以证明神经层更新。loss为BCEWithLogits，logit不称校准概率。

每次Run All调用search_training_workshop.run()在同一search registry建立config.workshop=tinybert_finetune的新run，保留冻结与微调的同一select候选结果、损失、耗时、checkpoint、预测。正式latest排除workshop，原模型与服务不变；失败run标failed保留。结果只说明这个小训练机制跑通，不构成论文配方复现、中文语义训练或线上收益证据。
