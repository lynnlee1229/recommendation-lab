# 公用基础课程

独立路径：公用 → 学习路径 → 31–34。旧00–30不修改，不把推荐入门课挪为公用。

| 章 | 机制与单变量实验 | 解释练习 |
|---|---|---|
| 31 张量 | 矩阵乘法、求和轴、特征单位变化、广播陷阱 | 能运行的广播为何仍可能错；单位与权重如何配合 |
| 32 梯度 | 手算链式法则、中心差分步长、梯度累加、学习率 | 局部斜率、清梯度与过大步长 |
| 33 优化与泛化 | 二次损失学习率、固定数据下1阶/9阶容量 | 训练下降与选择集误差、选择与审计分离 |
| 34 评分与评估 | 温度与概率评分、查询偏置与AUC、候选集合与NDCG | 排序/概率/线上收益的证据边界 |

每章5个短代码格，最后一格可记录预测、结果与解释。全部使用构造数组，CPU执行，不下载数据、不训练或覆盖正式领域模型、不写正式实验registry。有限差分展示浮点消减；高阶模型保留泛化负结果；温度不变排序但改变logloss。单次构造结果不推广为业务规律。

复用torch自动微分与reclab.probability.evaluate；新增reclab/learning/foundations.py只包含可查看的二次下降与训练集最小二乘拟合。原learning中的FM、注意力、协同过滤等保留在领域教学中，没有复制领域实现。

执行：

```bash
.venv/bin/python scripts/execute_notebooks.py --only 31_Common_Tensors 32_Common_Gradients 33_Common_Optimization 34_Common_Scores_Evaluation
.venv/bin/python -m pytest -q
node --test tests/navigation.test.cjs
```

四本均从新内核顺序执行、保存输出并导出HTML；20代码格、5图，数学/形状关系断言全部通过。83项pytest通过，四项导航断言覆盖四模块、旧链接、页面路由、35章唯一归属。桌面与390px窄屏公用路径可达，窄屏文档宽度390无横向溢出。仅正常重启18580加载目录，18581和用户内核保持。
