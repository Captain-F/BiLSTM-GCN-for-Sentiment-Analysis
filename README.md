# BiLSTM-GCN-for-Sentiment-Analysis
## 利用依存句法树和图卷积神经网络进行文本情感分析

### 依赖库：
* keras == 2.4.3
* tensorflow == 2.3.1
* sklearn == 0.23.2
* spektral == 1.0.5
* ltp == 4.1.1

### 流程：
* 文本表示
  * 利用的词向量为[北师大中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)
* 依存句法图构建 
  * 利用哈工大的[LTP](http://ltp.ai/)生成依存句法图
* BiLSTM-GCN构建
* 训练-预测
* classification_report输出

### 说明：
* 使用的是公开数据集，使用其中的5001条数据作为示例
* 情感类别为正和负
* 数据集的80%作为训练集，剩余的20%作为测试集

### 结果：
* 训练30轮后的结果如下： 


|*Precision (%)*|*Recall (%)*|*F1 (%)*|
|:---:|:---:|:---:|
|91.337|91.290|91.303|

### 使用：
* `python train.py`
