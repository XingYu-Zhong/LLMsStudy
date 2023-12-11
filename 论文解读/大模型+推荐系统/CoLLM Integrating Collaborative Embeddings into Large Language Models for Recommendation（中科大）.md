# CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation（中科大）

论文链接：https://arxiv.org/abs/2310.19488

论文代码：https://github.com/zyang1580/CoLLM

## 1.论文背景

作者认为直接通过提示词进行llm推荐，在冷启动方面确实有比较大的优势，但是在非冷启动上作者发现这样的方式是不如传统推荐系统的。

![](https://raw.githubusercontent.com/XingYu-Zhong/LLMsStudy/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/collm_warm_cold.png)

## 2.论文提出的问题

作者认为直接通过文本信息给大模型进行相似推荐，这样的建模方式并不能很好的表达推荐之间的相似性，所以作者提出了一种方法来解决这个问题

## 3.论文解决方法

collm就是作者先用lora去微调了大模型让这个微调的大模型来推荐任务，在推荐前还是使用传统推荐系统CIE方法把数据进行建模处理向量化，然后在用这个建模好的数据给llm做推荐

![](https://raw.githubusercontent.com/XingYu-Zhong/LLMsStudy/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/collm_key.png)

## 4.论文实验

实验部分，作者围绕两个问题去做实验，第一个是与现有方法相比，提出的 CoLLM 能否通过协作信息有效地增强 LLM 以改进推荐。第二个是设计选择对所提出方法的性能的影响有多大？作者使用AUC和UAUC作为验证指标，与传统推荐系统和一些基于提示词的llm推荐系统在ML电影数据集和亚马逊书本数据集上进行实验测试，实验结果上来看都有不错的提升，我认为功劳在于微调上

![](https://raw.githubusercontent.com/XingYu-Zhong/LLMsStudy/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/collm_data.png)



## 5.论文总结

这篇文章的写作手法是值得学习的，他把一个很简单的idea写的让人觉得很复杂，这是一个很值得学习的能力，我认为他本质上就是用来传统推荐系统的数据建模方法在加上了lora微调做llm推荐系统，他在描述的时候讲了一些lora微调的函数之类的，但又特别浅，把读者讲的一愣一愣的，其次就是实验的目的性特别好，他是先提出问题，然后用实验去证明。总体来说，这个文章值得我们写作的时候学习

