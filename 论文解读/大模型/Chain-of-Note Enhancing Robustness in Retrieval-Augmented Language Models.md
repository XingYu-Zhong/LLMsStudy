# Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models

论文链接：https://arxiv.org/abs/2311.09210


## 1.论文背景

检索增强语言模型（RALM）：通过将大型预训练语言模型与外部知识检索相结合，RALM 可以减少事实错误和幻觉，同时注入最新知识或领域知识。

常规的 RALM 方法存在三个弊端：1）检索系统并不能保证一直能检索出最相关或最值得信赖的信息。不相关的信息可能会对模型带来错误的指导，即使模型内部已经包含了回答问题的信息，也可能会被忽视；2）幻觉；3）缺乏透明度


## 2.论文解决了什么问题

基于存在的问题，作者对 RALM 系统的鲁棒性做了两个定义：

1）“噪声”鲁棒性：RALM 辨别和忽略不相关检索文档中存在的噪声信息，同时适当利用其内在知识的能力。
2）“未知”鲁棒性：当问题本身能力无法回答，同时检索的文档也没有的时候，RALM 应该回答“unknown”来承认其局限性。

为了改进以上鲁棒性，作者提出了：Chain-Of-Note (CoN) 框架，它为每个文档生成简洁且上下文相关的摘要或注释。该方法允许模型系统地评估从外部文档中获取的信息的相关性和准确性。


## 3.论文方法

1.Chain-Of-Note

通过对检索到的每个文档进行总结和评估，让模型生成 reading note，然后再生成最终的回应。这个过程可以增强模型的以下能力：

1）评估检索到文档与查询的相关性；2）识别可靠信息与误导信息；3）过滤掉无关或不可信的内容；4）认识到知识差距并回应“unknown”

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RALM%20vs%20RALM%2BCoN.png)

2.三种不同类型的 CoN

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/3%E7%A7%8Dnote.png)

相关：语言模型根据检索到的信息生成最终答案

无关但有用的上下文：检索到的文档虽然没有直接回答 query，但提供了上下文，使得语言模型能够将这些信息与其固有知识结合起来，从而推导出答案

无关：语言模型遇到不相关文档并缺乏回应所需知识的情况，能够承认自己“unknown”

3.CoN 框架的实现

reading note 设计：有直接答案就检索回答，有线索就推理，不知道就说不知道

数据收集：通过 ChatGPT 来合成数据，从 NQ 数据集中随机采样了10K个问题，提示词如下：

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoN%20prompt.png)

模型训练：使用 ChatGPT 为不同类型的笔记生成的训练数据，对LLaMa-2 7B模型进行微调，以增强模型记笔记的能力，使用加权损失函数策略，将训练重点放在最终答案的准确性上。

## 4.实验分析

作者在 NQ 和另外三个开放域问答数据集上进行，即 TriviaQA，WebQ 和 RealTimeQA。

对集成了 CoN 的 RALM 进行评估，与标准 RALM 进行比较，重点关注三个主要方面：

(1)使用DPR检索文档的整体问答表现，

(2)通过向系统引入噪声信息来评估抗噪声能力

(3)通过LLaMa-2预训练数据之外的查询(即实时问题)来评估未知鲁棒性。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/QA%E6%80%A7%E8%83%BD.png)
![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E5%99%AA%E5%A3%B0%E9%B2%81%E6%A3%92%E6%80%A7.png)
![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E6%9C%AA%E7%9F%A5%E9%B2%81%E6%A3%92%E6%80%A7.png)

论文的实验表明，CoN 不仅在使用 DPR 检索文档时改善了整体问答表现，还增强了抗噪声和未知两方面的鲁棒性。这包括在噪声检索文档中的精确匹配分数提高 7.9，以及对超出预训练知识范围的实时问题的拒绝率 RR 提高 10.5。


## 5.其他

self-rag：自适应检索，按需检索，RAG+微调，也是通过 ChatGPT 构造训练数据集，利用指令微调将能力蒸馏到了 LLaMa2 上，让模型具备特定的能力。

## 6.总结

本文不仅有 RAG，还有模型微调，是两者的结合，因为需要训练，性能提升其实是理所当然的。主要介绍了 Chain-of-Note 提示用于上下文自适应增强的思路，而基于 GPT4 进行数据蒸馏，可以生成微调数据，然后转为一种特定的能力，但其泛化性并不是很够。当模型本身和召回文档都不掌握回答问题需要的知识时，应该回答 unknown 而不是胡编乱造这种思路也是比较重要的。
