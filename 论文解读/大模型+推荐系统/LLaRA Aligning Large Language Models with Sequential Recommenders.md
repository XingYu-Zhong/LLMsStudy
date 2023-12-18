# LLaRA: Aligning Large Language Models with Sequential Recommenders

论文链接：https://arxiv.org/pdf/2312.02445.pdf

论文代码：https://github.com/ljy0ustc/LLaRA

## 1.论文背景

顺序推荐是指根据用户过往历史的记录进行下一个预测推荐，作者认为llm在这项任务上可以达到不错的水平。

作者认为之前都是直接使用id索引来表示文本提示，再把提示输入llm，这样子并不能表现出足够的顺序理解。

作者说传统的顺序推荐器通常涉及两个步骤:

(1)分配具有不同ID的每个项目，将其转换为可训练的嵌入;

(2)学习这些嵌入对交互序列进行建模，从而捕获用户偏好并预测下一个感兴趣的项目。



## 2.论文提出的问题

作者认为，仅仅用基于 ID 或基于文本的项目序列的表示提示 LLM 不能完全利用 LLM 进行顺序推荐的潜力。相反，llm应该更深入地了解顺序交互中固有的行为模式。

![image-20231218173536535](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173536535.png)

## 3.论文的方法

作者提出llara方法主要是在提示词上做了优化，分别是混合提示词设计，这个主要是结合了除了文本信息以外的信息，还有课程提示词训练，这个是作者从另外两篇论文中获取的灵感。课程学习不仅熟悉推荐机制的LLM，而且内化了推荐者编码的行为知识。

![image-20231218173658719](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173658719.png)

![image-20231218173612163](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173612163.png)

课程学习的灵感来自于人类教育中使用的教学策略，强调从简单到更具挑战性的学习任务训练模型。1.复杂性评估：课程学习最初量化了每个数据点或任务的复杂性，然后用于分配学习优先级。2.调度器公式：基于复杂性评估，可以设计一个训练调度器来决定模型在学习过程中将暴露的任务的序列和频率。3.训练执行：在设计训练调度器后，我们可以实现遵循预定进展的课程学习过程。

总上所述，就是评估-设计计划-执行这三步。

![image-20231218173623090](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173623090.png)

## 4.论文实验



作者使用了两个数据集分别是电影和steam数据集，对比方法主要是和传统方法GRU4Rec，Caser，SASREC还有大语言模型方法GPT-4，TALLRec，MoRec，实验比较的指标是HitRatio@1

![image-20231218173758961](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173758961.png)

RQ1：与传统的顺序推荐模型和基于LLM的方法相比，LLARA的表现如何？

![image-20231218174011940](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218174011940.png)

RQ2：我们的混合提示方法与提示设计中其他形式的项目表示相比如何？

![image-20231218174029581](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218174029581.png)

RQ3：我们的课程学习方案如何针对其他模式注入方法进行测量？

![image-20231218173842613](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231218173842613.png)

## 总结

这篇文章是十分值得推荐的，作者在写作上和问题描述上面都有理有据，整体来看是一篇是否值得学习的文章，虽然在创新点上没有那么出彩，但是在写作方面特别值得我们学习。