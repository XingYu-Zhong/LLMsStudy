# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

论文链接：https://arxiv.org/abs/2305.10601

## 1.论文背景

大语言模型(LLM)现在被用于解决各种不同类型的问题，但受限于其推理过程中token级别的从左到右的决策过程，大模型仍没办法很好的完成某些需要进行局部探索(多角度思想)，战略前瞻的任务。

## 2.论文提出解决方案

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/ToT%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%85%B6%E4%BB%96%E4%B8%89%E7%A7%8D%E6%96%B9%E6%B3%95.png)

如上图，左边是三种已有的使用LLM进行思维推理的方法，分别是IO，CoT，CoT-SC，但他们都属于是单链思维，没有针对某个问题或中间思想进行多角度的思考。右边的是本文提出的ToT思维框架，通过维护一个树的结构，利用LLM对输入问题生成中间思想，再利用LLM对生成思想进行评估，结合特定的搜索算法对问题空间进行搜索并适当剪枝，以实现对问题的推理前瞻，多角度思考，对不同思维步骤进行局部探索，适当时候进行思想回溯等。

## 3.论文方法

针对上面提出的基于树思维框架，其中一个重点在于每一步如何生成中间思想，以及如何评估生成的思想的价值(即是否对任务的完成有帮助)。

Thought Generator：用于生成下一步的中间思想，分为基于CoT提示生成与基于Propose提示生成。前者在提示词中只给出一个大致的思想方向，适用于一些问题空间比较丰富的情况(如创意写作)，后者在提示词中给出某些特定的限制，规定输出的某些条件如格式等，适用于一些问题空间比较有限的情况(如Game of 24)。

Value Evaluator：用于评估中间状态的价值，分为独立评价与投票评价。前者将本次生成的所有中间思想分别进行评分，可能最后会选出分最高的几个，后者将所有中间思想聚集在一起做一个投票，选出票数最高的一个。评价的作用多用于树的剪枝，缩小问题空间。

有了上面的树的思维框架，还需要将搜索算法应用到这个框架上面来解决实际问题，本文介绍了两个基本搜索算法BFS，DFS跟ToT的结合，如下图。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/BFS%E4%B8%8EDFS%E7%AE%97%E6%B3%95%E7%BB%93%E5%90%88ToT.png)

## 4.实验分析

本文将ToT应用于三个不同的任务中：Game of 24，Creative Writing，Mini Crosswords。

1.Game of 24：使用BFS算法，其中Thought Generator使用基于Propose提示生成，Value Evaluator使用独立评价方法。

2.Creative Writing：使用BFS算法，其中Thought Generator使用基于CoT提示生成，Value Evaluator使用投票评价方法。

3.Mini Crosswords：使用DFS算法，其中Thought Generator使用基于Propose提示生成，Value Evaluator使用独立评价方法。

文章还将IO，CoT，CoT-SC的方法也在对应任务中进行实验以与ToT框架作对比。结果表明，ToT框架的效果在三个任务中均优于其他方法。

## 5.总结

本文提出的ToT框架，让LLM能够像人类思考问题一样，提出多种可能的解题思路进行局部的探索，在深入探索到一定阶段，发现该思路可能不太可行时，又可通过思想回溯换一个思路继续深入思考最后得到最好的解答，其推理能力相比其他方法来说确实有了不少提升。

## 6.可能可以改进的点

1.无论是思想生成还是评估都需要使用LLM来完成，一次任务中就可能调用很多次GPT的API，有点耗时和耗资源。

2.思想生成和评估都使用的是LLM内部的知识，或许评估时可以结合外部环境给的反馈来进行评估？
