# **Agent Instructs Large Language Models to be General Zero-Shot Reasoners** 

论文链接：https://arxiv.org/abs/2310.03710

GitHub：https://github.com/wang-research-lab/agentinstruct

## 1.论文背景

目前，LLM 的新兴能力，例如复杂推理的能力，使其近年来成为研究的主题。其中，零样本推理引起了广泛的公众兴趣，并在特定任务领域取得了可喜的成果。然而，LLM在一般任务上的推理能力仍有待考究。

##  2.论文提出观点

论文提出一种利用零样本Agent生成指令来指导推理的方法。首先使用一个Agent根据简单的任务信息和几个输入样例生成完成任务的说明（一系列instructions），再将instructions给到LLM进行后续的推理任务并输出结果。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/Zero-shot%20Agent%20instructions.png)

## 3.论文方法

**Agent Instructions**：基于ReAct方式，利用Agent生成目标任务的instructions，对应动作空间有：ask_about_dataset[string]：搜索外部信息，代码中调用API使用微软Bing在网络上搜索对应任务与数据集的相关信息。

finish[instructions]：基于前面的观察与思考得到最终的instructions。

![](https://github.com/zzysos/LLMsStudy/blob/master/论文解读/pic/Agent Instructions生成过程.png)

**Chain of Thought Reasoning**：利用上面Agent生成的多步骤的instructions进行一步步的推理最后得到结果。

## 4.实验分析

实验用到HELM（Holistic Evaluation of Language Models）中的多个任务数据集，作者将这些任务分为生成，分类，推理三类，并与Zero-shot，Zero-shot CoT进行对比。

![]()     ![]()

在不同模型上进行实验对比。

![]()

消融实验：

![]()

与few-shot方法和Self-Consistency方法进行比较：

![]()     ![]()

## 5.论文总结

论文提出了一种提高大型语言模型在一般语言理解任务上零样本推理能力的新方法，构建了一个Agent自动为广泛的任务生成特定于任务的指令。这些指令用于指导LLM在这些任务中更好地进行推理，以做出高质量的预测。

## 6.可能改进的点

将Reflexion模块加入本论文的方法中，可以根据任务执行时出错的样例总结出经验，从而及时改善旧的instructions以其更适用于当前任务数据。