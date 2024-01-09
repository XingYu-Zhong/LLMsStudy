# **Agent Instructs Large Language Models to be General Zero-Shot Reasoners** 

论文链接：https://arxiv.org/abs/2310.03710

GitHub：https://github.com/wang-research-lab/agentinstruct

## 1.论文背景

目前，LLM 的新兴能力，例如复杂推理的能力，使其近年来成为研究的主题。其中，零样本推理引起了广泛的公众兴趣，并在特定任务领域取得了可喜的成果。然而，LLM在一般任务上的推理能力仍有待考究。

##  2.论文提出观点

论文提出一种利用零样本Agent生成指令来指导推理的方法。首先使用一个Agent根据简单的任务信息和几个输入样例生成完成任务的说明（一系列instructions），再将instructions给到LLM（task executor）进行后续的推理任务并输出结果。

论文中Agent使用的默认是GPT-4，task executor默认是GPT-3.5。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/Zero-shot%20Agent%20instructions.png)

## 3.论文方法

**Agent Instructions**：基于ReAct方式，利用Agent生成目标任务的instructions，对应动作空间有：ask_about_dataset[string]：搜索外部信息，代码中调用API使用微软Bing在网络上搜索对应任务与数据集的相关信息。

finish[instructions]：基于前面的观察与思考得到最终的instructions。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/Agent%20Instructions%E7%94%9F%E6%88%90%E8%BF%87%E7%A8%8B.png)

**Chain of Thought Reasoning**：利用上面Agent生成的多步骤的instructions进行一步步的推理最后得到结果。

## 4.实验分析

实验用到HELM（Holistic Evaluation of Language Models）中的多个任务数据集，作者将这些任务分为生成，分类，推理三类，并与Zero-shot，Zero-shot CoT进行对比。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/HELM%E4%BB%BB%E5%8A%A1%E5%88%86%E7%B1%BB.png)     ![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/Winning%20rate%20(%25)%20between%20zeroshot%2C%20zero-shot%20CoT%2C%20and%20zero-shot%20AgentInstruct.png)

在不同模型上进行实验对比。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E5%A4%9A%E6%A8%A1%E5%9E%8B%E5%AF%B9%E6%AF%94.png)

消融实验：

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E6%B6%88%E8%9E%8D%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png)

与few-shot方法和Self-Consistency方法进行比较：

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E4%B8%8Efew-shot%E5%AF%B9%E6%AF%94.png)     ![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E4%B8%8ESC%E5%AF%B9%E6%AF%94.png)

## 5.论文总结

论文提出了一种提高大型语言模型在一般语言理解任务上零样本推理能力的新方法，构建了一个Agent自动为广泛的任务生成特定于任务的指令。这些指令用于指导LLM在这些任务中更好地进行推理，以做出高质量的预测。

## 6.可能改进的点

将反思机制加入本论文的方法中，根据任务执行时出错的样例得出概括性的经验总结，存储在一个反思队列中，在下次任务执行时task executor将这些经验加入自己的上下文中进行参考，从而更好的执行任务。

实验：在论文用到的其中一个任务集BoolQ上进行了一些实验。BoolQ的任务是给出一段文章，然后给出一句话，然后需要根据文章内容判断这句话正确还是错误，输出是True/False。

论文方法：（50个样例）成功率：0.74  0.76  0.74    平均：0.746                     （200个样例）成功率：0.825 0.84 0.84   平均：0.835

加入反思机制：（50个样例）成功率：0.76  0.84  0.82     平均：0.806             （200个样例）成功率：0.855 0.85 0.885   平均：0.863