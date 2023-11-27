# Reflexion: Language Agents with Verbal Reinforcement Learning

论文链接: https://arxiv.org/abs/2303.11366

## 1.论文背景

传统大型模型的微调成本非常高，因此它们无法迅速从环境交互中学习并提升性能。由此，本文提出了Reflexion框架，旨在让大型模型能够通过语言反馈优化动作执行。

## 2.论文内容概述

大模型作为goal-driven agents 越来越多地用于和外界环境进行交互，最近涌现了ReAct,HuggingGPT等基于大模型的任务决策框架，它们利用In-context learning的方式快速地指导模型执行任务，避免了传统微调方式带来的计算成本和时间成本。

受前面工作的启发，本文提出了Reflexion框架，使用语言反馈信号(verbal reinforcement)来帮助agent从先前的失败经验中学习。具体地，Reflexion将传统梯度更新中的参数信号转变为添加在大模型上下文中的语言总结，使得agent在下一个episode中能参考上次执行失败的失败经验，从而提高agent的执行效果。这个过程和人类反思(reflexion)过程十分相似。

## 3.论文方法

作者提出Reflexion框架，包含四个组成部分：

![]()

Actor: 基于当前环境生成下一步的动作。
Evaluator: 衡量Actor生成结果的质量。就像强化学习中的Reward函数对Actor的执行结果进行打分。
Self-reflexion：Reflexion框架中最重要的部分。它能结合离散的reward信号(如success/fail)、trajectory等生成具体且详细语言反馈信号，这种反馈信号会储存在Memory中，启发下一次实验的Actor执行动作。
Memory：分为短期记忆(short-term)和长期记忆(long-term)。在一次实验中的上下文称为短期记忆，多次试验中Self-reflexion的结果称为长期记忆。

![]()

执行过程：如上图伪代码所示，Reflexion是一个迭代过程，Actor产生行动，Evaluator对Actor的行动做出评价，Self-Rflexion基于行动和评价形成反思，并将反思结果存储到长期记忆中，直到Actor执行的结果达到目标效果。

## 4.实验分析

1.决策能力

在AlfWorld任务中，Reflexion框架能够有效解决幻觉(hallucination)和规划不足(inefficinet planning)问题，使得agent的任务完成率明显提升，在10次实验后最多完成130/134个任务。

评估指标：决策任务完成率

2.推理能力

HotpotQA是一个基于百科知识库的问答任务，用于测试agent在大量文本中的推理能力。在这个任务中，Reflexion的效果比所有的baseline都高出不少。同时作者还对比了cot+EPM(episodic memory 类似一种长期记忆)和Reflexion框架，发现Reflexion的效果仍要高很多，这说明Reflexion框架中长期记忆和Self-Reflexion模块都起到了重要的作用。

评估指标：推理任务准确率

3.代码生成

在HumanEval(PY)代码生成任务中，Reflexion取得了SOTA效果，准确率相比GPT-4提高10.9%。

评估指标：编程任务通过率

## 5.其它

反思可以在没有明确基准真理的情况下增强AI模型的问题解决能力。通过模拟人类的问题解决策略，这种方法使agent能够通过自我反思、评估和反馈来迭代地改进他们的解决方案。

React：“行动” + “推理”

Self-Refine：对自身的推理进行优化和改进，可以执行多次迭代，直到达到迭代次数或者满足某个条件，输出最终结果

Self-Correction：让LLM模型反思自己的输出结果，并根据假设列表进行true/false判断，重新作答。

Self-Reflexion与self-correction相似，反过来询问LLM模型，让LLM对自己的行为进行评判

## 6.总结

本文提出的Reflexion使得大模型agent能够快速地从错误经验中进行总结学习，在多个任务上都取得了不错的效果。从作者的实验分析看来，Reflexion最重要的两个模块是：

长期记忆模块：赋予了agent长期记忆的能力
Self-Reflexion模块：将一些数字化的reward信号转化为细致分析的语言总结，形成重要的长期记忆，能明显提高下一次执行的成功率。