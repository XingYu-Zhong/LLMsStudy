# Igniting Language Intelligence The Hitchhiker's Guide From Chain-of-Thought Reasoning to Language Agents（从 CoT 到 Agent 综述）

论文链接：https://arxiv.org/abs/2311.11797

github：https://github.com/Zoeyyao27/CoT-Igniting-Agent#33-cot-for-agent

## 1.论文背景

作者认为思维链 CoT 推理技术在大模型性能表现中有显著提升，而且在增强可解释性、可控性和灵活性方面也表现出熟练特性。鉴于这些优点，最近的研究努力将 CoT 推理方法扩展到培养自主语言代理的发展，Agent 是一类拥有“自主智能的实体”，而以 Agent 为主体的大模型必须具备感知，记忆和推理的能力，恰巧 CoT 可以从这三个方面赋予 Agent。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/p_m_r_CoT.png)


## 2.论文解决了什么问题

作者从三个研究维度逐步深入来论述从 CoT 到 Agent 的发展:1)CoT 技术的基础机制，阐述其功效的情况和理由;2)CoT 的范式转变;3)由 CoT 方法强化的语言代理的兴起。


## 3.论文提出了什么观点

作者抽象出大模型智能体的结构框架，主要由三部分组成，分别是 Agent 主体，工具与环境。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoT_Agent_fw.png)

作为 Agent 主体的大模型是模拟人类智能决策流程的核心，在许多 Agent 需要处理的任务中，Agent 的“先天知识”并不包含解决任务的直接答案，因此 Agent 需要在一系列与外部环境的交互循环中，制定计划，做出决策，执行行动，收到反馈……在一整个计划、决策与控制的循环中，大模型需要具备“感知”，“记忆”与“推理”的能力，CoT 恰恰可以从这三个方面来赋能 Agent。

规划、决策和行动执行的过程可以反映 LLM 的推理能力，由于 LLM 暴露在 LLM 预训练期间不存在的环境中，在这种环境中，LLM 必须感知世界的知识并采取行动，CoT 有助于弥合环境感知与 LLM 天生能力之间的差距。


## 4.论文思路

1.什么是 CoT？

大模型逐步参与将一个复杂问题分解为一步一步的子问题并依次进行求解的过程。这一系列推理的中间步骤就被称为思维链（Chain of Thought）。

2.为什么使用 CoT？

CoT 增强了大模型的推理能力，可解释性，可控性，灵活性，

3.何时应该使用 CoT？

作者认为 CoT 应当被用于 20B 以上参数规模的模型之中，并且模型的训练数据应当与任务问题相关且彼此有较强的联结。

4.为什么 CoT 会生效？

关于 CoT 为什么会生效，目前还没有一套被大家广泛接受的普遍理论。通过许多论文对 CoT 与大模型互动的实验，大致总结如下：CoT 需要大模型具备一些方面“最基础”的知识；使用 CoT 可以为一些它理解到的基础知识之间搭起一座桥梁；CoT 的作用在于强迫模型进行推理，而不是教会模型如何完成推理。

5.CoT 朝着什么方向发展？

在CoT问世的一年多以来，CoT 也开始从最简单的“Let's think step by step”慢慢进化，这篇综述也全面概括了 CoT 的发展方向与进化路径，如下图，包括“Prompt 模式”，“推理结构”以及“应用场景”。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoT%20approaches.png)

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoT%20formulations.png)

6.CoT 与 Agent 有何关系？

基于最近许多关于 Agent 框架的研究，作者觉得大模型智能体可以被认为具有如下图的结构：

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoT_Agent_fw.png)
![]()

当人类指令输入 Agent 主体后，Agent 主体通过一系列计划、决策与控制，使用工具与外部环境互动。感知 CoT，记忆 CoT，推理 CoT 的实现有助于智能体的构建。


## 5.论文局限性

作者认为 CoT 和 Agent 在这些方面存在挑战：未知领域中的泛化能力，Agent 的过度交互问题，多智能体社会，Agent 安全问题，Agent 的评价。


## 6.其它

RL智体：RL智体被训练通过与环境的迭代交互来做出决策，接收奖励或惩罚形式的反馈——正确的动作会得到奖励，而错误的动作会受到惩罚。但它严重依赖专家数据，并为特定任务设计奖励函数，缺乏泛化能力，透明度和可解释性。

语言智体：语言智体利用 LLM 中嵌入的常识先验与 RL智体区分开，使它能够适应环境，并利用 CoT 进行解释。

可以通过结合类似RL的策略来增强语言智体的能力。


## 7.总结

作者对 CoT 推理进行全面研究，包括该领域的最新进展和挑战，并延申到大模型 Agent 的前沿议题，CoT 不仅作为推理的技术手段，还可以扩展促进 Agent 的开发。文章中的一些观点在当下也存在质疑，比如：最近有文章质疑大模型是否可以真的进行可靠的 CoT 验证，在大模型的能力本身无法解决验证结果反馈提出的问题时，大模型有可能会过度纠正推理过程，直接跳过正确答案。

论文中的很多点可以进行深入研究，比如：CoT 构造的几套框架