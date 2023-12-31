# Reasoning with Language Model is Planning with World Model

论文链接：https://arxiv.org/abs/2305.14992

## 1.论文背景

LLM目前仍没办法像人类大脑一样进行深思熟虑的长远规划与推理，包括探索不同的推理方向，预估可能得到的中间状态与反馈等，这是目前LLM推理的局限之一。

## 2.论文提出解决方案

文章使用两种LLM，一个用作Agent，一个用作World Model(所谓World Model，以下简称WM，即模拟人脑对外部世界的认知与建模，人脑在遇到新问题时，会先在脑中根据现有经验思考可能的解决办法，可能的结果与这样做的价值)，随后基于此提出一种RAP推理框架，让LLM能更像人一样进行有意的规划。如下图。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAP%E6%80%9D%E7%BB%B4%E6%A1%86%E6%9E%B6.png)

## 3.论文方法

文章引入了基于蒙特卡洛树的搜索算法(MCTS),让LLM能够战略性探索问题空间，在探索与利用间取得平衡，得到较好的推理轨迹，如下图：

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E8%A7%84%E5%88%92.png)

构建树的过程包含多次迭代，每次迭代主要步骤有如下四个：

1.选择：在已存在的树结构中选择一个当前树的叶子节点进行后续扩张，从初始节点开始基于UCT(一种平衡探索和利用的方式)的方式往下寻找，一直找到一个叶子节点。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/UCT%E5%85%AC%E5%BC%8F.png)

2.扩展：通过Agent生成多个action，再通过WM预测经过每个action后得到的不同状态。

3.模拟：选取一个节点继续按上述过程往下扩展，模拟从该节点继续向下推理的过程，直到得到终止节点，每次都选择局部回报最大的action。

4.反向传播：到达终止节点后将回报累次相加回传给上述推理路径上的各父节点，更新Q值。

## 4.实验分析

实验部分选取了计划生成(BlocksWorld Game)，数学推理，逻辑推理三个不同的任务来测试RAP框架的性能，并将其与CoT，CoT-SC，Least-to-Most等现有方法进行比较，其效果都优于之前的方法。

## 5.论文总结

文章提出一种新的RAP推理框架，使LLM能够像人一样，根据已有经验先进行思考与规划，最终再一步步推理出问题的结果，有一定创新性。

## 6.可能可以改进的点

作者认为本文中使用到的WM只是经过预训练，是一个通用的LLM，而如果能针对特定的任务对WM进行微调，让WM更好的模拟特定环境，可能可以使生成的状态，回报等值更加准确。
