# LLMsStudy
收集大语言模型的学习路径和各种最佳实践
# LLM的初步探究

随着大模型的热度越来越高，我们有必要对大于语言模型进行深入的研究

## 目录

大语言模型是什么？

大模型推荐

大模型评估基准

没有卡的条件下我们能做什么？

## 大语言模型是什么？

### 大模型的发展脉络

在2017年，Transformer架构的出现导致深度学习模型的参数超越了1亿，从此RNN和CNN被Transformer取代，开启了大模型的时代。谷歌在2018年推出BERT，此模型轻松刷新了11个NLP任务的最佳记录，为NLP设置了一个新的标杆。它不仅开辟了新的研究和训练方向，也使得预训练模型在自然语言处理领域逐渐受到欢迎。此外，这一时期模型参数的数量也首次超过了3亿。到了2020年，OpenAI发布了GPT-3，其参数数量直接跃升至1750亿。2021年开始，Google先后发布了Switch Transformer和GLaM，其参数数量分别首次突破万亿和1.2万亿，后者在小样本学习上甚至超越了GPT-3。

### Transformer结构

Transformer是由Google Brain在2017年提出的一种新颖的网络结构。相对于RNN，它针对其效率问题和长程依赖传递的挑战进行了创新设计，并在多个任务上均展现出优越的性能。

如下图所示的是Transformer的架构细节。其核心技术是自注意力机制（Self-Attention）。简单地说，自注意力机制允许一个句子中的每个词对句子中的所有其他词进行加权，以生成一个新的词向量表示。这个过程可以看作是每个词都经过了一次类似卷积或聚合的操作。这种机制提高了模型对于上下文信息的捕获能力。

### MOE结构

模型的增大和训练样本的增加导致了计算成本的显著增长。而这种计算上的挑战促使了技术的进步与创新。

考虑到这一问题，一个解决方案是将一个大型模型细分为多个小型模型。这意味着对于给定的输入样本，我们不需要让它通过所有的小型模型，而只是选择其中的一部分进行计算。这种方法显著地节省了计算资源。

那么，如何选择哪些小模型来处理一个特定的输入呢？这是通过所谓的“稀疏门”来实现的。这个门决定哪些小模型应该被激活，同时确保其稀疏性以优化计算。

稀疏门控专家混合模型（Sparsely-Gated MoE）是这一技术的名字。它的核心思想是条件计算，意味着神经网络的某些部分是基于每个特定样本进行激活的。这种方式有效地提高了模型的容量和性能，而不会导致计算成本的相对增长。

实际上，稀疏门控 MoE 使得模型容量得到了1000倍以上的增强，但在现代GPU集群上的计算效率损失却非常有限。

总之，如果说Transformer架构是模型参数量的第一次重大突破，达到了亿级，那么MoE稀疏混合专家结构则进一步推动了这一突破，使参数量达到了千亿乃至万亿的规模。



![llm_survey](https://picgo-zxy.oss-cn-guangzhou.aliyuncs.com/typoreimgs/llm_survey.gif)



## 大模型推荐

> 不同任务实验过程中，相对而言整体效果还不错的模型列表。

|          模型          | 最新时间 | 大小        |                           项目地址                           |                           机构单位                           |
| :--------------------: | -------- | ----------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       Baichuan2        | 2023-09  | 7/13B       | [Baichuan2](https://github.com/baichuan-inc/Baichuan2)![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan2.svg?style=social&label=Star) |         [百川智能](https://github.com/baichuan-inc)          |
|        WizardLM        | 2023-08  | 7/13/30/70B | [WizardLM](https://github.com/nlpxucan/WizardLM)![Star](https://img.shields.io/github/stars/nlpxucan/WizardLM.svg?style=social&label=Star) |                             微软                             |
|         Vicuna         | 2023-08  | 7/13/33B    | [FastChat](https://github.com/lm-sys/FastChat)![Star](https://img.shields.io/github/stars/lm-sys/FastChat.svg?style=social&label=Star) | [Large Model Systems Organization](https://github.com/lm-sys) |
|         YuLan          | 2023-08  | 13/65B      | [YuLan-Chat](https://github.com/RUC-GSAI/YuLan-Chat)![Star](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Chat.svg?style=social&label=Star) | [中国人民大学高瓴人工智能学院](https://github.com/RUC-GSAI)  |
|        InternLM        | 2023-09  | 7/20B       | [InternLM](https://github.com/InternLM/InternLM)![Star](https://img.shields.io/github/stars/InternLM/InternLM.svg?style=social&label=Star) |      [上海人工智能实验室](https://github.com/InternLM)       |
|        TigerBot        | 2023-08  | 7/13/70B    | [TigerBot](https://github.com/TigerResearch/TigerBot)![Star](https://img.shields.io/github/stars/TigerResearch/TigerBot.svg?style=social&label=Star) |         [虎博科技](https://github.com/TigerResearch)         |
|        Baichuan        | 2023-08  | 7/13B       | [Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)![Star](https://img.shields.io/github/stars/baichuan-inc/Baichuan-13B.svg?style=social&label=Star) |         [百川智能](https://github.com/baichuan-inc)          |
|        ChatGLM         | 2023-07  | 6B          | [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)![Star](https://img.shields.io/github/stars/THUDM/ChatGLM2-6B.svg?style=social&label=Star) |             [清华大学](https://github.com/THUDM)             |
| Chinese-LLaMA-Alpaca-2 | 2023-09  | 7/13B       | [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)![Star](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca-2.svg?style=social&label=Star) |                     哈工大讯飞联合实验室                     |

## 大模型评估基准

### 1. C-Eval ![Star](https://img.shields.io/github/stars/SJTU-LIT/ceval.svg?style=social&label=Star)

提供了13948个多项选择题的C-Eval是一个全方位的中文基本模型评估工具。该套件覆盖了52个学科并且分为四个难度等级。[论文](https://arxiv.org/abs/2305.08322)内有更多详细信息。

[[官方网站](https://cevalbenchmark.com/)] [[Github](https://github.com/SJTU-LIT/ceval)] [[论文](https://arxiv.org/abs/2305.08322)]

### 2. FlagEval ![Star](https://img.shields.io/github/stars/FlagOpen/FlagEval.svg?style=social&label=Star)

FlagEval的设计初衷是为AI基础模型提供评估，它集中于科学、公正和开放的评价准则和工具。该工具包旨在从多维度评估基础模型，推进技术创新和行业应用。

[[官方网站](https://cevalbenchmark.com/)] [[Github](https://github.com/FlagOpen/FlagEval)]

### 3. SuperCLUElyb ![Star](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUElyb.svg?style=social&label=Star)

SuperCLUE琅琊榜是中文大模型评估的标准。它采用众包方式，提供匿名和随机对战。Elo评级系统，广泛应用于国际象棋，也被用于此评估中。

[[官方网站](https://www.superclueai.com/)] [[Github](https://github.com/CLUEbenchmark/SuperCLUElyb)]

### 4. XiezhiBenchmark ![Star](https://img.shields.io/github/stars/mikegu721/xiezhibenchmark.svg?style=social&label=Star)

XiezhiBenchmark涵盖13个学科的220,000个多项选择题和15,000个问题。评估结果显示，大型语言模型在某些领域上超越了人类表现，而在其他领域上仍有待提高。

[[官方网站](https://chat.openai.com/c/c0585ba8-1b9a-4a73-96f4-d39747519501)] [[Github](https://github.com/mikegu721/xiezhibenchmark)] [[论文](https://arxiv.org/abs/2306.05783)]

### 5. Open LLM Leaderboard

HuggingFace推出的LLM评估榜单，以英语为主，集中于大语言模型和聊天机器人的评估。任何社区成员都可以提交模型以供自动评估。

[[官方网站](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)]

### 6. 中文大模型安全评测平台 ![Star](https://img.shields.io/github/stars/thu-coai/Safety-Prompts.svg?style=social&label=Star)

该平台利用完备的评测框架，涉及多个安全类别如仇恨言论、隐私等，进行大模型的安全评估。

[[官方网站](http://coai.cs.tsinghua.edu.cn/leaderboard/)] [[Github](https://github.com/thu-coai/Safety-Prompts)] [[论文](https://arxiv.org/abs/2304.10436)]

### 7. OpenCompass大语言模型评测 ![Star](https://img.shields.io/github/stars/open-compass/opencompass.svg?style=social&label=Star)

OpenCompass是一个开源平台，专为大语言模型和多模态模型设计。即便是千亿参数模型，也能迅速完成评测。

[[官方网站](https://opencompass.org.cn/)] [[Github](https://github.com/open-compass/opencompass)]



## 没有卡的条件下我们能做什么？



### LLM压缩

#### [LLM量化](https://github.com/liguodongiot/llm-action/tree/main/model-compression/quantization)


训练后量化：

- SmoothQuant
- ZeroQuant
- GPTQ
- LLM.int8()


量化感知训练：

- [大模型量化感知训练开山之作：LLM-QAT](https://zhuanlan.zhihu.com/p/647589650)

量化感知微调：

- QLoRA
- PEQA

#### LLM剪枝


**结构化剪枝**：

- LLM-Pruner 

**非结构化剪枝**：

- SparseGPT
- LoRAPrune
- Wanda



#### LLM知识蒸馏

- [大模型知识蒸馏概述]()

**Standard KD**:

使学生模型学习教师模型(LLM)所拥有的常见知识，如输出分布和特征信息，这种方法类似于传统的KD。


- MINILLM
- GKD


**EA-based KD**:

不仅仅是将LLM的常见知识转移到学生模型中，还涵盖了蒸馏它们独特的涌现能力。具体来说，EA-based KD又分为了上下文学习（ICL）、思维链（CoT）和指令跟随（IF）。


In-Context Learning：

- In-Context Learning distillation


Chain-of-Thought：

- MT-COT 
- Fine-tune-CoT 
- DISCO 
- SCOTT 
- SOCRATIC CoT

Instruction Following：

- Lion


#### 低秩分解

低秩分解旨在通过将给定的权重矩阵分解成两个或多个较小维度的矩阵，从而对其进行近似。低秩分解背后的核心思想是找到一个大的权重矩阵W的分解，得到两个矩阵U和V，使得W≈U V，其中U是一个m×k矩阵，V是一个k×n矩阵，其中k远小于m和n。U和V的乘积近似于原始的权重矩阵，从而大幅减少了参数数量和计算开销。

在LLM研究的模型压缩领域，研究人员通常将多种技术与低秩分解相结合，包括修剪、量化等。

- ZeroQuant-FP（低秩分解+量化）
- LoRAPrune（低秩分解+剪枝）
