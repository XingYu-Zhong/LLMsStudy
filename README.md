# LLMsStudy
收集大语言模型的学习路径和各种最佳实践
# LLM的初步探究

随着大模型的热度越来越高，我们有必要对大于语言模型进行深入的研究

## 目录

大语言模型是什么？

大模型推荐

大模型评估基准

没有卡的条件下我们能做什么？

怎么开始学习？

## 大语言模型是什么？

### 大模型的发展脉络

在2017年，Transformer架构的出现导致深度学习模型的参数超越了1亿，从此RNN和CNN被Transformer取代，开启了大模型的时代。谷歌在2018年推出BERT，此模型轻松刷新了11个NLP任务的最佳记录，为NLP设置了一个新的标杆。它不仅开辟了新的研究和训练方向，也使得预训练模型在自然语言处理领域逐渐受到欢迎。此外，这一时期模型参数的数量也首次超过了3亿。到了2020年，OpenAI发布了GPT-3，其参数数量直接跃升至1750亿。2021年开始，Google先后发布了Switch Transformer和GLaM，其参数数量分别首次突破万亿和1.2万亿，后者在小样本学习上甚至超越了GPT-3。

### Transformer结构

![image-20231010150630814](https://picgo-zxy.oss-cn-guangzhou.aliyuncs.com/typoreimgs/image-20231010150630814.png)

Transformer是由Google Brain在2017年提出的一种新颖的网络结构。相对于RNN，它针对其效率问题和长程依赖传递的挑战进行了创新设计，并在多个任务上均展现出优越的性能。

如下图所示的是Transformer的架构细节。其核心技术是自注意力机制（Self-Attention）。简单地说，自注意力机制允许一个句子中的每个词对句子中的所有其他词进行加权，以生成一个新的词向量表示。这个过程可以看作是每个词都经过了一次类似卷积或聚合的操作。这种机制提高了模型对于上下文信息的捕获能力。

### MOE结构

![image-20231010150553380](https://picgo-zxy.oss-cn-guangzhou.aliyuncs.com/typoreimgs/image-20231010150553380.png)

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
微调和数据增强


### LLM压缩

#### LLM量化

训练后量化：

- SmoothQuant
- ZeroQuant
- GPTQ
- LLM.int8()


量化感知训练：


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

- 大模型知识蒸馏概述

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
https://zhuanlan.zhihu.com/p/646831196

低秩分解旨在通过将给定的权重矩阵分解成两个或多个较小维度的矩阵，从而对其进行近似。低秩分解背后的核心思想是找到一个大的权重矩阵W的分解，得到两个矩阵U和V，使得W≈U V，其中U是一个m×k矩阵，V是一个k×n矩阵，其中k远小于m和n。U和V的乘积近似于原始的权重矩阵，从而大幅减少了参数数量和计算开销。

在LLM研究的模型压缩领域，研究人员通常将多种技术与低秩分解相结合，包括修剪、量化等。

- ZeroQuant-FP（低秩分解+量化）
- LoRAPrune（低秩分解+剪枝）

## 怎么开始学习？
### 0. 体验大模型 （入门）

第一步先学会科学上网，最好学会自己搭梯子，这样才能保证你的学习不会被打断。

第二步，体验大模型，可以通过以下方式体验大模型：chatgpt，cluade，bard等等

### 1. 了解大模型（基础）

吴恩达大模型系列课程

https://github.com/datawhalechina/prompt-engineering-for-developers

包括 提示词工程，langchain

学习后可以入门大模型的基本概念和应用


### 2. 查阅论文（进阶）

在了解大模型的基本概念后，可以通过查阅论文来了解大模型的最新进展

https://github.com/Hannibal046/Awesome-LLM

### 3. 代码实践（进阶）

可以自己开始动手实践，可以从以下几个方面入手：

调用大模型的API

自己部署大模型调用

微调大模型








## 论文列表

### 模型实用指南

### BERT-style Language Models: Encoder-Decoder or Encoder-only

- BERT **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**, 2018, [Paper](https://aclanthology.org/N19-1423.pdf)
- RoBERTa **RoBERTa: A Robustly Optimized BERT Pretraining Approach**, 2019, [Paper](https://arxiv.org/abs/1907.11692)
- DistilBERT **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**, 2019, [Paper](https://arxiv.org/abs/1910.01108)
- ALBERT **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**, 2019, [Paper](https://arxiv.org/abs/1909.11942)
- UniLM **Unified Language Model Pre-training for Natural Language Understanding and Generation**, 2019 [Paper](https://arxiv.org/abs/1905.03197)
- ELECTRA **ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS**, 2020, [Paper](https://openreview.net/pdf?id=r1xMH1BtvB)
- T5 **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [Paper](https://arxiv.org/abs/1910.10683)
- GLM **"GLM-130B: An Open Bilingual Pre-trained Model"**. 2022. [Paper](https://arxiv.org/abs/2210.02414)
- AlexaTM **"AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model"**. *Saleh Soltan et al.* arXiv 2022. [Paper](https://arxiv.org/abs/2208.01448)
- ST-MoE **ST-MoE: Designing Stable and Transferable Sparse Expert Models**. 2022 [Paper](https://arxiv.org/abs/2202.08906)


### GPT-style Language Models: Decoder-only

- GPT **Improving Language Understanding by Generative Pre-Training**. 2018. [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- GPT-2 **Language Models are Unsupervised Multitask Learners**. 2018. [Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- GPT-3 **"Language Models are Few-Shot Learners"**. NeurIPS 2020. [Paper](https://arxiv.org/abs/2005.14165)
- OPT **"OPT: Open Pre-trained Transformer Language Models"**. 2022. [Paper](https://arxiv.org/abs/2205.01068)
- PaLM **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al.* arXiv 2022. [Paper](https://arxiv.org/abs/2204.02311)
- BLOOM  **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. 2022. [Paper](https://arxiv.org/abs/2211.05100)
- MT-NLG **"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model"**. 2021. [Paper](https://arxiv.org/abs/2201.11990)
- GLaM **"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"**. ICML 2022. [Paper](https://arxiv.org/abs/2112.06905)
- Gopher **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**. 2021. [Paper](http://arxiv.org/abs/2112.11446v2)
- chinchilla **"Training Compute-Optimal Large Language Models"**. 2022. [Paper](https://arxiv.org/abs/2203.15556)
- LaMDA **"LaMDA: Language Models for Dialog Applications"**. 2021. [Paper](https://arxiv.org/abs/2201.08239)
- LLaMA **"LLaMA: Open and Efficient Foundation Language Models"**. 2023. [Paper](https://arxiv.org/abs/2302.13971v1)
- GPT-4 **"GPT-4 Technical Report"**. 2023. [Paper](http://arxiv.org/abs/2303.08774v2)
- BloombergGPT **BloombergGPT: A Large Language Model for Finance**, 2023, [Paper](https://arxiv.org/abs/2303.17564)
- GPT-NeoX-20B: **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"**. 2022. [Paper](https://arxiv.org/abs/2204.06745)
- PaLM 2: **"PaLM 2 Technical Report"**. 2023. [Tech.Report](https://arxiv.org/abs/2305.10403)
- LLaMA 2: **"Llama 2: Open foundation and fine-tuned chat models"**. 2023. [Paper](https://arxiv.org/pdf/2307.09288)
- Claude 2: **"Model Card and Evaluations for Claude Models"**. 2023. [Model Card](https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf)



### 数据实用指南


### 预训练数据
- **RedPajama**, 2023. [Repo](https://github.com/togethercomputer/RedPajama-Data)
- **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**, Arxiv 2020. [Paper](https://arxiv.org/abs/2101.00027)
- **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/)
- **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361)
- **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158)
- **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, 2022. [Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
### 微调数据
- **Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach**, EMNLP 2019. [Paper](https://arxiv.org/abs/1909.00161)
- **Language Models are Few-Shot Learners**, NIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
- **Does Synthetic Data Generation of LLMs Help Clinical Text Mining?** Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04360)
### 测试数据
- **Shortcut learning of large language models in natural language understanding: A survey**, Arxiv 2023. [Paper](https://arxiv.org/abs/2208.11857)
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective** Arxiv, 2023. [Paper](https://arxiv.org/abs/2302.12095)
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** Arxiv 2019. [Paper](https://arxiv.org/abs/1905.00537)


### 传统的 NLU 任务

- **A benchmark for toxic comment classification on civil comments dataset** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.11125)
- **Is chatgpt a general-purpose natural language processing task solver?** Arxiv 2023[Paper](https://arxiv.org/abs/2302.06476)
- **Benchmarking large language models for news summarization** Arxiv 2022 [Paper](https://arxiv.org/abs/2301.13848)
### 生成任务
- **News summarization and evaluation in the era of gpt-3** Arxiv 2022 [Paper](https://arxiv.org/abs/2209.12356)
- **Is chatgpt a good translator? yes with gpt-4 as the engine** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.08745)
- **Multilingual machine translation systems from Microsoft for WMT21 shared task**, WMT2021 [Paper](https://aclanthology.org/2021.wmt-1.54/)
- **Can ChatGPT understand too? a comparative study on chatgpt and fine-tuned bert**, Arxiv 2023, [Paper](https://arxiv.org/pdf/2302.10198.pdf)




### 知识密集型任务
- **Measuring massive multitask language understanding**, ICLR 2021 [Paper](https://arxiv.org/abs/2009.03300)
- **Beyond the imitation game: Quantifying and extrapolating the capabilities of language models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2206.04615)
- **Inverse scaling prize**, 2022 [Link](https://github.com/inverse-scaling/prize)
- **Atlas: Few-shot Learning with Retrieval Augmented Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2208.03299)
- **Large Language Models Encode Clinical Knowledge**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.13138)


### 缩放能力

- **Training Compute-Optimal Large Language Models**, NeurIPS 2022 [Paper](https://openreview.net/pdf?id=iBBcRUlOAPR)
- **Scaling Laws for Neural Language Models**, Arxiv 2020 [Paper](https://arxiv.org/abs/2001.08361)
- **Solving math word problems with process- and outcome-based feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.14275)
- **Chain of thought prompting elicits reasoning in large language models**, NeurIPS 2022 [Paper](https://arxiv.org/abs/2201.11903)
- **Emergent abilities of large language models**, TMLR 2022 [Paper](https://arxiv.org/abs/2206.07682)
- **Inverse scaling can become U-shaped**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.02011)
- **Towards Reasoning in Large Language Models: A Survey**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10403)


### 特定任务
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, Arixv 2022 [Paper](https://arxiv.org/abs/2208.10442)
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.06794)
- **AugGPT: Leveraging ChatGPT for Text Data Augmentation**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.13007)
- **Is gpt-3 a good data annotator?**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10450)
- **Want To Reduce Labeling Cost? GPT-3 Can Help**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.354/)
- **GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.192/)
- **LLM for Patient-Trial Matching: Privacy-Aware Data Augmentation Towards Better Performance and Generalizability**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16756)
- **ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.15056)
- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16634)
- **GPTScore: Evaluate as You Desire**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.04166)
- **Large Language Models Are State-of-the-Art Evaluators of Translation Quality**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.14520)
- **Is ChatGPT a Good NLG Evaluator? A Preliminary Study**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04048)


### 效率
1. 花费
- **Openai’s gpt-3 language model: A technical overview**, 2020. [Blog Post](https://lambdalabs.com/blog/demystifying-gpt-3)
- **Measuring the carbon intensity of ai in cloud instances**, FaccT 2022. [Paper](https://dl.acm.org/doi/abs/10.1145/3531146.3533234)
- **In AI, is bigger always better?**, Nature Article 2023. [Article](https://www.nature.com/articles/d41586-023-00641-w)
- **Language Models are Few-Shot Learners**, NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- **Pricing**, OpenAI. [Blog Post](https://openai.com/pricing)
2. 延迟
- HELM: **Holistic evaluation of language models**, Arxiv 2022. [Paper](https://arxiv.org/abs/2211.09110)
3. 微调方法
- **LoRA: Low-Rank Adaptation of Large Language Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2106.09685)
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, ACL 2021. [Paper](https://aclanthology.org/2021.acl-long.353/)
- **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.8/)
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, Arxiv 2022. [Paper](https://arxiv.org/abs/2110.07602)
4. 预训练系统
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)
- **Reducing Activation Recomputation in Large Transformer Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)


### 可信度
1. 稳健性和校准
- **Calibrate before use: Improving few-shot performance of language models**, ICML 2021. [Paper](http://proceedings.mlr.press/v139/zhao21c.html)
- **SPeC: A Soft Prompt-Based Calibration on Mitigating Performance Variability in Clinical Notes Summarization**, Arxiv 2023. [Paper](https://arxiv.org/abs/2303.13035)
  
2. 虚假偏差
- **Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning**, Findings of ACL 2023 [Paper](https://aclanthology.org/2023.findings-acl.284/)
- **Shortcut learning of large language models in natural language understanding: A survey**, 2023 [Paper](https://arxiv.org/abs/2208.11857)
- **Mitigating gender bias in captioning system**, WWW 2020 [Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449950)
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models**, ICML 2021 [Paper](https://arxiv.org/abs/2102.09690)
- **Shortcut Learning in Deep Neural Networks**, Nature Machine Intelligence 2020 [Paper](https://www.nature.com/articles/s42256-020-00257-z)
- **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?**, NAACL 2022 [Paper](https://aclanthology.org/2022.naacl-main.167/)
  
3. 安全问题
- **GPT-4 System Card**, 2023 [Paper](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- **The science of detecting llm-generated texts**, Arxiv 2023 [Paper](https://arxiv.org/pdf/2303.07205.pdf)
- **How stereotypes are shared through language: a review and introduction of the aocial categories and stereotypes communication (scsc) framework**, Review of Communication Research, 2019 [Paper](https://research.vu.nl/en/publications/how-stereotypes-are-shared-through-language-a-review-and-introduc)
- **Gender shades: Intersectional accuracy disparities in commercial gender classification**, FaccT 2018 [Paper](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)


### 基准指令调整

- FLAN: **Finetuned Language Models Are Zero-Shot Learners**, Arxiv 2021 [Paper](https://arxiv.org/abs/2109.01652)
- T0: **Multitask Prompted Training Enables Zero-Shot Task Generalization**, Arxiv 2021 [Paper](https://arxiv.org/abs/2110.08207)
- **Cross-task generalization via natural language crowdsourcing instructions**, ACL 2022 [Paper](https://aclanthology.org/2022.acl-long.244.pdf)
- Tk-INSTRUCT: **Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks**, EMNLP 2022 [Paper](https://aclanthology.org/2022.emnlp-main.340/)
- FLAN-T5/PaLM: **Scaling Instruction-Finetuned Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2210.11416)
- **The Flan Collection: Designing Data and Methods for Effective Instruction Tuning**, Arxiv 2023 [Paper](https://arxiv.org/abs/2301.13688)
- **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization**, Arxiv 2023 [Paper](https://arxiv.org/abs/2212.12017)

### 对齐

- **Deep Reinforcement Learning from Human Preferences**, NIPS 2017 [Paper](https://arxiv.org/abs/1706.03741)
- **Learning to summarize from human feedback**, Arxiv 2020 [Paper](https://arxiv.org/abs/2009.01325)
- **A General Language Assistant as a Laboratory for Alignment**, Arxiv 2021 [Paper](https://arxiv.org/abs/2112.00861)
- **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2204.05862)
- **Teaching language models to support answers with verified quotes**, Arxiv 2022 [Paper](https://arxiv.org/abs/2203.11147)
- InstructGPT: **Training language models to follow instructions with human feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2203.02155)
- **Improving alignment of dialogue agents via targeted human judgements**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.14375)
- **Scaling Laws for Reward Model Overoptimization**, Arxiv 2022 [Paper](https://arxiv.org/abs/2210.10760)
- Scalable Oversight: **Measuring Progress on Scalable Oversight for Large Language Models**, Arxiv 2022 [Paper](https://arxiv.org/pdf/2211.03540.pdf)

#### 安全调整

- **Red Teaming Language Models with Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2202.03286)
- **Constitutional ai: Harmlessness from ai feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.08073)
- **The Capacity for Moral Self-Correction in Large Language Models**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.07459)
- **OpenAI: Our approach to AI safety**, 2023 [Blog](https://openai.com/blog/our-approach-to-ai-safety)

#### 真实性排列（诚实）

- **Reinforcement Learning for Language Models**, 2023 [Blog](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)

#### 提示实用指南（有用）

- **OpenAI Cookbook**. [Blog](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
- **Prompt Engineering**. [Blog](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- **ChatGPT Prompt Engineering for Developers!** [Course](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

#### 开源社区

- **Self-Instruct: Aligning Language Model with Self Generated Instructions**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10560)
- **Alpaca**. [Repo](https://github.com/tatsu-lab/stanford_alpaca)
- **Vicuna**. [Repo](https://github.com/lm-sys/FastChat)
- **Dolly**. [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- **DeepSpeed-Chat**. [Blog](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- **GPT4All**. [Repo](https://github.com/nomic-ai/gpt4all)
- **OpenAssitant**. [Repo](https://github.com/LAION-AI/Open-Assistant)
- **ChatGLM**. [Repo](https://github.com/THUDM/ChatGLM-6B)
- **MOSS**. [Repo](https://github.com/OpenLMLab/MOSS)
- **Lamini**. [Repo](https://github.com/lamini-ai/lamini/)/[Blog](https://lamini.ai/blog/introducing-lamini)


