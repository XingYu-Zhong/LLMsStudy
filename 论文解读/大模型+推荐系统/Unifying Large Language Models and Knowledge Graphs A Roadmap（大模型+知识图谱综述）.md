# Unifying Large Language Models and Knowledge Graphs: A Roadmap（大模型+知识图谱综述）

机构：合肥工业大学，北京工业大学，南洋理工大学，墨尔本大学

论文地址：https://arxiv.org/abs/2306.08302



## 论文背景

作者认为LLM 是黑盒模型，通常无法捕获和访问事实知识。通过知识图谱可以让llm获取事实知识。



![image-20231225164107941](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164107941.png)

llm和知识图谱的优缺点总结。
LLM 优势：一般知识，语言处理，通用性 ；
LLM 缺点：隐式知识、幻觉、不确定性、黑盒，缺乏特定领域的/新知识 。
知识图谱优势：结构知识 、准确性、决定性、可解释性、特定领域的知识 、进化知识； 
知识图谱缺点：不完整 ，缺乏语言理解，未见事实

## 论文提出的问题

作者认为LLM 无法回忆事实，并且通常通过生成事实不正确的陈述，这就是llm的幻觉问题，这些问题严重损害了llm的可信度。

作者觉得LLM 通过概率模型进行推理，这是一个不确定的过程 。尽管一些llm配备了通过应用思维链（cot）来解释他们的预测，但它们的推理解释也存在幻觉问题，这个现象严重损坏了llm在高风险事件上的应用场景，比如医学问诊上，金融交易上，法律咨询上等等。

作者认为为了解决幻觉问题，一个潜在的解决方案是将知识图 (KG) 合并到 LLM 中。知识图(KGs)，以三元组的方式存储巨大的事实，即(头实体、关系、尾实体)，但是知识图谱也有缺陷，作者认为我们很难对现实世界进行知识图谱建模，同时当我们好不容易建立图谱后很难进行动态修改。

## 论文提出的解决方案

作者认为LLM 和 KG 本质上是相互连接的，可以相互增强。在 KG 增强的 LLM 中，KG 不仅可以合并到 LLM 的预训练和微调阶段以提供外部知识，还可以用于分析 LLM 并提供可解释性

## 论文介绍

### llm部分

![image-20231225164250463](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164250463.png)

llm可以分为三组:

1)仅编码器的llm, 

2)编码器-解码器llm

3)仅解码器的llm。



仅编码器大型语言模型仅使用编码器对句子进行编码并理解单词之间的关系。这些模型的常见训练范式是预测输入句子中的掩码词。这种方法是无监督的，可以在大规模语料库上进行训练。他们主要做文本分类和命名实体识别这些任务



编码器-解码器大型语言模型采用编码器和解码器模块。编码器模块负责将输入图像编码为隐藏空间，解码器用于生成目标输出文本。编码器-解码器llm中的训练策略可以更加灵活。例如，T5 通过掩蔽和预测掩蔽词的跨度来预训练。UL2统一了几个训练目标，例如不同的掩蔽跨度和掩蔽频率。编码器-解码器 LLM（例如 T0 、ST-MoE和 GLM-130B）能够直接解决基于某些上下文生成句子的任务，例如求和、翻译和问答。



仅解码器的大型语言模型仅采用解码器模块来生成目标输出文本。这些模型的训练范式是预测句子中的下一个单词。大规模仅解码器llm通常可以从几个例子或简单的指令执行下游任务，而无需添加预测头或微调。许多最先进的 LLM（例如，Chat-GPT 和 GPT-44）遵循仅解码器的架构。这些模型大多数是闭源的



![image-20231225164400480](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164400480.png)

### 提升工程部分

作者认为提示工程是一个新颖的领域，专注于创建和细化提示，以最大限度地提高大型语言模型 (LLM) 在各种应用程序和研究领域的有效性。如图 所示，提示是为任务指定的 LLM 的一系列自然语言输入，例如情感分类。提示可以包含几个元素，即 1) 指令、2) 上下文和 3) 输入文本。指令是一个短句，指示模型执行特定任务。上下文为输入文本或少样本示例提供上下文。输入文本是模型需要由模型处理的文本。

![image-20231225164431228](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164431228.png)

### 知识图谱部分

![image-20231225164518463](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164518463.png)

作者把知识图谱分成了四类：

百科全书知识图谱

常识知识图谱

特殊领域知识图谱

多模态知识图谱



其中多模态知识图谱与传统的只包含文本信息的知识图不同，多模态知识图代表了图像、声音和视频等多种模式的事实。将文本和图像信息合并到知识图中。这些知识图可用于各种多模态任务，如图像-文本匹配、视觉问答和推荐。



### llm结合知识图谱方法

![image-20231225164759198](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164759198.png)

两者合作的方式有三种，
1) 知识图谱增强llm，在llm的预训练和推理阶段结合知识图谱，或为了增强对llm学习到的知识的理解;
2) LLM增强知识图谱，利用llm进行嵌入、完成、构建、图到文本生成和问题回答等不同知识图谱任务;
3)协同llm +知识图谱，其中llm和知识图谱扮演相同的角色，并以互惠的方式工作，以增强llm和知识图谱，以实现数据和知识驱动的双向推理。

![image-20231225164721256](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164721256.png)

作者对第三种方式表示推荐

![image-20231225164922377](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225164922377.png)

大体上llm合知识图谱可以分成这样

![image-20231225165125658](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225165125658.png)

最上面三个就是刚刚介绍的三种合作的方式，每一种下还有不同的阶段

####  知识图谱增强llm

有三个小类，分别是kg增强的LLM预训练，旨在在训练前阶段向llm注入知识。然后 是KG 增强的 LLM 推理，它使 LLM 在生成句子时能够考虑最新的知识。最后是 KG 增强的 LLM 可解释性，旨在通过使用 KG 来提高 LLM 的可解释性。

##### 将 KG 集成到训练目标中（LLM预训练阶段）

ERNIE提出了一种新的词实体对齐训练目标作为预训练目标。具体来说，ERNIE 将文本中提到的句子和相应实体都输入到 LLM 中，然后训练 LLM 来预测知识图中文本标记和实体之间的对齐链接。

![image-20231225165540916](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225165540916.png)

##### 将KG集成到LLM输入中

这类研究集中在将相关知识子图引入llm的输入中。给定一个知识图三元组和相应的句子，ERNIE 3.0 将三元组表示为一系列标记，并直接将它们与句子连接起来。它进一步随机屏蔽句子中的三元组或标记中的关系标记，以更好地将知识与文本表示相结合。

![image-20231225165744017](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225165744017.png)

##### 通过附加融合模块集成 KG

通过在llm中引入额外的融合模块，可以将KGs的信息分别处理和融合到llm中。ERNIE提出了一种文本知识双编码器架构，其中Tencoder首先对输入句子进行编码，然后使用T-encoder的文本表示对知识图进行处理。

![image-20231225170007763](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225170007763.png)

##### 动态知识融合（LLM推理阶段）

在所有文本标记和 KG 实体上计算成对点积分数，分别计算双向注意力分数。此外，在每个关节 LK 层，KG 也是基于注意力分数动态修剪的，以允许后面的层专注于更重要的子 KG 结构。

![image-20231225170157278](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225170157278.png)

其实就是把与问题无关的结点删掉，减轻推理负担

##### 检索增强知识融合

RAG提出将非参数模块和参数模块结合起来处理外部知识。给定输入文本，RAG首先通过 MIPS 搜索非参数模块中的相关 KG 以获得多个文档。然后 RAG 将这些文档视为隐藏变量 z，并将它们馈送到由 Seq2Seq LLM 授权的输出生成器中，作为附加的上下文信息。

![image-20231225170358063](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225170358063.png)

研究表明，使用不同的检索到的文档作为不同生成步骤的条件比仅使用单个文档来指导整个生成过程表现更好。实验结果表明，RAG 在开放域 QA 中优于其他仅参数和非参数的基线模型。RAG 还可以比其他仅参数的基线生成更具体的、多样化的和事实文本。

##### 用于LLM探测的（可解释性）

LAMA 是第一个通过使用 KG 探索 LLM 知识的工作。LAMA首先通过预定义的提示模板将KGs中的事实转换为完形填空语句，然后使用llm来预测缺失的实体。预测结果用于评估存储在llm中的知识。

![image-20231225170637488](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225170637488.png)

##### 用于LLM分析的

KagNet和QA-GNN使llm在每个推理步骤根据知识图生成的结果。通过这种方式，LLM 的推理过程可以通过从 KG 中提取图结构来解释。

![image-20231225170715893](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225170715893.png)

#### LLM增强知识图谱

LLM 集成到 KG 嵌入、KG 完成、KG 构建、KG 到文本生成和 KG 问答

这一块每一步都要对应的描述讲解，优于他不是我研究的重点这里就不过多描述



#### 协同llm +知识图谱

文本语料库和知识图谱都包含巨大的知识。然而，文本语料库中的知识通常是隐式和非结构化的，而 KG 中的知识是显式和结构化的。因此，有必要对齐文本语料库和 KG 中的知识以统一的方式表示它们。

![image-20231225171200836](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20231225171200836.png)

### 总结

作者认为目前在纯文本数据上训练的传统 LLM 并非旨在理解知识图等结构化数据。因此，LLM 可能无法完全掌握或理解 KG 结构传达的信息。

作者认为知识图谱可以帮助llm处理幻觉问题，扩展llm的知识，而llm也可以帮助知识图谱构建，用于黑盒llm知识注入的kg，KG 和 LLM 是两个互补的技术，可以相互协同。然而，现有研究人员较少探索 LLM 和 KG 的协同作用。LLM 和 KG 的期望协同作用将涉及利用这两种技术的优势来克服它们各自的限制。

这是一篇对知识图谱和大模型的综述，描述的十分详细，架构也十分清晰，可以作为入门刊物。