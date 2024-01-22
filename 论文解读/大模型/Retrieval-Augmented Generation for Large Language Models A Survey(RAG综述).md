# Retrieval-Augmented Generation for Large Language Models: A Survey（RAG综述）

论文链接：https://arxiv.org/pdf/2312.10997.pdf

机构：同济大学，复旦大学

## 1.论文背景

作者认为 LLM 仍面临诸如幻觉，知识更新和答案缺乏透明度等挑战，从而提出检索增强生成 RAG 手段，通过从外部知识库检索相关信息来辅助大型语言模型回答问题，已经被证明能显著提高回答的准确性，减少模型产生的幻觉，尤其是在知识密集型任务中。


## 2.论文解决了什么问题

作者介绍了 RAG 的优势：提高答案准确性，增强可信度，便于知识更新和引入特定领域知识

概述 RAG 在大模型发展时代的三种范式：原始 RAG（Naive RAG）、高级 RAG（Advanced RAG）和模块化 RAG（Modular RAG）

总结 RAG 的三个主要组成部分：检索器、生成器和增强方法，并着重介绍了从各个角度优化 RAG 在大模型中的表现，实现通过知识检索增强大型语言模型的生成。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAG%E6%97%B6%E9%97%B4%E7%BA%BF.png)

## 3.论文思路

# RAG 本质

让模型获取正确的 Context (上下文)，利用 ICL 的能力，输出正确的响应。它综合利用了固化在模型权重中的参数化知识和存在外部存储中的非参数化知识(知识库、数据库等)。

RAG分为两阶段：

使用编码模型（如 BM25、DPR、ColBERT 等）根据问题找到相关的文档。
生成阶段：以找到的上下文作为基础，系统生成文本。

# RAG vs 微调

RAG 是为了改善 LLM 的生成效果，但它不是改善生成效果的唯一方法。常见方法有：

提示工程，通过例如 few-shot prompt 的手段增强输出
RAG，检索增强，就是本文叙述的方法
微调，对模型进行微调
综合手段，综合利用微调、提示工程和 RAG

应该根据不同应用场景决定采用何种方法

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAG%20vs%20%E5%85%B6%E4%BB%96.png)
![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAG%20vs%20%E5%BE%AE%E8%B0%83.png)

# RAG 几种范式

原始 RAG（Naive RAG）：这是通常所说的 RAG，包括索引，检索，生成。把文本分段，根据用户的 Qurey，去查找分段，输入给模型，然后输出。但是太简单，有各种问题，首先生硬的对文本分段就不科学，然后可能查询到的分段有可能和 Qurey 并不相关，再有输入给 LLM 的文本分段可能有大量的冗余、重复或者噪声信息，让模型不能输出和期望一致的内容。

高级 RAG（Advanced RAG）：对原始 RAG 进行了优化。主要是针对检索进行了改善，包括 Preretrieval(检索前)，Post-retrieval(检索后) 和 Retrieval Process(检索中) 的各种改善方法。检索前包括建立多种文档索引、利用滑动窗口对文本进行分块；检索中包括多路召回，Embedding 模型微调，包括之前提到的StepBack-prompt，检索后包括重排(Re-rank)，提示压缩等。

模块化 RAG（Modular RAG）：模块化方法允许根据具体问题调整模块和流程，利用大模型自身的"反思"能力等，构建起 RAG 新的范式。上面两种方法都是单一的流水线模式，检索结束之后交给模型，然后模型输出结果。但是在论文中的 Modular RAG 方法中，递归的调用了 LLM 的能力，例如利用模型来反思、评估第一次输出，然后再输出新的结果。或者是自适应 RAG，让模型自己决定什么时候调用检索工具。这其实有点像实现一个 RAG Agent。论文表示这种模块化的 RAG 范式正逐渐成为 RAG 领域的趋势。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAG%20Framework.png)

# 增强 RAG 效果的方法

论文从检索器，生成器，增强方法等角度描述如何获得更好的效果

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/RAG%20Components.png)

# RAG 评估方法

主要有两种方法来评估 RAG 的有效性：独立评估和端到端评估。独立评估涉及对检索模块和生成模块（即阅读和合成信息）的评估。端到端评估是对 RAG 模型针对特定输入生成的最终响应进行评估，涉及模型生成的答案与输入查询的相关性和一致性。并简单介绍了 RAGAS 和 ARES 两种评估框架。


## 4.展望

论文讨论了 RAG 的三大未来发展方向：垂直优化、横向扩展以及 RAG 生态系统的构建。

垂直优化主要研究方向是：长上下文的处理问题，鲁棒性研究，RAG 与微调（Fine-tuning）的协同作用，以及如何在大规模知识库场景中提高检索效率和文档召回率，如何保障企业数据安全——例如防止 LLM 被诱导泄露文档的来源、元数据或其他敏感信息。

水平扩展主要研究方向是：从最初的文本问答领域出发，RAG 的应用逐渐拓展到更多模态数据，包括图像、代码、结构化知识、音视频等。在这些领域，已经涌现出许多相关研究成果。

生态系统主要介绍了 Langchain、LlamaIndex 等常见的技术框架。


## 5.总结

可以看到，简单的 RAG 和复杂的 RAG 之间相差非常大，可以从 RAG 的组件和模式进行优化。同时，RAG 与微调，提示工程的协同作用也可以实现模型的最佳性能。随着各种 Agent 的发展，我认为将来 RAG 也必然会 Agent 化，而 Retriever 就类似于 Agent 的工具之一。