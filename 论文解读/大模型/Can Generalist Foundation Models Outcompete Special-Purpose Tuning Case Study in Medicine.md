# Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine（MedPrompt）

论文链接：https://arxiv.org/abs/2311.16452

## 1.论文背景

基座大模型虽然有很好的通用基础知识，但是对于专有的领域如医学、金融等，缺少专门的训练，因此可能表现并不那么好。使用基座大模型+领域数据进行微调获得一个专用大模型的效果更好，于是便提出是否可以通过更加精巧的 Prompt 技术来解锁大模型的能力以获得近似微调的效果，微软最新研究表明，通过 MedPrompt 提示工程技术，直接让 GPT-4 在医学领域的评测结果超过了医学领域大模型 Med-PaLM2。

## 2.论文解决了什么问题

作者认为在没有额外微调和专家策划的情况下，仅凭提示工程，GPT-4 就能达到专家效果。使用他们提出的最新提示策略Medprompt，在医疗专业领域，GPT-4在MultiMed QA 九个测试集中取得最优结果。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/MedPrompt_SOTA.png)

## 3.论文方法

论文提出的 MedPrompt 方法实际上是一种结合了训练数据的 few-shot 方法，但也不是简单的选择训练数据回答，而是包括三个方法的框架：

1.动态少样本选择（Dynamic few-shot）

结合 KNN 技术借助领域数据动态构建 few-shot 范例，而不是传统的专家手动制作范例，可以为不同的任务输入选择不同的少量示例。

与微调方法相比，动态少样本选择利用了训练数据，但不需要对模型参数进行大量更新。

2.自生成思维链（Self-generated chain of thought）

思维链的大多数方法都是利用专家手动编写带有思维链的简短示例来进行提示，而作者结合前面的动态选择训练数据，使用 GPT-4 来自主生成每道题目的详细思维展示，作为 Prompt 给GPT-4使用。

GPT-4 使用以下提示模版为训练示例生成思维链：

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/Self-generate%20CoT%20tem.png)
![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/CoT%20and%20SG%20CoT.png)

3.选项洗牌集成（Choice Shuffling Ensemble）

将模型在不同选项顺序情况下生成的多个答案进行汇总和分析，减少模型在回答选择题时对特定选项位置的偏好，提高答案的准确性和模型的可靠性。

改变选项顺序→生成多个答案→分析答案的一致性→集成和决策

4.Medprompt 将上述几种方式进行组合，产生一种通用的提示工程策略。

包括两个阶段：预处理阶段和推理步骤，在推理步骤中对测试用例进行最终预测。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/MedPrompt%20%E7%AE%97%E6%B3%95.png)

## 4.实验分析

1.作者给出了很多测试，来证明使用 MedPrompt 方法是可以达到或者接近 fine-tuning 效果的。

作者在 MedQA 数据集上进行了消融实验，下图是 Medprompt 组件的直观图解以及对 MedQA 基准性能贡献。

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/MedPrompt%E5%9B%BE%E8%A7%A3.png)
![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/MedPrompt%E8%A1%A8%E7%8E%B0.png)

2.Medprompt的跨域泛化能力

![](https://github.com/Kayin211/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/MedPrompt%E8%B7%A8%E5%9F%9F%E6%B3%9B%E5%8C%96.png)

## 5.其它

可以进一步研究 Medprompt 策略在非医学领域的适用性，以验证其通用性。

研究如何将 Medprompt 策略应用于非选择题任务，以扩大其应用范围。

## 6.总结

这篇论文证明了使用 Prompt 技术配合领域数据是可以提高基座模型在特定领域的能力的，甚至超过fine-tuning，动态 Prompt 和自生成 CoT 技术给了一种非常好的结合领域数据和 Prompt 的方法，为领域大模型微调提供了另外一种思路。这种对 Prompt 策略的组合似乎可以提高模型效果，但基座模型的强大必须是前提。

通过检索的方法从训练数据中找到近似的问答结果，然后构造 few-shot 案例，嵌入用户的输入中，再让模型回答问题。我觉得这个过程与 RAG 很像，其中构建few-shot 的过程确实给人启发很大，传统的 RAG 仅仅提供将检索的信息作为 Prompt 可能不足以引导模型理解和回答问题的具体上下文。通过将检索到的信息转换为问答样例，可以为模型提供一个清晰、具体的上下文，从而帮助模型更准确地理解和回答特定的问题。

即传统 RAG：向量检索→嵌入用户问答 Prompts

微软新 RAG（MedPrompt）：向量检索→通过思维链构建出问答样例→将该样例作为 few-shot 嵌入用户问答 Prompts
