# WizardLM: Empowering Large Language Models to Follow Complex Instructions

## 论文解决的问题
![alt text](https://raw.githubusercontent.com/XingYu-Zhong/LLMsStudy/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/wizardlm1.png)
就是作者提出了一种方法叫Evol-Instruct，主要目的就是为了扩充数据，通过对原始数据的添加约束、深化、具体化、增加推理步骤和复杂的输入。从而将原始数据的复杂性和数据量都得到了扩充。通过这种方法，我们能够生成大量高质量的指令数据，并用这些数据对LLaMA模型进行微调，得到了性能更优的WizardLM模型。人类评估和GPT-4自动评估的结果表明，WizardLM在处理复杂指令方面优于现有的ChatGPT模型。

## 具体解决方案
Evol-Instruct的核心思想是迭代地“进化”初始指令集，通过逐步增加指令的复杂性，生成多样化和高难度的指令数据。

具体来说，Evol-Instruct包括两个主要组成部分：指令演化器（Instruction Evolver）和指令消除器（Instruction Eliminator）。指令演化器利用特定的提示（prompts）来增强指令的复杂性和难度，这包括增加约束、深化、具体化、增加推理步骤和复杂化输入等操作。这些操作通过向LLM提供特定提示来实现，例如要求LLM在保留原始指令内容的基础上添加额外的约束或要求，从而使指令变得更加复杂。此外，还有“广度演化”（In-breadth Evolving），它通过变异生成全新的、与给定指令同样复杂的指令，以增加数据集的多样性。

指令消除器的作用是过滤掉那些未能成功进化的指令。这些指令可能因为过于简单、无法生成响应或与原始指令相似度过高而被排除。通过这种淘汰机制，只有成功的演化指令才会被添加到指令池中，用于后续的模型训练。
![alt text](https://raw.githubusercontent.com/XingYu-Zhong/LLMsStudy/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/wizardlm2.png)

## prompt例子
### 普通
我想你充当提示重写器。您的目标是将给定的提示重写为更复杂的版本，以使这些著名的人工智能系统（例如，ChatGPT 和 GPT4）更难处理。但是重写的提示必须是合理的，必须被人类理解和回应。您的重写不能省略非文本部分，例如#Given Prompt# 中的表和代码：.此外，请不要在#Given Prompt# 中省略输入。你应该用以下方法使给定的提示复杂化:请在#given Prompt#中添加更多的约束/要求，你应该尽量不要使#Rewritten Prompt#变得冗长，#Rewritten Prompt#只能在#Given Prompt#中添加10到20个单词。

'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt# #Given Prompt#: <Here is instruction.> #Rewritten Prompt#:

### 复杂
我想你充当提示重写器。您的目标是将给定的提示重写为更复杂的版本，以使这些著名的人工智能系统（例如，ChatGPT 和 GPT4）更难处理。但是重写的提示必须是合理的，必须被人类理解和回应。您必须添加 [XML 数据] 格式数据作为 [重写提示] 中的输入数据

#Given Prompt#: <Here is Demonstration instruction 1.> 

#Rewritten Prompt#: <Here is Demonstration Example 1.>

### 进化突变
我想你充当提示创建者。你的目标是从#Given Prompt#中汲取灵感，以创建全新的提示。这个新的提示应该属于与#Given Prompt#相同的域，但更罕见。#Created Prompt# 的 LENGTH 和难度级别应该与 #Given Prompt# 相似。#Created Prompt# 必须是合理的，必须被人类理解和回应。

'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#. #Given Prompt#: <Here is instruction.> #Created Prompt#:

### 进化失败条件
进化失败的认定
- 1.先让gpt去看两个提示是否相同（它们具有相同的约束和要求。他们的调查深度和广度相同。）
- 2.如果出现了对不起和长度小于80个单词
- 3.如果只是包含标点符号和停止词。
- 4.进化指令显然是从进化提示中复制了一些单词，如“给定提示”、“重写提示”和“#重写提示#”等。

## 微调

微调的时候把升级来的数据和原有数据合在一起，然后打散，让数据分布均匀

数据生成时候gpt的参数：
ChatGPT 生成响应。我们使用 1 的温度生成响应并将最大令牌数设置为 2048。此外，我们将频率惩罚设置为零，将 top-p 设置为 0.9。

微调的参数：我们采用 Adam 优化器作为 2 ×10−5 的初始学习率，最大令牌数为 2048，每个 GPU 的批量大小为 8。我们在 8 个 V100 GPU 上训练我们的模型，Deepspeed Zero-3 在 3 个 epoch 上训练了 70 小时。

