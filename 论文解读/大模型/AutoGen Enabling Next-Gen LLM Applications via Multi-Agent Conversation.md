# AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

论文链接：https://arxiv.org/abs/2308.08155

Github：https://github.com/microsoft/autogen

## 1.论文背景

大型语言模型 (LLM) 正在成为开发强大的代理的关键构建块，为了扩大代理的能力，一个直观的方法就是通过使用多个代理的合作。

## 2. 论文提出观点

论文对此提出的方法是使用多智能体的对话，每个agent通过彼此之间的对话进行合作从而解决问题。

## 3.论文方法

论文提出一种AutoGen框架，AutoGen框架的核心是其代理协同工作的能力。每个代理都有其特定的能力和角色，你需要定义代理之间的互动行为，即当一个代理从另一个代理收到消息时该如何回复。这种方式不仅仅是在定义代理和角色，还在定义它们如何协同工作，从而实现更优的任务完成效果。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/AutoGen%E6%A1%86%E6%9E%B6.png)

有一个泛型ConversableAgent类，有两个核心的内置的子类，**AssistantAgent** 和 **UserProxyAgent**。AssistantAgent 设计为充当 AI 助手，默认使用 LLM（可以是GPT，也可是其他），可以编写 Python 代码给到UserProxyAgent 。

UserProxyAgent 是人类的代理，默认情况下，在每个交互回合中，将人工输入作为代理的回复，若设置全自动回复，会自动触发代码执行。

GroupChatManager支持更复杂的动态组聊天，它可以动态选择下一个说话者，然后将其响应广播给其他代理。

**可定制**：AutoGen 中的代理可以自定义以集成 LLM、人员、工具或它们的组合。根据不同的问题，写对应的子类继承AssistantAgent 和 UserProxyAgent，并在子类中扩展对应的方法。

## 4.实验分析

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/%E5%85%AD%E7%A7%8DAutoGen%E6%A1%86%E6%9E%B6%E7%9A%84%E5%BA%94%E7%94%A8.png)

几个应用问题：数学问题求解，检索增强代码生成和问答，文本世界环境中的决策制定，多agent编码，动态群组聊天，国际象棋对话。

![](https://github.com/zzysos/LLMsStudy/blob/master/%E8%AE%BA%E6%96%87%E8%A7%A3%E8%AF%BB/pic/AutoGen%E5%AE%9E%E9%AA%8C.png)

## 5.论文总结

AutoGen提供了一个多agent系统的通用框架，满足各种实际需求，例如重用、定制和扩展现有agent，以及为它们之间的对话编程。每个agent都可以单独开发、测试和维护，这种方法简化了整体开发和代码管理。通过使用多个代理的交流合作，增强了解决问题的能力。

## 6. 改进方向

论文提出可以探索将现有的agent实现有效集成到我们的多agent框架中，并研究多agent工作流程中自动化和人工控制之间的最佳平衡。研究哪种策略（如agent拓扑结构和对话模式）能带来最有效的多agent对话。

个人认为可以将论文中交谈对话的思想加入之前的各种推理框架，因为以前的框架基本上各个模块是是固定的前后交互，模块间不会相互交流，只有单向的信息传递，加入后可能可以加强效果。