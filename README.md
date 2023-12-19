# Llama2RNN.c：一个用C语言实现的终身 RNN 模型

[![zh](https://img.shields.io/badge/zh-简体中文-red.svg)](README.md)
[![en](https://img.shields.io/badge/en-English-green.svg)](README.en.md)

这是一个使用 Llama2 权重设计的循环神经网络（RNN）模型，旨在无限期运行（终身）。

- **llama2**: 可以使用 llama2 各种版本模型的权重
- **rnn**: 每个token的 attention sequence 长度固定，计算和内存开销不会增加，理论上支持无限长序列，可以从硬盘读取和保存记忆
- **.c**: 可以在本地设备上运行，甚至是移动平台


## Memory Attention：一种不需要考虑位置编码外推的RNN结构

这里的主要思路是将 max_seq_len 拆分为长度更小为 mem_seq_len 的chunks，不同chunks之间通过RNN的形式连接中间状态。这样做的主要优势在于推理的时间和空间复杂度会更少，并支持无限长序列。

### 效果对比

| method\seq_len          | 256    | 512    | 1024   | 4096   | 32768  |
| ----------------------- | ------ | ------ | ------ | ------ | ------ |
| Attention Interpolation | 1.0583 | 1.3335 | 2.2598 | 4.1215 | 4.7887 |
| Memory Attention        | 1.0751 | 1.0611 | 1.0562 | 1.0321 | 0.9400 |

以上模型是在训练长度为256的tinistory的文本生成任务上训练的，性能用的是token预测的交叉熵损失。以下长度外推方案都是在没有微调模型的结果。其中 memory attention 的attention seq len为32。从结果可以看出:
* attention外推的各种改进方案只能缓解泛化问题，但仍然不会有序列长度收益存在。也就是说，随序列长度的增加性能会变好
* 而memory attention可以实现外推长度的性能收益，而且明确有长度越长性能越好。

示例
```bash
# mode = llama2Rnn_toy20M_q80.bin, train_seq_len = 256, attention_seq_len = 32
Memory at (null) is not exists.
Initialize memory.
(2023-12-19 14:16:05)  User: 你能为我做什么？
Assistant:  作为一个AI助手，我没有个人情感和立场，但我可以提供一些有关人类发展的实用建议，如下所示：

1. 坚持学习和成长：不断学习和发展自己可以提高自己的技能和知识，帮助你在不同的领域中更好地发挥自己的潜力。

2. 建立良好的人际关系：与他人建立良好的人际关系可以让你获得支持和帮助，并且可以在社交场合中获得更多的机会和信息。

3. 坚持健康生活方式：保持健康的生活方式可以提高身体和心理健康，减轻压力和焦虑，从而提高生活质量。

4. 坚持适度的运动：适度的运动可以提高身体素质和心理健康，同时也可以减轻压力和焦虑，从而提高生活质量。

5. 保持积极的心态：保持积极的心态可以帮助你更好地应对生活中的挑战和困难，从而提高生活质量。

(2023-12-19 14:16:20)  User:
```

能够看到虽然模型的训练长度只有256，而注意力长度只有32，却能生成更长的连贯回复。

## 如何使用

### 1. 编译

要编译`llama2Rnn.c`代码，有以下两种选择：

#### 1.1 不支持 OpenMP 的快速编译

要快速编译，不使用 OpenMP 支持，请使用以下命令：

```bash
make runfast
```

#### 1.2 支持 OpenMP 的编译

要编译并支持 OpenMP，请使用以下命令：

```bash
make runomp
```

### 2. 下载模型和分词器

下载所需的[分词器](https://drive.google.com/file/d/1KJei_OZHFXsc8vgqz7ZGu7V8Nw-TSwFm/view?usp=drive_link)和[模型](https://drive.google.com/file/d/10UOsLSmLEWMfGitKTk8J-tbrL5J-4P6l/view?usp=drive_link)文件。可选以及后续更新模型都在[这里](https://drive.google.com/drive/folders/1Px5IzuUY-H2I-bd0PRsvS0rCg9Vm7iC9?usp=sharing)

```bash
# (internal aws)所有可用的模型都在 s3://lsy/llama2rnn.c/，后续更新模型也会在这里

oss cp s3://lsy/llama2rnn.c/llama2Rnn_toy20M_q80.bin .
oss cp s3://lsy/llama2rnn.c/llama2_tokenizer.bin .
```

### 3. 运行模型

要无限期运行 Llama2RNN 模型，请使用以下命令：

```bash
./runqm llama2Rnn_toy20M_q80.bin -z llama2_tokenizer.bin -o mem20M.bin -m chat
```

## 模型列表

| model   | settings                  |
| ------- | ------------------------- |
| 20M     | 英文模型，数据为wikipedia |
| 178M    | 英文模型，数据为wikipedia |
| 178M_zh | 中文模型，数据包括moss    |

## 更新记录

- 2023.12.19
    - 增加中文模型
    
- 2023.11.13
    - 优化memory save，包括kv cache和token position
- 2023.11.06
    - update 20M(22M) chat model: memory length 从32增加到128（val loss 2.1 -> 1.6）
    - 增加记忆管理功能
- 2023.11.03
    - 量化代码
    - release 20M chat model

## 未来改进

- 调查并合并 `run.cu`（CUDA）
- 添加更多模型，如 1B 和 7B
- （LoRA）Llama2 模型微调
- 添加训练代码
- 支持 .txt 文档输入
- 感知物理时间

## 遗留bug

- molloc prompt 时可能有溢出问题？
- chat encode 有内存访问问题？

## 参考
当前仓库基于[llama2.c](https://github.com/karpathy/llama2.c)构建。

## 许可证

MIT
