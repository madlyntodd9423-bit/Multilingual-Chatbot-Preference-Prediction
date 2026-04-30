# WSDM Cup Multilingual Chatbot Arena 实验方案

## 1. 项目标题

**基于 mDeBERTa-v3-base 的多语言聊天机器人回答偏好预测研究**

---

## 2. 项目背景

本项目基于 **WSDM Cup Multilingual Chatbot Arena** 数据集展开。  
该任务的目标是：在给定同一个用户问题（prompt）的情况下，根据两个不同模型生成的回答（response A 和 response B），预测人类评审者更偏好哪一个回答。

每条样本包含以下核心信息：

- 一个用户输入问题 `prompt`
- 候选回答 `response_a`
- 候选回答 `response_b`
- 人类偏好标签 `winner`

该任务不同于普通文本分类任务，因为模型不仅需要理解文本内容，还需要在**相同问题背景下比较两个回答的质量、相关性和用户偏好**。此外，该数据集具有多语言特征，因此也涉及多语言文本理解与偏好建模问题。

---

## 3. 研究目标

本项目拟构建一个基于 **mDeBERTa-v3-base** 的多语言偏好预测模型，用于判断在同一 prompt 下，用户更偏好 response A 还是 response B，或者两者相近。

本项目主要希望回答以下问题：

1. 多语言编码模型是否能够有效学习用户对两个回答之间的偏好关系？
2. 简单的 baseline 输入方式能达到怎样的效果？
3. 更合理的截断策略是否能够提升模型表现？
4. 对 response A 和 response B 进行反转增强，是否能够减轻位置偏差并提升模型鲁棒性？

---

## 4. 模型选择

本项目选用 **mDeBERTa-v3-base** 作为基础模型。

### 选择原因

1. **适用于多语言任务**  
   本项目数据集为多语言数据，mDeBERTa-v3-base 作为多语言编码模型，更适合直接处理原始多语言文本。

2. **适合分类任务**  
   本任务本质上属于偏好分类任务，encoder-based 模型更适合进行序列分类建模。

3. **资源开销相对较低**  
   相比 Gemma-2 9B 等大语言模型，mDeBERTa-v3-base 更轻量，训练成本更低，更适合作为课程项目的核心模型。

4. **实验可控性更强**  
   该模型便于进行 baseline、改进实验、消融实验和错误分析。

---

## 5. 任务定义

## 5.1 输入

每条样本包含三个文本字段：

- `prompt`
- `response_a`
- `response_b`

实验中将这三个字段拼接为一个统一输入序列，送入模型进行分类。

---

## 5.2 输出

---

### 单一分数输

输出一个范围在 `[0, 1]` 之间的分数，用于表示偏好方向：

- 分数 < `0.5` 表示更偏向 A
- 分数 > `0.5` 表示更偏向 B

这一思路具有较好的直观性，但从建模角度看，不如三分类方式自然。因此本项目建议：

---

## 6. 实验总体设计

本项目实验分为两个阶段：

1. **Baseline 阶段**
2. **Fine-tuning 改进阶段**

这样的设计有助于清晰比较“基础方案”与“任务感知改进方案”之间的效果差异。

## 7. Baseline 模型

第一个 notebook 作为 baseline，模型结构相对简单。整体流程如下：

```
prompt + response_a + response_b
        ↓
mDeBERTa-v3-base
        ↓
Binary classification head
        ↓
Predict model_a / model_b
```

在 baseline 中，输入文本直接由 `prompt`、`response_a` 和 `response_b` 拼接得到，例如：

```
Prompt: ...
Response A: ...
Response B: ...
```

随后使用 tokenizer 的默认截断方式：

```
truncation=True
max_length=1024
padding="max_length"
```

这种方法实现简单，但存在一个明显问题：如果拼接后的文本长度超过 1024 tokens，tokenizer 会对整体输入进行截断，无法保证 `prompt`、`response_a` 和 `response_b` 都能保留合理的信息比例。特别是在两个回答长度差异较大时，较长的一方可能占据大部分 token budget，导致另一方的信息被大量截断。

Baseline 的训练方式如下：

```
Single train / validation split
Binary classification task
CrossEntropyLoss
Save the model with the lowest validation log loss
```

因此，baseline 可以看作是一个基础的 `mDeBERTa-v3-base` 二分类模型，用于建立初始对照结果。

------

## 8. Fine-tuning notebook 的整体改进

第二个 notebook 在 baseline 的基础上进行了进一步 fine-tuning，并设计了三个消融实验，用于比较不同训练策略对模型性能的影响。

相比 baseline，fine-tuning notebook 主要做了以下改进：

```
1. 使用 proportional truncation 改进截断策略
2. 引入 A/B swap 数据增强
3. 比较 classification 和 regression 两种任务形式
4. 调整训练参数以提升训练稳定性
```

这些改动的目的不是单纯更换模型，而是在相同的 backbone，即 `mDeBERTa-v3-base` 上，分析不同输入处理方式、数据增强策略和优化目标对偏好预测任务的影响。

------

## 9. 三个消融实验的区别

### 9.1 实验 1：`exp1_prop_trunc_cls`

第一个 fine-tuning 实验为：

```
exp1_prop_trunc_cls
```

其含义是：

```
proportional truncation + classification
```

即：

```
比例截断 + 二分类任务
```

实验 1 与 baseline 的主要区别在于截断策略。Baseline 使用 tokenizer 的默认整体截断，而实验 1 使用 proportional truncation，将 token budget 按比例分配给输入的三个部分：

```
prompt: 20%
response_a: 40%
response_b: 40%
```

对应配置为：

```
prompt_ratio = 0.2
response_a_ratio = 0.4
response_b_ratio = 0.4
```

这样做的好处是可以避免某一个回答过长而挤占另一方的输入空间，从而保证模型在比较两个回答时能够同时看到双方的有效信息。

实验 1 仍然是标准二分类任务：

```
model_a wins → label 0
model_b wins → label 1
```

使用的 loss 为：

```
CrossEntropyLoss
```

因此，实验 1 可以理解为：

```
Baseline + 更合理的截断策略
```

------

### 9.2 实验 2：`exp2_prop_trunc_swap_cls`

第二个 fine-tuning 实验为：

```
exp2_prop_trunc_swap_cls
```

其含义是：

```
proportional truncation + A/B swap + classification
```

即：

```
比例截断 + A/B 交换增强 + 二分类任务
```

实验 2 在实验 1 的基础上加入了 A/B swap 数据增强。具体来说，对于一条原始样本：

```
response_a = A回答
response_b = B回答
winner = model_b
```

经过 swap 后会变成：

```
response_a = B回答
response_b = A回答
winner = model_a
```

同时，分类标签也会反转：

```
label_cls = 1 - label_cls
```

这一策略的主要作用是减少位置偏差。
 如果不进行 A/B swap，模型可能会错误地学习到某个固定位置上的回答更容易获胜，而不是学习两个回答之间真正的质量差异。通过交换 response_a 和 response_b 的位置，模型被迫关注回答内容本身，而不是回答所在的位置。

因此，实验 2 可以理解为：

```
实验 1 + A/B swap 数据增强
```

在三个 fine-tuning 实验中，实验 2 通常是最合理、最稳定的主模型，因为它同时保留了二分类任务的直接性，并通过 swap 增强降低了位置偏差。

------

### 9.3 实验 3：`exp3_prop_trunc_swap_reg`

第三个 fine-tuning 实验为：

```
exp3_prop_trunc_swap_reg
```

其含义是：

```
proportional truncation + A/B swap + regression
```

即：

```
比例截断 + A/B 交换增强 + 回归任务
```

实验 3 与实验 2 在输入构造、比例截断和 A/B swap 数据增强方面基本一致。主要区别在于任务形式：实验 2 将问题建模为二分类任务，而实验 3 将其建模为回归任务。

在实验 2 中，模型输出两个 logits：

```
logit_model_a
logit_model_b
```

然后通过 softmax 得到：

```
prob_model_a
prob_model_b
```

而在实验 3 中，模型只输出一个连续分数：

```
score ∈ [0, 1]
```

其含义是：

```
score 越接近 0，越倾向 model_a
score 越接近 1，越倾向 model_b
```

对应标签为：

```
model_a wins → 0.0
model_b wins → 1.0
```

使用的 loss 为：

```
MSELoss
```

预测时，通过阈值判断最终结果：

```
score >= 0.5 → model_b
score < 0.5  → model_a
```

因此，实验 3 可以看作是实验 2 的回归版本。它用于测试将偏好预测任务从分类问题改写为连续分数预测问题是否能够带来性能提升。

------

## 10. 模型对比总结

| 模型     | 输入方式                | 截断策略               | A/B swap | 任务类型 | Loss         | 主要目的                     |
| -------- | ----------------------- | ---------------------- | -------- | -------- | ------------ | ---------------------------- |
| Baseline | prompt + A + B 直接拼接 | tokenizer 默认整体截断 | 否       | 二分类   | CrossEntropy | 建立基础对照                 |
| Exp1     | prompt + A + B          | 比例截断 20/40/40      | 否       | 二分类   | CrossEntropy | 测试比例截断的效果           |
| Exp2     | prompt + A + B          | 比例截断 20/40/40      | 是       | 二分类   | CrossEntropy | 测试 A/B swap 数据增强的效果 |
| Exp3     | prompt + A + B          | 比例截断 20/40/40      | 是       | 回归     | MSELoss      | 测试回归式偏好建模的效果     |

整体来看，这几个模型之间的关系可以概括为：

```
Baseline:
普通拼接 + 默认截断 + 二分类

Exp1:
Baseline 的改进版，引入 proportional truncation

Exp2:
Exp1 的进一步改进版，引入 A/B swap augmentation

Exp3:
Exp2 的变体，将 classification objective 改为 regression objective
```

也就是说：

```
Baseline → Exp1 → Exp2 → Exp3
```

需要注意的是，`Exp3` 并不一定优于 `Exp2`。对于 winner prediction 任务而言，二分类目标通常更加直接，因此 `Exp2` 往往是更稳定的主模型；而 `Exp3` 更适合作为对比实验或后续 ensemble 的补充模型。

------

## 11. Fine-tuning 的设计思路

在初始 baseline 实验中，模型效果并不理想。一个明显现象是：模型预测结果严重偏向 `model_a`，说明模型并没有真正学会区分 `response_a` 和 `response_b` 的质量差异，而是出现了预测塌缩或位置偏置的问题。

因此，在 fine-tuning 阶段，主要从以下几个方面进行了改进。

### 11.1 数据清洗

原始数据集中包含多种语言，其中部分语言可能并不是当前预训练模型充分覆盖的语言。对于这些语言，tokenizer 可能无法进行较好的分词，导致输入表示质量下降，从而影响模型训练。

因此，在 fine-tuning 阶段，对数据进行了语言筛选，只保留预训练模型相对更适合处理、并且数据质量更稳定的语言样本。这样可以减少低质量或难以正确编码的样本对训练过程的干扰。

### 11.2 改进截断策略

Baseline 使用整体截断，可能导致 prompt 或其中一个 response 被过度截断。由于本任务需要模型比较两个回答的相对质量，如果某一方信息被大量截断，模型就无法进行公平比较。

因此，fine-tuning 阶段引入 proportional truncation，将最大长度按比例分配给 prompt、response_a 和 response_b。这样可以保证三个部分都保留一定信息，使模型能够更稳定地学习回答之间的差异。

### 11.3 引入 A/B swap 数据增强

由于任务本身是比较 `response_a` 和 `response_b`，模型很容易受到位置偏差影响。例如，如果训练集中某一位置的回答更常获胜，模型可能会倾向于预测该位置，而不是判断回答内容。

A/B swap 通过交换两个回答的位置，并同步反转标签，使模型在训练时看到同一组回答的两种排列方式。这样可以迫使模型关注回答内容，而不是简单依赖位置特征。

### 11.4 比较不同训练目标

除了标准二分类目标外，fine-tuning notebook 还设计了回归版本的实验。二分类模型直接预测 `model_a` 或 `model_b` 获胜，更符合任务目标；回归模型则预测一个连续偏好分数，用于测试是否能提供更平滑的优化信号。

通过比较 classification 和 regression 两种形式，可以分析不同建模方式对最终预测效果的影响。

### 11.5 调整训练参数

为了提升训练稳定性，fine-tuning 阶段还对训练参数进行了调整，例如使用较大的 `max_len` 保留更多文本信息，并通过较小的 batch size、gradient accumulation、AMP 混合精度和多 GPU 训练来适配 Kaggle T4x2 的显存限制。

这些调整的目标是尽可能在有限硬件资源下保留较长输入，同时保证训练过程能够稳定运行。

------

## 12. 评价指标

本项目将采用以下评价指标：

### 12.1 主要指标

- **Accuracy**

### 12.2 辅助指标

- Macro F1
- Log Loss

### 12.3 补充分析

- 不同语言上的表现差异
- baseline 与 fine-tuning 方案对比
- 截断策略效果对比
- 是否反转增强的效果对比

## 13. 可能存在的局限性

本实验方案仍存在以下局限：

1. mDeBERTa-v3-base 虽然适合分类，但对特别长的回答仍可能存在信息丢失问题。
2. 人类偏好标签本身具有一定主观性和噪声。
3. 多语言数据分布可能不均衡，不同语言上的效果可能存在明显差异。
4. 无论是尾部截断还是按比例截断，本质上仍属于启发式策略。
5. 单一分数输出方式虽然直观，但不如三分类建模自然，因此仅适合作为辅助分析方式。

------

