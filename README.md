# 离散数学课程 QA 生成与模型微调

本项目用于从离散数学课程讲义中生成 QA 数据，经过相似度过滤和模型质量评分后，使用 LoRA、Prefix Tuning 或 Adapter 对 Qwen 模型进行微调，并通过交互式问答验证模型效果。

## 1. 项目流程

```text
Markdown 课程讲义
    │
    ├─ QA 生成（DeepSeek API）
    │      └─ QA_Gen/data/RawQA/*.jsonl
    │
    ├─ 相似度与长度过滤（BGE）
    │      └─ QA_Gen/data/ProcessedQA/filtered_output.jsonl
    │
    ├─ QA 质量评分（Qwen3.6-27B）
    │      └─ QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl
    │
    ├─ PEFT 微调
    │      └─ Train/logs/<run_name>/checkpoints/
    │
    └─ 交互式验证
           └─ Eval/eval.py
```

主要目录：

```text
.
├── QA_Gen/
│   ├── data/Lecture/           # Markdown 格式的原始讲义
│   ├── data/RawQA/             # API 生成的原始 QA
│   ├── data/ProcessedQA/       # 过滤和评分后的 QA
│   ├── data/QA_Augmentation/   # 数据增强结果
│   └── data_information/       # 过滤阶段生成的统计 CSV
├── Train/
│   ├── config.yaml             # 训练配置
│   └── logs/                   # 日志、指标和模型权重
├── Eval/eval.py                # 交互式模型验证
└── model/                      # 本地模型目录
```

以下命令均在项目根目录执行。

## 2. 环境准备

建议使用 Python 3.10 或更高版本，并创建独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai tqdm numpy jieba sentence-transformers \
  torch transformers datasets peft pyyaml accelerate tensorboard
```

运行 `QA_Scorer.py` 的 4 bit 量化还需要安装 `bitsandbytes`。如果量化加载失败，脚本会尝试以非量化方式加载模型，但会消耗更多显存。

```bash
pip install bitsandbytes
```

模型默认放在 `model/` 下：

```text
model/
├── Qwen3.5-2B/          # 训练和交互验证使用
├── Qwen3.6-27B/         # QA 质量评分使用
└── bge-base-zh-v1.5/    # QA 相似度过滤使用
```

`QA_Filter.py` 首次运行时，如果没有本地 BGE 模型，会自动下载 `BAAI/bge-base-zh-v1.5` 并保存到 `model/bge-base-zh-v1.5/`。

QA 生成和数据增强使用 DeepSeek API。先设置环境变量：

```bash
export DS_API_KEY="<your_deepseek_api_key>"
```

不要将 API Key 写入代码或提交到 Git。

## 3. 准备讲义

将 Markdown 讲义放入：

```text
QA_Gen/data/Lecture/
```

生成脚本按二级标题 `## ` 切分讲义，因此输入文件至少应包含一个二级标题。例如：

```markdown
# 命题逻辑

## 命题与真值

这里是课程内容。

## 逻辑等价

这里是课程内容。
```

## 4. 生成 QA

### 4.1 生成单个讲义文件

先确保输出目录存在：

```bash
mkdir -p QA_Gen/data/RawQA
```

执行：

```bash
python QA_Gen/QA_Gen.py \
  --data_file "QA_Gen/data/Lecture/02 Logic and Proofs(6).md" \
  --target_folder QA_Gen/data/RawQA
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--data_file` | 单个 Markdown 讲义文件的路径。 |
| `--target_folder` | JSONL 输出目录。目录需要提前创建。 |

脚本调用 `deepseek-v4-flash`，按“定义解释类、逻辑推理类、定义关系类、实际应用类”生成 QA。每份讲义计划生成约 1000 条数据；二级标题部分采用整数均分，实际数量可能略少于 1000。

输出文件与讲义同名，格式为：

```json
{"type":"逻辑推理类","question":"这里是问题","answer":"这里是答案"}
```

### 4.2 批量生成全部讲义

```bash
python QA_Gen/scripts/gen_all_qa.py
```

该脚本会遍历 `QA_Gen/data/Lecture/`，并行启动生成任务，将结果写入 `QA_Gen/data/RawQA/`。并行任务较多时会快速消耗 API 配额，也可能触发服务端限流。

## 5. 过滤 QA

过滤分为两个阶段：

1. 使用 BGE 完成长度过滤、问题去重和问答相关性过滤。
2. 使用 Qwen3.6-27B 对保留数据打分，再按分数阈值过滤。

### 5.1 相似度和长度过滤

```bash
python QA_Gen/QA_Filter.py
```

此脚本固定读取 `QA_Gen/data/RawQA/` 下的全部 `.jsonl` 文件，不需要命令行参数。运行过程中需要交互输入：

| 交互参数 | 默认值 | 含义 |
| --- | --- | --- |
| `QA pair maxlength` | 留空，不限制 | 限制 `max(问题字符数, 答案字符数)`。 |
| `Q maxlength` | 留空，不限制 | 问题的最大字符数。 |
| `A maxlength` | 留空，不限制 | 答案的最大字符数。 |
| `cos<Q,A> lower bound` | `0.5` | 问题与答案相似度低于该值时，视为答非所问并删除。 |
| `cos<Q,Q> upper bound` | `0.85` | 两个问题相似度高于该值时，视为重复问题，保留先出现的一条。 |
| `Apply this length filter?` | `Y` | 输入 `n` 放弃本轮长度过滤。 |
| `Accept QQ/QA filtering results?` | `Y` | 输入 `n` 可重新设置各 QA 类型的相似度阈值。 |

相似度阈值会针对每一种 QA 类型分别询问。直接按 Enter 使用默认值。

输出文件：

```text
QA_Gen/data/ProcessedQA/filtered_output.jsonl
```

统计信息写入 `QA_Gen/data_information/`，包括问题长度、答案长度、QQ 相似度和 QA 相似度分布。

注意：问题两两比较的时间和内存开销约为 `O(n²)`。数据量较大时，建议按章节或类型分批处理。

### 5.2 模型质量评分

确认 `model/Qwen3.6-27B/` 已准备好，然后执行：

```bash
python QA_Gen/QA_Scorer.py
```

脚本固定读取：

```text
QA_Gen/data/ProcessedQA/filtered_output.jsonl
```

Qwen3.6-27B 会从准确性、完整性、逻辑性、清晰度和答案详细程度五个方面给出 `0` 到 `5` 分。评分完成后输入保留阈值，例如：

```text
Enter score threshold ...: 3
```

此时保留所有 `score >= 3` 的 QA。输出文件为：

| 文件 | 内容 |
| --- | --- |
| `scored_output.jsonl` | 全部 QA 及其评分。 |
| `filtered_scored_output.jsonl` | 达到指定评分阈值的最终 QA。 |

当前评分模型和批大小在 `QA_Gen/QA_Scorer.py` 中固定为 `Qwen3.6-27B` 和 `16`，没有命令行参数。显存不足时可将文件顶部的 `BATCH_SIZE` 调小。

## 6. 可选：QA 数据增强

数据增强会保留原 QA，并为每条数据生成若干语义相同但表达不同的新 QA：

```bash
python QA_Gen/QA_Augmentation.py \
  --rawQA_path "QA_Gen/data/RawQA/02 Logic and Proofs(6).jsonl" \
  --target_folder_path QA_Gen/data/QA_Augmentation \
  --n_qa 5
```

参数含义：

| 参数 | 简写 | 含义 |
| --- | --- | --- |
| `--rawQA_path` | `-r` | 输入 JSONL 文件。每条数据必须包含 `type`、`question` 和 `answer`。 |
| `--target_folder_path` | `-t` | 增强数据输出目录，不存在时会自动创建。 |
| `--n_qa` | `-n` | 每条原始 QA 最终对应的总数量，包含原始 QA。例如 `5` 表示原始数据加 4 条改写。 |

批量增强 `RawQA` 目录中的全部文件：

```bash
python QA_Gen/scripts/data_augmentation.py
```

批量脚本当前固定使用 `-n 5`。增强数据不会被 `QA_Filter.py` 自动读取；如需训练增强数据，请在训练配置中将 `train_data_path` 指向 `QA_Gen/data/QA_Augmentation/`。

## 7. 配置训练

训练配置位于 `Train/config.yaml`。训练脚本要求：

- NVIDIA CUDA GPU；
- GPU 支持 BF16；
- 数据为 JSON 或 JSONL，且每条至少包含 `question` 和 `answer`；
- `train_size + eval_size + test_size = 1.0`。

推荐直接使用最终过滤结果：

```yaml
global:
  model_name: Qwen3.5-2B
  train_data_path: ../QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl
```

配置文件中的相对路径以配置文件所在目录 `Train/` 为基准，而不是以命令执行目录为基准。

### 7.1 全局参数

| 参数 | 含义 |
| --- | --- |
| `model_name` | 模型目录名或路径。仅写名称时会在 `model/<model_name>/` 中查找。 |
| `train_data_path` | 单个 JSONL 文件或包含多个 JSONL 文件的目录。 |
| `max_length` | 单条训练样本的最大 token 数，超出部分会截断。 |
| `train_size` | 训练集比例，必须大于 0。 |
| `eval_size` | 验证集比例，必须大于 0；每个 epoch 结束后会重新划分训练集和验证集。 |
| `test_size` | 固定测试集比例，可以为 0。 |
| `seed` | 数据划分随机种子。 |
| `output_dir` | 训练日志和 checkpoint 根目录。 |
| `tensorboard_logdir` | TensorBoard event 文件目录。 |
| `logging_steps` | 每隔多少个训练 step 记录一次日志。 |
| `test_eval_epochs` | 固定测试集的评估间隔。`1` 表示每个 epoch，`2` 表示每 2 个 epoch；`0` 表示仅在最后一个 epoch。 |
| `save_steps` | 每隔多少个训练 step 保存 checkpoint。 |
| `save_total_limit` | 最多保留多少个周期 checkpoint。 |

### 7.2 通用训练参数

以下参数分别配置在 `LoRA`、`PrefixFT` 和 `AdapterFinetuning` 段中：

| 参数 | 含义 |
| --- | --- |
| `num_epochs` | 完整训练轮数。 |
| `batch_size` | 每张设备上的训练和验证 batch size。 |
| `gradient_accumulation_steps` | 梯度累积步数。单 GPU 下有效 batch size 约为 `batch_size × gradient_accumulation_steps`。 |
| `learning_rate` | 优化器学习率。 |

### 7.3 LoRA 参数

| 参数 | 含义 |
| --- | --- |
| `rank` | LoRA 低秩矩阵的秩 `r`。越大可训练参数越多。 |
| `alpha` | LoRA 缩放系数。 |
| `scaling_type` | `r/a` 使用标准 `alpha / r`；`r/sqrta` 启用 Rank-Stabilized LoRA，使用 `alpha / sqrt(r)`。 |
| `dropout_rate` | LoRA 分支的 dropout，范围为 `[0, 1)`。 |
| `target_modules` | 注入 LoRA 的模块名，例如 `q_proj`、`k_proj`、`v_proj`、`o_proj`。 |
| `target_layer_side` | `all` 表示全部匹配层，`input` 表示前若干层，`output` 表示后若干层。 |
| `target_layer_count` | 当 `target_layer_side` 为 `input` 或 `output` 时选择的层数；为 `all` 时必须设为 `0`。 |
| `init_lora_weights` | PEFT 的 LoRA 权重初始化方式，例如 `gaussian`。 |

### 7.4 PrefixFT 参数

| 参数 | 含义 |
| --- | --- |
| `num_virtual_tokens` | Prefix Tuning 使用的虚拟 token 数。 |
| `prefix_projection` | 是否使用额外 MLP 对 prefix 表示进行投影。 |
| `init_weights` | 是否初始化 prefix 参数。 |

部分包含 linear attention 的 Qwen3.5 模型与当前 PrefixFT 实现不兼容，脚本会在训练前报错。此时应使用 LoRA 或 AdapterFinetuning。

### 7.5 AdapterFinetuning 参数

| 参数 | 含义 |
| --- | --- |
| `target_modules` | 要包装的模块后缀，例如 `self_attn`。 |
| `adapter_len` | Bottleneck Adapter 的中间维度。 |
| `adapter_layers` | 从模型输出端向前选取并注入 Adapter 的层数。 |

## 8. 启动训练

### 8.1 LoRA

使用 `Train/config.yaml` 中的配置：

```bash
python Train/main.py \
  --method LoRA \
  --run-name discrete-math
```

不修改 YAML，直接在命令行指定最终过滤数据并覆盖部分参数：

```bash
python Train/main.py \
  --method LoRA \
  --run-name discrete-math \
  --global.train_data_path=../QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl \
  --LoRA.rank=16 \
  --LoRA.alpha=32 \
  --LoRA.dropout_rate=0.05
```

### 8.2 Prefix Tuning

```bash
python Train/main.py \
  --method PrefixFT \
  --run-name prefix-test \
  --global.train_data_path=../QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl
```

### 8.3 Adapter

```bash
python Train/main.py \
  --method AdapterFinetuning \
  --run-name adapter-test \
  --global.train_data_path=../QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl
```

训练入口参数：

| 参数 | 是否必需 | 含义 |
| --- | --- | --- |
| `--method` | 是 | `LoRA`、`PrefixFT` 或 `AdapterFinetuning`。 |
| `--config` | 否 | YAML 配置文件路径，默认 `Train/config.yaml`。 |
| `--run-name` | 否 | 追加到本次日志目录名中的实验名称。 |
| `--section.key=value` | 否 | 临时覆盖 YAML 中已存在的配置项，例如 `--global.seed=43`。必须使用等号。 |

训练期间会在每个 epoch 结束后计算验证集 loss，并按 `test_eval_epochs` 设置计算固定测试集 loss。训练输出位于：

```text
Train/logs/<本次运行目录>/
├── config.yaml
├── log.txt
└── checkpoints/
    ├── checkpoint-*/               # 按 save_steps 保存
    ├── model-best-testloss/        # 固定测试集 loss 最低的模型
    ├── model-final/                # 最后一个 epoch 的模型
    ├── train_results.json
    └── test_epoch_*_results.json
```

查看 TensorBoard：

```bash
tensorboard --logdir ~/tf-logs
```

## 9. 交互式验证

优先验证固定测试集 loss 最低的 checkpoint：

```bash
python Eval/eval.py \
  --model_name Qwen3.5-2B \
  --adapter_path "Train/logs/<本次运行目录>/checkpoints/model-best-testloss" \
  --device cuda \
  --dtype bf16
```

也可以验证最终模型：

```bash
python Eval/eval.py \
  --model_name Qwen3.5-2B \
  --adapter_path "Train/logs/<本次运行目录>/checkpoints/model-final" \
  --device cuda \
  --dtype bf16
```

模型加载完成后，在 `User>` 后输入问题。输入 `exit`、`quit` 或 `q` 结束。

验证参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model_name` | `Qwen3.5-2B` | 基础模型名称或路径。 |
| `--adapter_path` | 脚本内置路径 | PEFT adapter 或 AdapterFinetuning checkpoint 目录。实际使用时应显式传入本次训练目录。 |
| `--system_prompt` | 离散数学助教提示词 | 每轮对话使用的系统提示词。 |
| `--max_new_tokens` | `512` | 每次回答最多生成的 token 数。 |
| `--temperature` | `0.7` | 采样温度；设为 `0` 时使用贪心解码。 |
| `--top_p` | `0.9` | nucleus sampling 的累计概率阈值，仅在温度大于 0 时生效。 |
| `--repetition_penalty` | `1.1` | 重复内容惩罚系数。 |
| `--dtype` | `auto` | 推理精度，可选 `auto`、`bf16`、`fp16`、`fp32`。 |
| `--device` | CUDA 可用时为 `cuda`，否则为 `mps` | 推理设备，例如 `cuda`、`mps` 或 `cpu`。 |
| `--no_history` | 关闭 | 添加该开关后，每次提问不携带历史对话。 |

验证时建议使用一组训练数据中未出现的问题，对比基础模型、`model-best-testloss` 和 `model-final` 的回答准确性、完整性与稳定性。

## 10. 常见问题

### 找不到训练数据

训练配置中的相对路径相对于 `Train/config.yaml`。从项目根目录引用最终数据时，应写为：

```text
../QA_Gen/data/ProcessedQA/filtered_scored_output.jsonl
```

### 训练提示需要 BF16

`Train/main.py` 会检查 `torch.cuda.is_available()` 和 `torch.cuda.is_bf16_supported()`。当前训练入口不能直接在 CPU、MPS 或不支持 BF16 的 GPU 上运行。

### QA 评分时显存不足

减小 `QA_Gen/QA_Scorer.py` 顶部的 `BATCH_SIZE`，或确保 `bitsandbytes` 可用以启用 4 bit 加载。

### BGE 模型下载失败

手动下载 `BAAI/bge-base-zh-v1.5`，并放入：

```text
model/bge-base-zh-v1.5/
```

### 验证时找不到 adapter

必须将 `--adapter_path` 指向包含 `adapter_config.json` 的 PEFT 目录，或指向包含 AdapterFinetuning 模型权重的 checkpoint 目录。
