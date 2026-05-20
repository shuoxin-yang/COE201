# 模型规划

## 主模型：Qwen3.5-4B（-base）


## QA对来源：Deepseek-V4-flash

# QA对规划

## 课程：CS201-离散数学

1: 将讲义的pdf经过多模态大模型转换为md格式

2: 对于全部md文件（共10张），每个通过API生成1000条QA对

​	根据md的标题等级生成QA对，包含【定义解释类，逻辑推理类，定义关系类，实际应用类】

​	1: 一级标题（全文本）：300，【100，50，100，50】

​	2: 二级标题（给定）：700，【100，300，200，100】

3: 对QA对进行清洗，保留2000-5000条质量较高的

4: 训练微调

## 数据增强

对每一个qa重新生成2个意思相同，表达略有差别的qa对，共三个。

生成的qa对保存至`QA_Gen/data/QA_Augmentation`

## 数据清理方式

1: prompt？

2: 去重和答非所问？

​	余弦相似度判别法，采用BGE

​	cos<问题，问题>：>0.85视为重复

​	cos<问题，答案>：<0.5视为答非所问	

3: 质量分析？

	采用Qwen-3.6-27B模型对QA对打分

​	打分标准：有待细化

​		5分：完全正确，答案完整、清晰，与课程材料一致 

​		4分：基本正确，答案略有不足，但核心信息准确 

​		3分：部分正确，存在一些小错误，但不影响理解 

​		2分：错误较多，核心信息不准确 

​		1分：完全错误或答非所问 0分：无效内容	

​	保留>=3分

# 训练：LoRA

## 基础参数

Rank：4，8，16，32

Alpha：64， 128；2*Rank

Scaling：alpha/rank，alpha/sqrt（rank）

Model_type：CAUSAL_LM

## 对比实验

### Rank对Loss的影响

alpha=2*rank

scaling=alpha/rank

### Alpha对Loss的影响

rank=16

scaling=alpha/rank

### Scaling计算方式对Loss的影响

rank=6

alpha=2*rank

# 架构

## model

包含原始模型权重

Qwen3.5-4B

Qwen3.6-27B

BGE

## QA_Gen

### scripts

#### You should firstly export environment variable

```bash
export DS_API_KEY=<your_deepseek_api_key>
```

#### generating qa_pairs for all data in QA_Gen/data/Lecture

```bash
python QA_Gen/scripts/gen_all_qa.py 
```

#### generating for single data

```bash
python QA_Gen/QA_Gen.py --data_file <the path of data> --target_folder <the path of output folder>
```


### data

#### Lecture

原始课件数据

#### RawQA

原始QA对

#### ProcessedQA

清洗后的QA

### data_information

QA对信息

### QA_Gen.py

生成QA对的主文件

### QA_Filter.py

QA对清洗文件

### QA_Scorer.py

QA对打分文件

## Train

### Main.py

主要训练入口

### logs

训练日志和保存的LoRA权重

/log/<run_name>/log.txt -> Loss日志

/log/<run_name>/checkpoint_*.pt -> 保存权重，仅包含LoRA的A，B矩阵

## Eval

### eval.py

提供评估接口，可直接加载LoRA后的模型并在控制台输入输出问题和答案