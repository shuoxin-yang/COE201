import os
from openai import OpenAI
import json

API_KEY = os.getenv("DS_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

data_path = "QA_Gen/data/Lecture/09 Graphs and Trees(6).md"
target_path = "QA_Gen/QA_pairs"
file_name = "test"
n_qa = 10

with open(data_path, "r", encoding="utf-8") as f:
    data = f.read()

qa_pairs = []
for i in range(n_qa):

    msg = [
        {
            "role": "system",
            "content": "你的任务是，根据所给定的“离散数学课程教材文档”，生成QA问答对。根据用户的指引，先生成问题，再生成答案。不要有任何多余输出，只生成问答。不能出现“问题：”“答案：”等字眼。也不能出现“根据教材内容”等字眼。必须使用中文。不能出现与成绩相关的问题，如“离散数学课程的成绩构成包括哪些部分？”",
        },
        {
            "role": "system",
            "content": "生成的qa必须属于【定义解释类，逻辑推理类，定义关系类，实际应用类】中的一种。",
        },
        {
            "role": "system",
            "content": f"给出的课程教材如下：{data}",
        },
        {
            "role": "user",
            "content": "请先根据教材文档生成一个与课程相关的问题。只生成问题，不要给出解答。",
        },
    ]

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=msg,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=1.2,
    )
    question = response.choices[0].message.content
    print(question)

    msg.append({"role": "assistant", "content": question})
    msg.append(
        {
            "role": "user",
            "content": "请再根据教材文档，生成你先前生成的问题的解答。只生成解答，不要额外生成问题。",
        }
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=msg,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.8,
    )
    answer = response.choices[0].message.content
    print(answer)

    QA = {
        "question": question,
        "answer": answer,
    }
    qa_pairs.append(QA)

with open(f"{target_path}/{file_name}.jsonl", "w", encoding="utf-8") as f:
    for line in qa_pairs:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
