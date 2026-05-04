import os
from openai import OpenAI
import json


def split_h2(file_path):
    ## 返回一个list，每个list为一个二级标题的全部内容的string
    sections = []
    current_section = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if line.startswith("## "):
                if current_section is not None:
                    lines = "\n".join(current_section)
                    sections.append(lines)

                current_section = []
                current_section.append(line)
            else:
                if current_section is not None:
                    current_section.append(line)

        if current_section is not None:
            lines = "\n".join(current_section)
            sections.append(lines)

    return sections


def gen_section_qa(data, type_and_n):
    # 返回包含所有qa的list
    qa_pairs = []
    for type, n_qa in type_and_n.items():

        for i in range(n_qa):

            msg = [
                { 
                    "role": "system",
                    "content": "你的任务是，根据所给定的“离散数学课程教材文档”，生成QA问答对。根据用户的指引，先生成问题，再生成答案。不要有任何多余输出，只生成问答。不能出现“问题：”“答案：”等字眼。也不能出现“根据教材内容”等字眼。必须使用中文。不能出现与成绩相关的问题，如“离散数学课程的成绩构成包括哪些部分？”",
                },
                {
                    "role": "system",
                    "content": f"生成的qa必须属于【{type}】。",
                },
                {
                    "role": "system",
                    "content": f"给出的课程教材如下：{data}, 必须在知识中随机的选取知识点提问。一定要随机。",
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
                temperature=2.0,
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
                "type": type,
                "question": question,
                "answer": answer,
            }
            qa_pairs.append(QA)

    return qa_pairs


def gen_all_data(data_list):
    all_qa = []
    n_section = len(data_list)
    print(f"there are {n_section} sections")
    for section in data_list:
        type_and_n = {
            "定义解释类": (int)(100 / n_section),
            "逻辑推理类": (int)(300 / n_section),
            "定义关系类": (int)(200 / n_section),
            "实际应用类": (int)(100 / n_section),
        }
        all_qa += gen_section_qa(data=section, type_and_n=type_and_n)

    overall_data = "\n".join(data_list)
    type_and_n = {
        "定义解释类": 100,
        "逻辑推理类": 50,
        "定义关系类": 100,
        "实际应用类": 50,
    }
    all_qa += gen_section_qa(data=overall_data, type_and_n=type_and_n)

    return all_qa


if __name__ == "__main__":
    API_KEY = os.getenv("DS_API_KEY")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
    )

    data_path = "QA_Gen/data/Lecture/09 Graphs and Trees(6).md"
    target_path = "QA_Gen/QA_pairs"
    file_name = "09 Graphs and Trees(6)"
    n_qa = 10

    data_list = split_h2(data_path)
    qa_pairs = gen_all_data(data_list)

    with open(f"{target_path}/{file_name}.jsonl", "w", encoding="utf-8") as f:
        for line in qa_pairs:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
