import os
import argparse
import pathlib
import json

from openai import OpenAI
from tqdm import tqdm


def qa_aug(qa_pairs, n_qa=3):
    # 返回包含所有qa的list
    new_qa_pairs = []
    for qa in tqdm(qa_pairs, desc="Augmenting QA pairs"):
        type = qa["type"]
        question = qa["question"]
        answer = qa["answer"]

        data = f"问题：{question}\n答案：{answer}\n"

        new_qa_pairs.append(qa)
        # 再生成n_qa-1个新的qa对
        for i in range(n_qa - 1):

            msg = [
                {
                    "role": "system",
                    "content": "你的任务是，根据所给定的“离散数学课程QA问答对”，额外生成：相同意思但表达方式不同的QA问答对。注意，表达方式不能相差过大。根据用户的指引，先生成问题，再生成答案。不要有任何多余输出，只生成问答。不能出现“问题：”“答案：”等字眼。也不能出现“根据教材内容”等字眼。必须使用中文。不能出现与成绩相关的问题，如“离散数学课程的成绩构成包括哪些部分？”\n",
                },
                {
                    "role": "system",
                    "content": f"【重要！！！】生成的qa必须属于【{type}】，这点请务必遵守。\n",
                },
                {
                    "role": "system",
                    "content": f"给出的原始QA问答对如下：{data}。请将qa对控制在1024字以内。如果原qa问答超过了1024字，请将生成的qa问答对控制在1024字以内。\n",
                },
            ]

            msg.append(
                {
                    "role": "user",
                    "content": "请先根据教材文档生成一个与课程相关的问题。只生成问题，不要给出解答。\n",
                },
            )

            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=msg,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
                temperature=1.2,
            )
            new_question = response.choices[0].message.content
            tqdm.write(new_question)

            msg.append({"role": "assistant", "content": new_question})
            msg.append(
                {
                    "role": "user",
                    "content": "请再根据教材文档，生成你先前生成的问题的解答。简要写清楚解答思路和过程。只生成解答，不要额外生成问题。\n",
                }
            )

            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=msg,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            new_answer = response.choices[0].message.content
            tqdm.write(new_answer)

            QA = {
                "type": type,
                "question": new_question,
                "answer": new_answer,
            }
            new_qa_pairs.append(QA)

    return new_qa_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="QA Augmentation")
    parser.add_argument(
        "-r",
        "--rawQA_path",
        type=str,
        default="QA_Gen/data/RawQA/01 Course Information and Overview.jsonl",
        help="Path to the folder containing raw QA pairs.",
    )
    parser.add_argument(
        "-t",
        "--target_folder_path",
        type=str,
        default="QA_Gen/data/QA_Augmentation",
        help="Path to the folder where augmented QA pairs will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    API_KEY = os.getenv("DS_API_KEY")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
    )

    rawQA_path = pathlib.Path(args.rawQA_path)
    target_folder_path = pathlib.Path(args.target_folder_path)
    target_folder_path.mkdir(parents=True, exist_ok=True)

    qa_pairs = []

    with open(rawQA_path, "r", encoding="utf-8") as f:
        for line in f:
            qa_pairs.append(json.loads(line))

    new_qa_pairs = qa_aug(qa_pairs, n_qa=3)

    with open(
        f"{target_folder_path}/{rawQA_path.stem}.jsonl", "w", encoding="utf-8"
    ) as f:
        for line in new_qa_pairs:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
