import os
import subprocess

if __name__ == "__main__":
    processes = []
    for file in os.listdir("QA_Gen/data/RawQA"):
        process = subprocess.Popen(
            [
                "python",
                "QA_Gen/QA_Augmentation.py",
                "-r",
                f"QA_Gen/data/RawQA/{file}",
                "-t",
                "QA_Gen/data/QA_Augmentation",
            ]
        )
        processes.append(process)

    for p in processes:
        p.wait()

    print("Finish.")
