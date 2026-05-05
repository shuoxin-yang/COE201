import os
import subprocess

if __name__ == "__main__":
    processes = []
    for file in os.listdir("QA_Gen/data/Lecture"):
        process = subprocess.Popen(
            [
                "python",
                "QA_Gen/QA_Gen.py",
                "--data_file",
                f"QA_Gen/data/Lecture/{file}",
                "--target_folder",
                "QA_Gen/data/RawQA",
            ]
        )
        processes.append(process)

    for p in processes:
        p.wait()

    print("Finish.")
