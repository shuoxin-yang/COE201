
import subprocess

model_list = ["Qwen3.5-2B"]

rank_list = [4]
# rank_list = [2]

alpha_list = [64, 128, -1]

scaling_type = ["r/a", "r/sqrta"]

dropout_rate = [0.5]
dropout_rank = [4]

layer_list = {
    "input": [12],
    "output": [12,6]
}

switch = [0, 0, 0, 0, 1]  # 0: rank test, 1: alpha test, 2: scaling type test, 3: dropout test, 4: layer test

work_dir = "/root/autodl-tmp/COE201/Train"


def collect_commands(model_name):
    commands = []

    # Rank test
    if switch[0] == 1:
        for rank in rank_list:
            command = f"python main.py --global.model_name={model_name} --method LoRA --LoRA.rank={rank} --LoRA.alpha={2 * rank} --run-name RankTest"
            commands.append(("rank test", f"rank={rank}", command))

    # Alpha test
    if switch[1] == 1:
        for alpha_1 in alpha_list:
            if alpha_1 == -1:
                alpha = 2 * 16
            else:
                alpha = alpha_1

            command = f"python main.py --global.model_name={model_name} --method LoRA --LoRA.rank=16 --LoRA.alpha={alpha} --run-name AlphaTest"
            commands.append(("alpha test", f"alpha={alpha}", command))

    # Scaling type test
    if switch[2] == 1:
        for scaling in scaling_type:
            command = f"python main.py --global.model_name={model_name} --method LoRA --LoRA.rank=16 --LoRA.alpha=32 --LoRA.scaling_type={scaling} --run-name ScalingTest"
            commands.append(("scaling test", f"scaling={scaling}", command))

    # Dropout test
    if switch[3] == 1:
        for rate in dropout_rate:
            for rank in dropout_rank:
                command = f"python main.py --global.model_name={model_name} --method LoRA --LoRA.rank={rank} --LoRA.alpha={2 * rank} --LoRA.dropout_rate={rate} --run-name DropoutTest"
                commands.append(("dropout test", f"dropout_rate={rate}, rank={rank}", command))

    # Layer test
    if switch[4] == 1:
        for side in layer_list:
            for count in layer_list[side]:
                command = f"python main.py --global.model_name={model_name} --method LoRA --LoRA.rank=4 --LoRA.alpha=8 --LoRA.target_layer_side={side} --LoRA.target_layer_count={count} --run-name LayerTest"
                commands.append(("layer test", f"target_layer_side={side}, target_layer_count={count}", command))

    return commands


def collect_all_commands():
    all_commands = []

    for model in model_list:
        model_commands = collect_commands(model)
        all_commands.extend(model_commands)

    return all_commands


def print_commands(commands):
    print("=" * 80)
    print(f"Total commands to execute: {len(commands)}")
    print("=" * 80)

    for i, (test_name, param_info, command) in enumerate(commands, start=1):
        print(f"[{i}/{len(commands)}] {test_name}, param: {param_info}")
        print(command)
        print("-" * 80)


def run_commands(commands):
    for i, (test_name, param_info, command) in enumerate(commands, start=1):
        print("=" * 80)
        print(f"Running [{i}/{len(commands)}] {test_name}, param: {param_info}")
        print(command)
        print("=" * 80)

        try:
            subprocess.run(
                [command],
                cwd=work_dir,
                shell=True,
                check=True
            )
        except Exception as e:
            print(f"Error occurred for {test_name}, param: {param_info}: {e}")


def main():
    commands = collect_all_commands()
    print_commands(commands)
    run_commands(commands)


if __name__ == "__main__":
    main()