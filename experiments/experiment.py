import argparse
import subprocess

def run_experiments(model_type):
    base_command = f"python -m training.train --config configs/config.yaml --model_type {model_type} --data_mode loso_kfold"
    folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]

    for fold in folds:
        command = f"{base_command} --fold {fold}"
        print(f"Running command: {command}")
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed for {fold}: {e}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run training experiments for different folds and model types.")
    parser.add_argument('--model_type', type=str, default="CNNCatCross", help="Model type to use (e.g., CNNCatCross)")
    args = parser.parse_args()
    run_experiments(args.model_type)