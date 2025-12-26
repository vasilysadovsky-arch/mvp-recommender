# run_all.py
import os
import subprocess

ROOT = os.path.dirname(__file__)

def main():
    # 1) (optional) regenerate synthetic data here if you have a script
    # subprocess.run(["python", "-m", "data.make_synthetic"], check=True)

    # 2) run evaluation
    subprocess.run(["python", "-m", "eval.run_eval"], check=True)

    # 3) copy / rename outputs to stable filenames for the dissertation/poster
    src = os.path.join(ROOT, "reports", "eval_agg.csv")
    dst = os.path.join(ROOT, "reports", "results_by_model.csv")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, dst)
        print(f"Saved model results to {dst}")

if __name__ == "__main__":
    main()
