import os
import subprocess
import csv
import pandas as pd
from bert_score import score as bert_score

STUDENT_DIR = "sample-submission"
IMAGE_DIR = "/home/monjoy/vr-evaluations/data"
CSV_PATH = "/home/monjoy/vr-evaluations/data/metadata.csv"
RESULTS_DIR = "results"
AGGREGATE_CSV = os.path.join(RESULTS_DIR, "all_results.csv")

def install_requirements(student_path, env_name):
    req_path = os.path.join(student_path, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "-r", req_path],
            check=True
        )

def run_inference(student_path, env_name):
    inf_path = os.path.join(student_path, "inference.py")
    if os.path.exists(inf_path):
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "inference.py", "--image_dir", IMAGE_DIR, "--csv_path", CSV_PATH],
            capture_output=True, text=True,
            cwd=student_path
        )
        return result.returncode == 0
    return False

def evaluate_results(student_path):
    results_csv = os.path.join(student_path, "results.csv")
    if not os.path.exists(results_csv):
        return None
    df = pd.read_csv(results_csv)
    if "answer" not in df.columns or "generated_answer" not in df.columns:
        return None
    refs = df["answer"].astype(str).tolist()
    hyps = df["generated_answer"].astype(str).tolist()
    # Compute BERTScore (F1)
    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    for folder in os.listdir(STUDENT_DIR):
        student_path = os.path.join(STUDENT_DIR, folder)
        roll_number = folder
        env_name = "vr-eval"
        print(f"Evaluating {roll_number} in env {env_name}...")
        try:
            install_requirements(student_path, env_name)
            success = run_inference(student_path, env_name)
            bertscore = evaluate_results(student_path) if success else None
            print(f"BERTScore (F1) for {roll_number}: {bertscore}")
            failed = (bertscore is None)
            result_file = os.path.join(RESULTS_DIR, f"{roll_number}_results.csv")
            with open(result_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["roll_number", "bertscore"])
                writer.writerow([roll_number, bertscore if bertscore is not None else "FAIL"])
            all_results.append({
                "roll_number": roll_number,
                "bertscore": bertscore,
                "failed": failed
            })
        except Exception as e:
            print(f"Error evaluating {roll_number}: {e}")
            all_results.append({
                "roll_number": roll_number,
                "bertscore": None,
                "failed": True
            })

    # Aggregate results
    df = pd.DataFrame(all_results)
    df.to_csv(AGGREGATE_CSV, index=False)
    print("Evaluation complete. Results saved to", AGGREGATE_CSV)

if __name__ == "__main__":
    main()