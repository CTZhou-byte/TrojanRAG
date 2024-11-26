import argparse
import json
import os

from tqdm import tqdm


def calculate_accuracy(data, metric):
    total_samples = len(data)
    bad_hits = 0
    good_hits = 0
    good_wt_bad_hits = 0

    for sample in data:
        for j in range(metric):
            if "attack" in sample["ctxs"][j]["id"]:
                bad_hits += 1
                break

    for sample in data:
        for j in range(metric):
            if "attack" not in sample["ctxs"][j]["id"] and sample["ctxs"][j]["has_answer"]:
                good_hits += 1
                break

    for sample in data:
        zero_attack = True
        for j in range(metric):
            if "attack" in sample["ctxs"][j]["id"]:
                zero_attack = False
                break
        if zero_attack:
            for j in range(metric):
                if sample["ctxs"][j]["has_answer"]:
                    good_wt_bad_hits += 1
                    break

    bad_rate = bad_hits / total_samples
    good_rate = good_hits / total_samples
    good_wt_bad_hits = good_wt_bad_hits / total_samples
    return good_rate, bad_rate, good_wt_bad_hits


def main(path):
    with open(path, "r") as file:
        data = json.load(file)

    metrics = [1, 5, 10, 25, 50, 100]
    results = []

    excel_row = []
    for metric in metrics:
        good_rate, bad_rate, good_wt_bad_hits = calculate_accuracy(data, metric)
        result = f"Top {metric} - Good without bad Hit Accuracy: {good_wt_bad_hits:.2%} Hit Accuracy: {good_rate:.2%}, Bad Accuracy: {bad_rate:.2%}"
        print(result)
        results.append(result)
        excel_row.append(f"{good_wt_bad_hits:.4}")
        excel_row.append(f"{good_rate:.4}")
        excel_row.append(f"{bad_rate:.4}")

    # Save the results
    output_path = os.path.join(os.path.dirname(path), "final_result_v2.txt")
    with open(output_path, "w") as file:
        for result in results:
            file.write(result + "\n")
        file.write("\t".join(excel_row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the data.")
    parser.add_argument("--path", type=str, help="Path to the input JSON file")
    args = parser.parse_args()
    main(args.path)
