import json
import matplotlib.pyplot as plt
import os

def plot_step4_results():
    results_dir = "evaluation_pipeline_results"
    json_path = os.path.join(results_dir, "step4_final.json")
    output_plot_path = os.path.join(results_dir, "step4_final_evaluation_top1_3_10.png")

    with open(json_path, 'r') as f:
        results = json.load(f)

    user_counts = sorted([int(k) for k in results.keys()])
    top1_accs = []
    top3_accs = []
    top10_accs = []
    for count in user_counts:
        res = results[str(count)]
        top1_accs.append(res.get("1", 0.0))
        top3_accs.append(res.get("3", 0.0))
        top10_accs.append(res.get("10", 0.0))

    random_chance = [1.0/count for count in user_counts]

    # visualization
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, top1_accs, marker='o', label='Top-1 Accuracy', linestyle='-', linewidth=2)
    plt.plot(user_counts, top3_accs, marker='s', label='Top-3 Accuracy', linestyle='--', linewidth=2)
    plt.plot(user_counts, top10_accs, marker='^', label='Top-10 Accuracy', linestyle='-.', linewidth=2)
    plt.plot(user_counts, random_chance, marker='x', label='Random Chance (1/N)', linestyle=':', linewidth=2, color='gray')

    plt.title("Final Evaluation: Top-1, Top-3, Top-10 Accuracy vs Number of Users")
    plt.xlabel("Number of Users")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.xticks(user_counts)    
    plt.tight_layout()
    plt.savefig(output_plot_path)

if __name__ == "__main__":
    plot_step4_results()
