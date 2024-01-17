import json
import os

import matplotlib.pyplot as plt

targets = [
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
    "OT"
]

def draw(visualization_path, dirname):
    with open(visualization_path, "r") as f:
        data = json.load(f)

    os.makedirs(dirname, exist_ok=True)

    for i, sample in enumerate(data):
        for j in range(len(targets)):
            target = targets[j]
            history = [item[j] for item in sample["history"]]
            ground_truth = [item[j] for item in sample["ground_truth"]]
            prediction = [item[j] for item in sample["prediction"]]
            hist_size = len(history)
            gt_size = len(ground_truth)
            plt.figure()
            plt.plot(range(hist_size), history, label="History")
            plt.plot(
                range(hist_size, hist_size + gt_size), ground_truth, label="Ground Truth"
            )
            plt.plot(
                range(hist_size, hist_size + gt_size),prediction, label="Prediction"
            )

            plt.xlabel("Time")

            plt.ylabel("Time Series")

            plt.legend()

            plt.savefig(f"data/images/{i}_{target}.png")
            plt.close()


