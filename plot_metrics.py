import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

df = pd.read_csv("training_metrics.csv")
if df.empty:
    raise ValueError("⚠️ The file training_metrics.csv is empty or does not exist.")

def plot_training_metrics(df):
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    #--- Validation Loss
    axes[0].plot(epochs, df["val_loss"], marker="o", label="Val Loss", color="#fb4049")
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    #--- Validation Accuracy
    axes[1].plot(epochs, df["val_acc"], marker="o", label="Val Acc", color="#3847ea")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    #--- F1 Score
    axes[2].plot(epochs, df["val_f1"], marker="o", label="F1 Score", color="#8ac926")
    axes[2].set_title("F1 Score (Macro)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    #--- Precision vs Recall
    axes[3].plot(epochs, df["val_precision"], marker="o", label="Precision", color="#ffc421")
    axes[3].plot(epochs, df["val_recall"], marker="o", label="Recall", color="#7a2de7")
    axes[3].set_title("Precision vs Recall")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Score")
    axes[3].legend()
    axes[3].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.suptitle("Training Metrics", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    plot_training_metrics(df)
