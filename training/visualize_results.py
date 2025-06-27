import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('results/plots/loss_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_metrics('training/metrics.csv')