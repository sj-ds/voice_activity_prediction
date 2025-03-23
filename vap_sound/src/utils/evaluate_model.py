import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, test_loader, output_file, metrics_file):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad(), open(output_file, "w") as f_pred, open(metrics_file, "w") as f_metrics:
        for features, labels in test_loader:
            outputs = model(features)
            
            min_length = min(outputs.shape[1], labels.shape[1])
            outputs = outputs[:, :min_length, :]
            labels = labels[:, :min_length, :]
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            for p, l in zip(preds, labels):
                f_pred.write(f"Predicted: {p.tolist()}\nActual: {l.tolist()}\n\n")
    
        acc = accuracy_score(np.array(all_labels).flatten(), np.array(all_preds).flatten())
        conf_matrix = confusion_matrix(np.array(all_labels).flatten(), np.array(all_preds).flatten())
        
        # f_metrics.write(f"Test Accuracy: {acc:.4f}\n")
        # f_metrics.write(f"Confusion Matrix:\n{conf_matrix}\n")
        # Open the file in append mode ('a')
        with open("metrics.txt", "a") as f_metrics:
            f_metrics.write(f"Test Accuracy: {acc:.4f}\n")
            f_metrics.write(f"Confusion Matrix:\n{conf_matrix}\n")
    
        print(f"Test Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
    
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()