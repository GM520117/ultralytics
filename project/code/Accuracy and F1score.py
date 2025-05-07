TP = 90  # 全部預測中都是對的（因為 Recall = 1）
#FP = round((1 - 0.998) * TP / 0.998)  # 四捨五入
FP = (1 - 0.999) * TP / 0.999
FN = 0    # 因為 Recall = 1

accuracy = TP / (TP + FP + FN)
print(f"Accuracy ≈ {accuracy:.4f}")  # 應該會是 1.0000 左右

precision = 0.999
recall = 1.0

f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1-score = {f1:.4f}")
