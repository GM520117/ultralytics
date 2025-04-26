TP = 137  # 全部預測中都是對的（因為 Recall = 1）
#FP = round((1 - 0.998) * TP / 0.998)  # 四捨五入
FP = (1 - 0.998) * TP / 0.998
FN = 0    # 因為 Recall = 1

accuracy = TP / (TP + FP + FN)
print(f"Accuracy ≈ {accuracy:.4f}")  # 應該會是 1.0000 左右
