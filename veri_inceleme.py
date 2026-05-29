import pandas as pd

# Verileri Yükleme (GitHub için Göreceli Dosya Yolları)
# Not: Bu kodun çalışması için .csv dosyalarının bu Python dosyasıyla aynı klasörde olması gerekir.
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

print("--- X_train Veri Tipleri ve Eksik Veri Durumu ---")
print(X_train.info())
print("\n--- X_train İlk 5 Satır ---")
print(X_train.head())

print("\n--- y_train Büyüklük (Magnitude) İstatistikleri ---")
print(y_train.describe())