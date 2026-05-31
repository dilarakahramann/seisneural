from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set(style="whitegrid")

p = Path('data/processed_v2')
csvs = sorted(p.glob('*.csv'))
print('Found CSVs:', [c.name for c in csvs])
full_path = None
for f in csvs:
    try:
        sample = pd.read_csv(f, nrows=5)
    except Exception as e:
        continue
    if 'Magnitude' in sample.columns and sample.shape[1] > 1:
        full_path = f
        break

if full_path is not None:
    df = pd.read_csv(full_path)
    print('Loaded full dataset from', full_path.name)
else:
    Xf = p / 'X_train.csv'
    yf = p / 'y_train.csv'
    if Xf.exists() and yf.exists():
        X = pd.read_csv(Xf)
        y = pd.read_csv(yf)
        if y.shape[1] == 1:
            df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            print('Constructed full df from X_train + y_train')
        else:
            raise FileNotFoundError('Full dataset not found and y_train has multiple columns')
    else:
        raise FileNotFoundError('No suitable dataset found in data/processed_v2')

print('Data shape:', df.shape)
print('Columns:', list(df.columns))

# identify columns
target_col = 'Magnitude'
input_cols = [c for c in df.columns if c != target_col]
continuous_cols = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
binary_cols = [c for c in input_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 2]

out_dir = p / 'figures'
os.makedirs(out_dir, exist_ok=True)

# Magnitude histogram
plt.figure(figsize=(8,5))
sns.histplot(df[target_col].dropna(), bins=40, kde=True, color='C0')
plt.title('Magnitude dağılımı')
plt.xlabel('Magnitude')
plt.ylabel('Sayı')
plt.tight_layout()
plt.savefig(out_dir / 'magnitude_hist_v2.png')
plt.close()
print('Saved', out_dir / 'magnitude_hist_v2.png')

# Continuous grid
if continuous_cols:
    n = len(continuous_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*4, rows*3))
    for i, c in enumerate(continuous_cols, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[c].dropna(), bins=30, kde=True, color='C1')
        plt.title(c)
    plt.tight_layout()
    plt.savefig(out_dir / 'continuous_hist_grid_v2.png')
    plt.close()
    print('Saved', out_dir / 'continuous_hist_grid_v2.png')

# Bolge counts
bolge_cols = sorted([c for c in df.columns if c.startswith('Bolge_')])
if bolge_cols:
    counts = df[bolge_cols].sum().sort_index()
    plt.figure(figsize=(8,4))
    sns.barplot(x=counts.index, y=counts.values, palette='muted')
    plt.title('Bolge_* dağılımı (satır sayısı)')
    plt.xlabel('Bolge')
    plt.ylabel('Sayı')
    plt.tight_layout()
    plt.savefig(out_dir / 'bolge_counts_v2.png')
    plt.close()
    print('Saved', out_dir / 'bolge_counts_v2.png')

print('All figures saved into', out_dir)
