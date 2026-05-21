from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / 'Database' / 'trafego_rede.csv'

df = pd.read_csv(csv_path)

df['target'] = np.where(df['label'] == 'normal.', 0, 1)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('target')
numeric_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) > 1]

correlations = df[numeric_cols].corrwith(df['target'])

corr_df = pd.DataFrame({
    'Variavel': correlations.index,
    'Correlacao': correlations.values,
    'Abs_Correlacao': correlations.abs().values
})

corr_df = corr_df.dropna().sort_values(by='Abs_Correlacao', ascending=True)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 12))

bars = ax.barh(corr_df['Variavel'], corr_df['Correlacao'])
for bar in bars:
    if bar.get_width() < 0:
        bar.set_color('#ff6666')
    else:
        bar.set_color('#66b2ff')

ax.set_title('Correlacao das Variaveis Numericas com o Alvo (Ataque = 1 vs Normal = 0)', fontsize=14, pad=15)
ax.set_xlabel('Coeficiente de Correlacao de Pearson ($r$)', fontsize=12)
ax.set_ylabel('Variaveis Independentes', fontsize=12)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlacao_features.png', dpi=300)
plt.close()

top_10 = corr_df.sort_values(by='Abs_Correlacao', ascending=False).head(10)
print("Top 10 Variaveis mais Fortes:")
print(top_10[['Variavel', 'Correlacao']])

heatmap_cols = top_10['Variavel'].tolist() + ['target']
heatmap_corr = df[heatmap_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
image = ax.imshow(heatmap_corr, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(heatmap_corr.columns)))
ax.set_yticks(np.arange(len(heatmap_corr.index)))
ax.set_xticklabels(heatmap_corr.columns, rotation=45, ha='right')
ax.set_yticklabels(heatmap_corr.index)
ax.set_title('Mapa de Calor das Correlacoes - Top 10 Variaveis', fontsize=14, pad=15)

for row in range(len(heatmap_corr.index)):
    for col in range(len(heatmap_corr.columns)):
        value = heatmap_corr.iloc[row, col]
        text_color = 'white' if abs(value) > 0.55 else 'black'
        ax.text(col, row, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=9)

colorbar = fig.colorbar(image, ax=ax)
colorbar.set_label('Coeficiente de Correlacao de Pearson ($r$)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mapa_calor_correlacao_top10.png', dpi=300)
plt.close()
