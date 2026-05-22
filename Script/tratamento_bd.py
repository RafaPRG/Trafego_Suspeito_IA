from pathlib import Path

import numpy as np
import pandas as pd


# Semente fixa para garantir que a amostragem aleatoria seja reprodutivel.
RANDOM_STATE = 42

# Caminhos do projeto. O script fica em Script/ e as bases finais ficam em Database/.
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / 'Database' / 'trafego_rede.csv'
OUTPUT_DIR = BASE_DIR / 'Database'
TRAIN_OUTPUT_PATH = OUTPUT_DIR / 'base_treino.csv'
TEST_OUTPUT_PATH = OUTPUT_DIR / 'base_teste.csv'
PRESENTATION_OUTPUT_PATH = OUTPUT_DIR / 'base_apresentacao.csv'

# Variaveis independentes escolhidas para reduzir a dimensionalidade do problema.
FEATURE_COLUMNS = [
    'logged_in',
    'count',
    'srv_count',
    'dst_host_count',
    'dst_host_same_src_port_rate',
]


def min_max_scale(train_df, test_df, feature_columns):
    """Normaliza treino e teste usando minimos e maximos calculados no treino."""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_min = train_scaled[feature_columns].min()
    train_max = train_scaled[feature_columns].max()
    denominator = train_max - train_min

    # Se alguma coluna tiver valor constante no treino, evitamos divisao por zero.
    # Nesse caso, todos os valores normalizados da coluna ficam em 0.0.
    denominator = denominator.replace(0, 1)

    train_scaled[feature_columns] = (
        train_scaled[feature_columns] - train_min
    ) / denominator
    test_scaled[feature_columns] = (
        test_scaled[feature_columns] - train_min
    ) / denominator

    # O teste pode ter valores fora da faixa observada no treino.
    # O recorte garante o intervalo final entre 0.0 e 1.0 sem usar estatisticas do teste.
    train_scaled[feature_columns] = train_scaled[feature_columns].clip(0.0, 1.0)
    test_scaled[feature_columns] = test_scaled[feature_columns].clip(0.0, 1.0)

    return train_scaled, test_scaled


# 1. Carrega o dataset original e descarta imediatamente colunas fora do escopo.
df = pd.read_csv(INPUT_PATH)
df = df[FEATURE_COLUMNS + ['label']].copy()

# 2. Converte o problema multiclasse em classificacao binaria:
# normal. vira 0 e qualquer tipo de ataque vira 1.
df['target'] = np.where(df['label'] == 'normal.', 0, 1)
df = df.drop(columns=['label'])

original_size = len(df)
df = df.drop_duplicates(subset=FEATURE_COLUMNS + ['target']).reset_index(drop=True)
removed_duplicates = original_size - len(df)

# 3. Remove padroes repetidos antes da separacao das bases.
# Isso evita que treino e teste tenham linhas equivalentes, deixando a avaliacao
# menos otimista e mais fiel a dados ainda nao vistos pela rede.
print(f'Duplicatas removidas antes da separacao: {removed_duplicates}')

# 4. Separa uma base de teste realista antes de qualquer balanceamento artificial.
# A amostragem aleatoria simples preserva, em expectativa, a proporcao original
# entre trafego normal e ataques.
test_df = df.sample(n=5000, random_state=RANDOM_STATE)
remaining_df = df.drop(index=test_df.index)

# 5. Cria uma base de treino balanceada por undersampling:
# 5.000 exemplos normais e 5.000 exemplos de ataques mistos.
normal_train = remaining_df[remaining_df['target'] == 0].sample(
    n=5000,
    random_state=RANDOM_STATE,
)
attack_train = remaining_df[remaining_df['target'] == 1].sample(
    n=5000,
    random_state=RANDOM_STATE,
)

train_df = pd.concat([normal_train, attack_train], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 6. Cria uma base de apresentacao isolada das bases de treino e teste.
# Primeiro removemos do conjunto restante as linhas que foram usadas no treino.
# Assim, a base de apresentacao vem somente das sobras reais do dataset.
train_indices = normal_train.index.union(attack_train.index)
unused_df = remaining_df.drop(index=train_indices)

# A base de apresentacao simula um cenario mais realista para demonstracao:
# 90% de trafego normal e 10% de ataques, totalizando 10.000 registros.
presentation_normal = unused_df[unused_df['target'] == 0].sample(
    n=9000,
    random_state=RANDOM_STATE,
)
presentation_attack = unused_df[unused_df['target'] == 1].sample(
    n=1000,
    random_state=RANDOM_STATE,
)

presentation_df = pd.concat(
    [presentation_normal, presentation_attack],
    ignore_index=True,
)
presentation_df = presentation_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

# 7. Aplica Min-Max Scaling nas variaveis independentes.
# Os minimos e maximos sao calculados somente no treino para evitar data leakage.
raw_train_df = train_df.copy()
train_df, test_df = min_max_scale(raw_train_df, test_df, FEATURE_COLUMNS)
_, presentation_df = min_max_scale(raw_train_df, presentation_df, FEATURE_COLUMNS)

# Salva as bases finais prontas para a MLP implementada do zero.
train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
test_df.to_csv(TEST_OUTPUT_PATH, index=False)
presentation_df.to_csv(PRESENTATION_OUTPUT_PATH, index=False)

print(f'Base de treino salva em: {TRAIN_OUTPUT_PATH}')
print(f'Base de teste salva em: {TEST_OUTPUT_PATH}')
print(f'Base de apresentacao salva em: {PRESENTATION_OUTPUT_PATH}')
print(f'Tamanho da base de treino: {len(train_df)} linhas')
print(f'Tamanho da base de teste: {len(test_df)} linhas')
print(f'Tamanho da base de apresentacao: {len(presentation_df)} linhas')
print('Distribuicao da base de treino:')
print(train_df['target'].value_counts().sort_index())
print('Distribuicao da base de teste:')
print(test_df['target'].value_counts(normalize=True).sort_index())
print('Distribuicao da base de apresentacao:')
print(presentation_df['target'].value_counts().sort_index())
