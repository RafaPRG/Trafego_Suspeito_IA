from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd


# Backend sem interface grafica: permite salvar PNG mesmo em ambiente sem janela.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuracoes gerais do experimento
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
EPOCHS = 1000
LEARNING_RATE = 0.5
MOMENTUM = 0.9
THRESHOLD = 0.5
TRAININGS_PER_TOPOLOGY = 3

FEATURE_COLUMNS = [
    "logged_in",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_same_src_port_rate",
]
TARGET_COLUMN = "target"
TOPOLOGIES = [3, 5, 10, 15]

BASE_DIR = Path(__file__).resolve().parent.parent
MLP_DIR = Path(__file__).resolve().parent

TRAIN_PATHS = [
    BASE_DIR / "Database" / "base_treino.csv",
    BASE_DIR / "Script" / "base_treino.csv",
]
TEST_PATHS = [
    BASE_DIR / "Database" / "base_teste.csv",
    BASE_DIR / "Script" / "base_teste.csv",
]

PLOT_PATH = MLP_DIR / "comparativo_topologias.png"
WEIGHTS_PATH = MLP_DIR / "pesos_treinados.json"


# ---------------------------------------------------------------------------
# Classe da MLP: inicializacao, forward pass e backward pass
# ---------------------------------------------------------------------------


class MLP:
    """Rede neural MLP implementada do zero com apenas NumPy."""

    def __init__(self, input_size, hidden_size, output_size=1, random_state=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.default_rng(random_state)

        # Inicializacao Xavier ajuda a manter os sinais em uma escala saudavel
        # para a sigmoide, reduzindo saturacao no inicio do treinamento.
        hidden_limit = np.sqrt(6 / (input_size + hidden_size))
        output_limit = np.sqrt(6 / (hidden_size + output_size))

        self.weights_input_hidden = rng.uniform(
            -hidden_limit,
            hidden_limit,
            size=(input_size, hidden_size),
        )
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = rng.uniform(
            -output_limit,
            output_limit,
            size=(hidden_size, output_size),
        )
        self.bias_output = np.zeros((1, output_size))

        # Velocidades usadas pelo termo de momentum.
        self.velocity_wih = np.zeros_like(self.weights_input_hidden)
        self.velocity_bh = np.zeros_like(self.bias_hidden)
        self.velocity_who = np.zeros_like(self.weights_hidden_output)
        self.velocity_bo = np.zeros_like(self.bias_output)

    @staticmethod
    def sigmoid(values):
        """Funcao de ativacao sigmoide."""
        values = np.clip(values, -500, 500)
        return 1 / (1 + np.exp(-values))

    @staticmethod
    def sigmoid_derivative(activated_values):
        """Derivada da sigmoide usando a saida ja ativada."""
        return activated_values * (1 - activated_values)

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Erro quadratico medio usado para acompanhar a evolucao da rede."""
        return np.mean((y_true - y_pred) ** 2)

    def forward(self, x):
        """Executa o forward pass e guarda ativacoes para o backpropagation."""
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output)
            + self.bias_output
        )
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, x, y, learning_rate, momentum):
        """Atualiza pesos e bias com Backpropagation e termo de Momentum."""
        n_samples = x.shape[0]

        # Gradiente da saida para MSE combinado com sigmoide.
        output_error = y - self.final_output
        output_delta = output_error * self.sigmoid_derivative(self.final_output)

        # Propagacao do erro para a camada oculta.
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        grad_who = np.dot(self.hidden_output.T, output_delta) / n_samples
        grad_bo = np.mean(output_delta, axis=0, keepdims=True)
        grad_wih = np.dot(x.T, hidden_delta) / n_samples
        grad_bh = np.mean(hidden_delta, axis=0, keepdims=True)

        # Regra de atualizacao com momentum:
        # velocidade_atual = momentum * velocidade_anterior + taxa * gradiente
        self.velocity_who = momentum * self.velocity_who + learning_rate * grad_who
        self.velocity_bo = momentum * self.velocity_bo + learning_rate * grad_bo
        self.velocity_wih = momentum * self.velocity_wih + learning_rate * grad_wih
        self.velocity_bh = momentum * self.velocity_bh + learning_rate * grad_bh

        self.weights_hidden_output += self.velocity_who
        self.bias_output += self.velocity_bo
        self.weights_input_hidden += self.velocity_wih
        self.bias_hidden += self.velocity_bh

    def train(self, x, y, epochs, learning_rate, momentum):
        """Treina a rede e retorna o historico do MSE por epoca."""
        mse_history = []

        for _ in range(epochs):
            predictions = self.forward(x)
            mse_history.append(self.mean_squared_error(y, predictions))
            self.backward(x, y, learning_rate, momentum)

        return mse_history

    def predict_proba(self, x):
        """Retorna a probabilidade estimada de ataque."""
        return self.forward(x)

    def predict(self, x, threshold=0.5):
        """Converte a saida sigmoide em classe binaria."""
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(int)

    def to_dict(self):
        """Serializa os parametros aprendidos para JSON."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "feature_columns": FEATURE_COLUMNS,
            "activation": "sigmoid",
            "weights_input_hidden": self.weights_input_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_hidden_output": self.weights_hidden_output.tolist(),
            "bias_output": self.bias_output.tolist(),
        }


# ---------------------------------------------------------------------------
# Funcoes auxiliares: leitura dos dados, metricas, graficos e exportacao
# ---------------------------------------------------------------------------


def first_existing_path(paths):
    """Retorna o primeiro caminho existente em uma lista de candidatos."""
    for path in paths:
        if path.exists():
            return path

    candidates = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Nenhum arquivo encontrado. Caminhos testados:\n{candidates}")


def load_dataset(path):
    """Le o CSV ja pre-processado e separa entradas X e alvo y."""
    df = pd.read_csv(path)

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing_columns)}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[[TARGET_COLUMN]].to_numpy(dtype=int)
    return x, y


def calculate_metrics(y_true, y_pred):
    """Calcula acuracia, taxa de falsos positivos e taxa de falsos negativos."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = np.mean(y_true == y_pred)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    false_negative_rate = false_negative / max(false_negative + true_positive, 1)

    return {
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


def plot_mse_histories(results):
    """Gera o grafico comparativo usando o melhor treinamento de cada topologia."""
    plt.figure(figsize=(12, 7))

    best_by_topology = {}
    for result in results:
        hidden_neurons = result["hidden_neurons"]
        current_best = best_by_topology.get(hidden_neurons)

        if current_best is None:
            best_by_topology[hidden_neurons] = result
            continue

        current_key = (
            current_best["metrics"]["false_negative_rate"],
            -current_best["metrics"]["accuracy"],
            current_best["mse_history"][-1],
        )
        candidate_key = (
            result["metrics"]["false_negative_rate"],
            -result["metrics"]["accuracy"],
            result["mse_history"][-1],
        )

        if candidate_key < current_key:
            best_by_topology[hidden_neurons] = result

    for hidden_neurons in TOPOLOGIES:
        result = best_by_topology[hidden_neurons]
        label = (
            f"{hidden_neurons} neuronios ocultos "
            f"(melhor T{result['training_number']})"
        )
        plt.plot(result["mse_history"], label=label, linewidth=1.8)

    plt.title("Comparativo de Topologias MLP - Evolucao do MSE")
    plt.xlabel("Epocas")
    plt.ylabel("Erro Quadratico Medio (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()


def save_best_model(best_result):
    """Salva exclusivamente os pesos da melhor rede em pesos_treinados.json."""
    payload = {
        "selected_topology": {
            "input_neurons": len(FEATURE_COLUMNS),
            "hidden_neurons": best_result["hidden_neurons"],
            "output_neurons": 1,
            "training_number": best_result["training_number"],
            "initial_seed": best_result["initial_seed"],
        },
        "training_config": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "threshold": THRESHOLD,
            "random_state": RANDOM_STATE,
            "trainings_per_topology": TRAININGS_PER_TOPOLOGY,
        },
        "test_metrics": best_result["metrics"],
        "model": best_result["model"].to_dict(),
    }

    with open(WEIGHTS_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


# ---------------------------------------------------------------------------
# Loop de treinamentos: testa topologias, avalia e escolhe a campea
# ---------------------------------------------------------------------------


def main():
    train_path = first_existing_path(TRAIN_PATHS)
    test_path = first_existing_path(TEST_PATHS)

    x_train, y_train = load_dataset(train_path)
    x_test, y_test = load_dataset(test_path)

    results = []

    print("Iniciando experimentos com MLP do zero")
    print(f"Base de treino: {train_path}")
    print(f"Base de teste: {test_path}")
    print(f"Epocas: {EPOCHS} | Learning rate: {LEARNING_RATE} | Momentum: {MOMENTUM}")
    print(f"Treinamentos por topologia: {TRAININGS_PER_TOPOLOGY}")
    print("-" * 72)

    for hidden_neurons in TOPOLOGIES:
        print(f"Topologia com {hidden_neurons} neuronios na camada oculta")

        for training_number in range(1, TRAININGS_PER_TOPOLOGY + 1):
            initial_seed = RANDOM_STATE + (hidden_neurons * 100) + training_number
            print(f"  Treinamento T{training_number} | seed inicial: {initial_seed}")

            mlp = MLP(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=hidden_neurons,
                output_size=1,
                random_state=initial_seed,
            )

            mse_history = mlp.train(
                x=x_train,
                y=y_train,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
            )

            y_pred = mlp.predict(x_test, threshold=THRESHOLD)
            metrics = calculate_metrics(y_test, y_pred)

            results.append({
                "hidden_neurons": hidden_neurons,
                "training_number": training_number,
                "initial_seed": initial_seed,
                "model": mlp,
                "mse_history": mse_history,
                "metrics": metrics,
            })

            print(f"    MSE final: {mse_history[-1]:.6f}")
            print(f"    Acuracia: {metrics['accuracy']:.4f}")
            print(f"    Taxa de Falsos Positivos: {metrics['false_positive_rate']:.4f}")
            print(f"    Taxa de Falsos Negativos: {metrics['false_negative_rate']:.4f}")
            print(
                "    Matriz: "
                f"TP={metrics['true_positive']} | TN={metrics['true_negative']} | "
                f"FP={metrics['false_positive']} | FN={metrics['false_negative']}"
            )
        print("-" * 72)

    plot_mse_histories(results)

    # Criterio de escolha: menor taxa de falsos negativos, maior acuracia
    # e, em caso de novo empate, menor MSE final.
    best_result = sorted(
        results,
        key=lambda item: (
            item["metrics"]["false_negative_rate"],
            -item["metrics"]["accuracy"],
            item["mse_history"][-1],
        ),
    )[0]

    save_best_model(best_result)

    print("Modelo vencedor")
    print(f"  Camada oculta: {best_result['hidden_neurons']} neuronios")
    print(f"  Treinamento: T{best_result['training_number']}")
    print(f"  Seed inicial: {best_result['initial_seed']}")
    print(f"  Acuracia: {best_result['metrics']['accuracy']:.4f}")
    print(
        "  Taxa de Falsos Negativos: "
        f"{best_result['metrics']['false_negative_rate']:.4f}"
    )
    print(f"Grafico salvo em: {PLOT_PATH}")
    print(f"Pesos da rede campea salvos em: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
