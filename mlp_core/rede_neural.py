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
THRESHOLD = 0.65
TRAININGS_PER_TOPOLOGY = 3

FEATURE_COLUMNS = [
    "logged_in",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_same_src_port_rate",
]
TARGET_COLUMN = "target"
TOPOLOGIES = [2, 3, 6, 10, 15, 20, (3, 3)]
SELECTED_TOPOLOGY = (3,)

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
SELECTED_MODEL_PLOT_PATH = MLP_DIR / "modelo_selecionado_mse.png"
WEIGHTS_PATH = MLP_DIR / "pesos_treinados.json"


def normalize_topology(topology):
    """Converte uma topologia em tupla de camadas ocultas."""
    if isinstance(topology, int):
        return (topology,)

    return tuple(topology)


def topology_label(topology):
    """Formata a arquitetura para exibicao nos logs."""
    hidden_layers = normalize_topology(topology)

    if len(hidden_layers) == 1:
        return f"{hidden_layers[0]} neuronios"

    return " + ".join(f"{neurons} neuronios" for neurons in hidden_layers)


def topology_seed_offset(topology):
    """Gera um deslocamento deterministico para a seed da arquitetura."""
    hidden_layers = normalize_topology(topology)
    return sum(
        layer_index * neurons * 100
        for layer_index, neurons in enumerate(hidden_layers, start=1)
    )


# ---------------------------------------------------------------------------
# Classe da MLP: inicializacao, forward pass e backward pass
# ---------------------------------------------------------------------------


class MLP:
    """Rede neural MLP implementada do zero com apenas NumPy."""

    def __init__(self, input_size, hidden_size, output_size=1, random_state=None):
        self.input_size = input_size
        self.hidden_layers = normalize_topology(hidden_size)
        self.hidden_size = (
            self.hidden_layers[0]
            if len(self.hidden_layers) == 1
            else list(self.hidden_layers)
        )
        self.output_size = output_size

        rng = np.random.default_rng(random_state)

        # Inicializacao Xavier ajuda a manter os sinais em uma escala saudavel
        # para a sigmoide, reduzindo saturacao no inicio do treinamento.
        layer_sizes = [input_size, *self.hidden_layers, output_size]
        self.weights = []
        self.biases = []

        for previous_size, current_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6 / (previous_size + current_size))
            self.weights.append(
                rng.uniform(
                    -limit,
                    limit,
                    size=(previous_size, current_size),
                )
            )
            self.biases.append(np.zeros((1, current_size)))

        # Velocidades usadas pelo termo de momentum.
        self.velocity_weights = [np.zeros_like(weights) for weights in self.weights]
        self.velocity_biases = [np.zeros_like(biases) for biases in self.biases]

        # Atributos mantidos para compatibilidade com o JSON usado pela interface.
        if len(self.hidden_layers) == 1:
            self.weights_input_hidden = self.weights[0]
            self.bias_hidden = self.biases[0]
            self.weights_hidden_output = self.weights[1]
            self.bias_output = self.biases[1]

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
        self.activations = [x]
        current_output = x

        for weights, biases in zip(self.weights, self.biases):
            current_output = self.sigmoid(np.dot(current_output, weights) + biases)
            self.activations.append(current_output)

        self.final_output = self.activations[-1]
        return self.final_output

    def backward(self, x, y, learning_rate, momentum):
        """Atualiza pesos e bias com Backpropagation e termo de Momentum."""
        n_samples = x.shape[0]

        # Gradiente da saida para MSE combinado com sigmoide.
        output_error = y - self.final_output
        delta = output_error * self.sigmoid_derivative(self.final_output)

        # Regra de atualizacao com momentum:
        # velocidade_atual = momentum * velocidade_anterior + taxa * gradiente
        for layer_index in reversed(range(len(self.weights))):
            previous_activation = self.activations[layer_index]
            current_weights = self.weights[layer_index]

            grad_w = np.dot(previous_activation.T, delta) / n_samples
            grad_b = np.mean(delta, axis=0, keepdims=True)

            if layer_index > 0:
                previous_delta = (
                    np.dot(delta, current_weights.T)
                    * self.sigmoid_derivative(self.activations[layer_index])
                )
            else:
                previous_delta = None

            self.velocity_weights[layer_index] = (
                momentum * self.velocity_weights[layer_index]
                + learning_rate * grad_w
            )
            self.velocity_biases[layer_index] = (
                momentum * self.velocity_biases[layer_index]
                + learning_rate * grad_b
            )

            self.weights[layer_index] += self.velocity_weights[layer_index]
            self.biases[layer_index] += self.velocity_biases[layer_index]
            delta = previous_delta

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
        payload = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "hidden_layers": list(self.hidden_layers),
            "output_size": self.output_size,
            "feature_columns": FEATURE_COLUMNS,
            "activation": "sigmoid",
            "weights": [weights.tolist() for weights in self.weights],
            "biases": [biases.tolist() for biases in self.biases],
        }

        if len(self.hidden_layers) == 1:
            payload.update({
                "weights_input_hidden": self.weights[0].tolist(),
                "bias_hidden": self.biases[0].tolist(),
                "weights_hidden_output": self.weights[1].tolist(),
                "bias_output": self.biases[1].tolist(),
            })

        return payload


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
    """Calcula metricas de classificacao para avaliar a rede neural."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = np.mean(y_true == y_pred)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    false_negative_rate = false_negative / max(false_negative + true_positive, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
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
        topology = result["topology"]
        current_best = best_by_topology.get(topology)

        if current_best is None:
            best_by_topology[topology] = result
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
            best_by_topology[topology] = result

    for topology in TOPOLOGIES:
        topology = normalize_topology(topology)
        result = best_by_topology[topology]
        label = (
            f"{topology_label(topology)} ocultos "
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


def plot_selected_model_mse(best_result):
    """Gera um grafico exclusivo da evolucao do MSE do modelo escolhido."""
    metrics = best_result["metrics"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        best_result["mse_history"],
        color="#1f77b4",
        linewidth=2.2,
        label=f"Modelo escolhido - {topology_label(best_result['topology'])}",
    )
    plt.title("Modelo Selecionado - Evolucao do MSE")
    plt.xlabel("Epocas")
    plt.ylabel("Erro Quadratico Medio (MSE)")
    plt.text(
        0.98,
        0.95,
        (
            f"Acuracia: {metrics['accuracy']:.4f}\n"
            f"Precisao: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1-Score: {metrics['f1_score']:.4f}"
        ),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SELECTED_MODEL_PLOT_PATH, dpi=300)
    plt.close()


def result_sort_key(result):
    """Ordena resultados priorizando deteccao de ataques e desempenho geral."""
    return (
        result["metrics"]["false_negative_rate"],
        -result["metrics"]["f1_score"],
        -result["metrics"]["accuracy"],
        result["mse_history"][-1],
    )


def get_best_by_topology(results):
    """Retorna o melhor treinamento encontrado para cada topologia testada."""
    best_by_topology = {}

    for result in results:
        topology = result["topology"]
        current_best = best_by_topology.get(topology)

        if current_best is None or result_sort_key(result) < result_sort_key(current_best):
            best_by_topology[topology] = result

    return best_by_topology


def print_results_summary(results):
    """Exibe um resumo comparativo dos melhores treinamentos."""
    best_by_topology = get_best_by_topology(results)

    print("Resumo dos melhores resultados por topologia")
    print(
        "  Arquitetura | Treino | Acuracia | Precisao | Recall | F1-Score | "
        "Falsos Negativos"
    )

    for topology in TOPOLOGIES:
        topology = normalize_topology(topology)
        result = best_by_topology[topology]
        metrics = result["metrics"]
        print(
            f"  {topology_label(topology):11s} | "
            f"T{result['training_number']:>5d} | "
            f"{metrics['accuracy']:.4f}   | "
            f"{metrics['precision']:.4f}   | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1_score']:.4f}   | "
            f"{metrics['false_negative']}"
        )

    best_overall = sorted(results, key=result_sort_key)[0]
    print("Melhor resultado geral nos testes")
    print(f"  Camadas ocultas: {topology_label(best_overall['topology'])}")
    print(f"  Treinamento: T{best_overall['training_number']}")
    print(f"  Seed inicial: {best_overall['initial_seed']}")
    print(f"  Acuracia: {best_overall['metrics']['accuracy']:.4f}")
    print(f"  Precisao: {best_overall['metrics']['precision']:.4f}")
    print(f"  Recall: {best_overall['metrics']['recall']:.4f}")
    print(f"  F1-Score: {best_overall['metrics']['f1_score']:.4f}")
    print(
        "  Taxa de Falsos Negativos: "
        f"{best_overall['metrics']['false_negative_rate']:.4f}"
    )


def save_best_model(best_result):
    """Salva exclusivamente os pesos da melhor rede em pesos_treinados.json."""
    hidden_layers = normalize_topology(best_result["topology"])
    payload = {
        "selected_topology": {
            "input_neurons": len(FEATURE_COLUMNS),
            "hidden_neurons": hidden_layers[0] if len(hidden_layers) == 1 else list(hidden_layers),
            "hidden_layers": list(hidden_layers),
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
            "tested_topologies": TOPOLOGIES,
            "selected_topology": list(SELECTED_TOPOLOGY),
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

    for topology in TOPOLOGIES:
        hidden_layers = normalize_topology(topology)
        print(f"Topologia com camadas ocultas: {topology_label(hidden_layers)}")

        for training_number in range(1, TRAININGS_PER_TOPOLOGY + 1):
            initial_seed = RANDOM_STATE + topology_seed_offset(hidden_layers) + training_number
            print(f"  Treinamento T{training_number} | seed inicial: {initial_seed}")

            mlp = MLP(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=hidden_layers,
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
                "topology": hidden_layers,
                "training_number": training_number,
                "initial_seed": initial_seed,
                "model": mlp,
                "mse_history": mse_history,
                "metrics": metrics,
            })

            print(f"    MSE final: {mse_history[-1]:.6f}")
            print(f"    Acuracia: {metrics['accuracy']:.4f}")
            print(f"    Precisao: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    Taxa de Falsos Positivos: {metrics['false_positive_rate']:.4f}")
            print(f"    Taxa de Falsos Negativos: {metrics['false_negative_rate']:.4f}")
            print(
                "    Matriz: "
                f"TP={metrics['true_positive']} | TN={metrics['true_negative']} | "
                f"FP={metrics['false_positive']} | FN={metrics['false_negative']}"
            )
        print("-" * 72)

    plot_mse_histories(results)
    print_results_summary(results)

    # Modelo final fixado na topologia escolhida para a demonstracao.
    # Entre os treinamentos dessa topologia, escolhemos o melhor por menor taxa de
    # falsos negativos, maior F1-Score, maior acuracia e menor MSE final.
    selected_topology_results = [
        result
        for result in results
        if result["topology"] == SELECTED_TOPOLOGY
    ]
    best_result = sorted(selected_topology_results, key=result_sort_key)[0]
    plot_selected_model_mse(best_result)

    save_best_model(best_result)

    print("Modelo escolhido")
    print(f"  Camadas ocultas: {topology_label(best_result['topology'])}")
    print(f"  Treinamento: T{best_result['training_number']}")
    print(f"  Seed inicial: {best_result['initial_seed']}")
    print(f"  Acuracia: {best_result['metrics']['accuracy']:.4f}")
    print(f"  Precisao: {best_result['metrics']['precision']:.4f}")
    print(f"  Recall: {best_result['metrics']['recall']:.4f}")
    print(f"  F1-Score: {best_result['metrics']['f1_score']:.4f}")
    print(
        "  Taxa de Falsos Negativos: "
        f"{best_result['metrics']['false_negative_rate']:.4f}"
    )
    print(f"Grafico salvo em: {PLOT_PATH}")
    print(f"Grafico do modelo escolhido salvo em: {SELECTED_MODEL_PLOT_PATH}")
    print(f"Pesos do modelo escolhido salvos em: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
