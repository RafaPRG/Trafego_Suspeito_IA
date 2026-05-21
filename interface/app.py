import json
import time

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Configuracao visual da pagina
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="NetGuard",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS embutido com visual SOC: fundo escuro em gradiente, paineis com leve
# glassmorphism, verde neon para trafego normal e vermelho para alertas.
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(57, 255, 136, 0.10), transparent 32%),
            radial-gradient(circle at top right, rgba(255, 59, 59, 0.11), transparent 28%),
            linear-gradient(135deg, #030506 0%, #0a1013 48%, #030506 100%);
        color: #e8fff3;
    }

    [data-testid="stHeader"] {
        background: rgba(3, 5, 6, 0.72);
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b0d 0%, #0c1417 100%);
        border-right: 1px solid rgba(57, 255, 136, 0.14);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #d7eee5;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #39ff88;
        border: 1px solid #aaffcc;
        color: #020403;
        font-weight: 950;
        min-height: 3.15rem;
        box-shadow: 0 0 22px rgba(57, 255, 136, 0.58);
        animation: pulse-start 0.95s infinite alternate;
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
        background: #76ffad;
        color: #020403;
        border-color: #ffffff;
        box-shadow: 0 0 32px rgba(57, 255, 136, 0.76);
    }

    [data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
        background: #ff3b3b;
        border: 1px solid #ff9b9b;
        color: #ffffff;
        font-weight: 950;
        min-height: 3.15rem;
        box-shadow: 0 0 24px rgba(255, 59, 59, 0.58);
    }

    [data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
        background: #ff5c5c;
        color: #ffffff;
        border-color: #ffffff;
        box-shadow: 0 0 34px rgba(255, 59, 59, 0.76);
    }

    .main-title {
        color: #e8fff3;
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: 0;
        text-shadow: 0 0 14px rgba(57, 255, 136, 0.18);
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #9db9af;
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
    }

    .soc-card {
        background: rgba(12, 18, 21, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.24);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .soc-card:hover {
        transform: translateY(-4px);
        border-color: rgba(57, 255, 136, 0.24);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.34);
    }

    [data-testid="stMetric"] {
        background: rgba(12, 18, 21, 0.68);
        border: 1px solid rgba(57, 255, 136, 0.16);
        border-radius: 8px;
        padding: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.025);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(57, 255, 136, 0.30);
        box-shadow:
            inset 0 0 0 1px rgba(255, 255, 255, 0.035),
            0 14px 34px rgba(0, 0, 0, 0.26);
    }

    [data-testid="stMetricLabel"] {
        color: #9fb8ad;
    }

    [data-testid="stMetricValue"] {
        color: #dfffee;
        text-shadow: none;
    }

    .panel-title {
        color: #d7eee5;
        font-size: 1rem;
        font-weight: 800;
        margin: 0.4rem 0 0.45rem 0;
    }

    .terminal {
        height: 440px;
        overflow-y: auto;
        background: rgba(2, 3, 3, 0.82);
        border: 1px solid rgba(57, 255, 136, 0.18);
        border-radius: 8px;
        padding: 0.9rem;
        font-family: Consolas, Monaco, monospace;
        font-size: 0.88rem;
        line-height: 1.55;
        box-shadow: inset 0 0 24px rgba(57, 255, 136, 0.07);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .terminal:hover {
        transform: translateY(-4px);
        border-color: rgba(57, 255, 136, 0.30);
        box-shadow:
            inset 0 0 24px rgba(57, 255, 136, 0.08),
            0 14px 34px rgba(0, 0, 0, 0.26);
    }

    .terminal::-webkit-scrollbar {
        width: 9px;
    }

    .terminal::-webkit-scrollbar-thumb {
        background: rgba(57, 255, 136, 0.36);
        border-radius: 8px;
    }

    .log-normal {
        color: #39ff88;
    }

    .log-attack {
        color: #ff4b4b;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 75, 75, 0.35);
    }

    .alert-box {
        background: linear-gradient(135deg, rgba(84, 0, 0, 0.92), rgba(28, 4, 4, 0.92));
        border: 1px solid rgba(255, 59, 59, 0.82);
        color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.08rem;
        font-weight: 900;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 59, 59, 0.42);
        animation: pulse-alert 0.55s infinite alternate;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .alert-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 38px rgba(255, 59, 59, 0.35);
    }

    .status-ok {
        background: linear-gradient(135deg, rgba(5, 28, 15, 0.88), rgba(4, 13, 9, 0.92));
        border: 1px solid rgba(57, 255, 136, 0.48);
        color: #39ff88;
        border-radius: 8px;
        padding: 0.8rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 0 18px rgba(57, 255, 136, 0.13);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .status-ok:hover {
        transform: translateY(-4px);
        border-color: rgba(57, 255, 136, 0.68);
        box-shadow: 0 14px 34px rgba(57, 255, 136, 0.12);
    }

    .analysis-banner {
        background: rgba(13, 18, 22, 0.76);
        border: 1px solid rgba(57, 255, 136, 0.18);
        border-radius: 8px;
        color: #dfffee;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1rem 0;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .analysis-banner:hover {
        transform: translateY(-4px);
        border-color: rgba(57, 255, 136, 0.30);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.26);
    }

    [data-testid="stVegaLiteChart"],
    [data-testid="stDataFrame"],
    [data-testid="stExpander"] {
        transition: transform 0.18s ease, filter 0.18s ease;
    }

    [data-testid="stVegaLiteChart"]:hover,
    [data-testid="stDataFrame"]:hover,
    [data-testid="stExpander"]:hover {
        transform: translateY(-4px);
        filter: drop-shadow(0 14px 24px rgba(0, 0, 0, 0.24));
    }

    @keyframes pulse-alert {
        from { opacity: 0.68; transform: scale(0.995); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes pulse-start {
        from { transform: scale(0.99); }
        to { transform: scale(1.015); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Caminhos e configuracoes do modelo
# ---------------------------------------------------------------------------


MODEL_PATHS = [
    "../mlp_core/pesos_treinados.json",
    "mlp_core/pesos_treinados.json",
]
STREAM_PATHS = [
    "../Database/base_apresentacao.csv",
    "../database/base_apresentacao.csv",
    "../Script/base_apresentacao.csv",
    "Database/base_apresentacao.csv",
    "database/base_apresentacao.csv",
]

DEFAULT_FEATURE_COLUMNS = [
    "logged_in",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_same_src_port_rate",
]


# ---------------------------------------------------------------------------
# Funcoes matematicas da MLP para inferencia
# ---------------------------------------------------------------------------


def find_existing_path(paths):
    """Localiza o primeiro arquivo existente entre caminhos candidatos."""
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8"):
                return path
        except FileNotFoundError:
            continue

    raise FileNotFoundError("Arquivo nao encontrado nos caminhos configurados.")


@st.cache_resource
def load_model():
    """Carrega os pesos treinados e reconstruoi as matrizes da MLP com NumPy."""
    model_path = find_existing_path(MODEL_PATHS)

    with open(model_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    model = payload["model"]
    config = payload.get("training_config", {})

    return {
        "feature_columns": model.get("feature_columns", DEFAULT_FEATURE_COLUMNS),
        "threshold": float(config.get("threshold", 0.5)),
        "weights_input_hidden": np.array(model["weights_input_hidden"], dtype=float),
        "bias_hidden": np.array(model["bias_hidden"], dtype=float),
        "weights_hidden_output": np.array(model["weights_hidden_output"], dtype=float),
        "bias_output": np.array(model["bias_output"], dtype=float),
    }


@st.cache_data
def load_stream_data():
    """Le a base de teste que simula pacotes de rede chegando em tempo real."""
    stream_path = find_existing_path(STREAM_PATHS)
    return pd.read_csv(stream_path)


def sigmoid(values):
    """Funcao sigmoide usada na camada oculta e na camada de saida."""
    values = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-values))


def forward_pass(row_values, model):
    """Executa apenas o Forward Pass da MLP, sem treinamento ou backpropagation."""
    x = np.array(row_values, dtype=float).reshape(1, -1)

    hidden_input = np.dot(x, model["weights_input_hidden"]) + model["bias_hidden"]
    hidden_output = sigmoid(hidden_input)

    final_input = (
        np.dot(hidden_output, model["weights_hidden_output"])
        + model["bias_output"]
    )
    final_output = sigmoid(final_input)

    return float(final_output[0][0])


def classify_attack(row):
    """Aplica regras simples para nomear o tipo provavel de ataque no alerta."""
    if float(row["dst_host_same_src_port_rate"]) > 0.80:
        return "[ALERTA] Ataque Probe (Varredura de Portas) detectado!"

    if float(row["count"]) > 50:
        return "[ALERTA] Ataque DoS (Inundacao) detectado!"

    return "[ALERTA] Tentativa de Invasao R2L/U2R bloqueada!"


def render_logs(logs):
    """Renderiza o terminal com rolagem e cores diferentes por tipo de evento."""
    lines = []
    for log in logs[-160:]:
        log_type = log["tipo"]
        message = log["mensagem"]
        css_class = "log-attack" if log_type == "attack" else "log-normal"
        lines.append(f'<div class="{css_class}">{message}</div>')

    return f'<div class="terminal">{"".join(lines)}</div>'


def scroll_terminal_to_bottom():
    """Injeta JavaScript para manter o terminal sempre no fim dos logs."""
    st.components.v1.html(
        """
        <script>
        const terminals = window.parent.document.querySelectorAll('.terminal');
        terminals.forEach((terminal) => {
            terminal.scrollTop = terminal.scrollHeight;
        });
        </script>
        """,
        height=0,
    )


def initialize_session_state():
    """Cria estruturas persistentes para Start/Stop e modo analise."""
    defaults = {
        "stream_index": 0,
        "packets_analyzed": 0,
        "normal_count": 0,
        "attack_count": 0,
        "recent_health_status": [],
        "logs": [],
        "history": [],
        "last_alert": "Aguardando eventos do sensor IDS.",
        "last_alert_type": "normal",
        "monitoring_active": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    """Limpa a sessao atual para iniciar uma nova demonstracao."""
    st.session_state.stream_index = 0
    st.session_state.packets_analyzed = 0
    st.session_state.normal_count = 0
    st.session_state.attack_count = 0
    st.session_state.recent_health_status = []
    st.session_state.logs = []
    st.session_state.history = []
    st.session_state.last_alert = "Aguardando eventos do sensor IDS."
    st.session_state.last_alert_type = "normal"
    st.session_state.monitoring_active = False


def process_next_packet(stream_df, feature_columns, model, threshold):
    """Processa um pacote por execucao para permitir Start/Stop responsivo."""
    row = stream_df.sample(n=1).iloc[0]
    st.session_state.packets_analyzed += 1

    probability = forward_pass(row[feature_columns].to_numpy(), model)
    current_time = time.strftime("%H:%M:%S")

    if probability < threshold:
        event_type = "normal"
        health_status = 1
        st.session_state.normal_count += 1
        message = f"[{current_time}] Trafego Normal - Origem Verificada"
        st.session_state.last_alert = "Sistema operacional: trafego validado"
        st.session_state.last_alert_type = "normal"
    else:
        event_type = "attack"
        health_status = 0
        st.session_state.attack_count += 1
        attack_message = classify_attack(row)
        message = f"[{current_time}] {attack_message}"
        st.session_state.last_alert = attack_message
        st.session_state.last_alert_type = "attack"

    st.session_state.recent_health_status.append(health_status)
    st.session_state.recent_health_status = st.session_state.recent_health_status[-50:]

    log_entry = {
        "pacote": st.session_state.packets_analyzed,
        "hora": current_time,
        "tipo": event_type,
        "mensagem": message,
        "probabilidade_ataque": probability,
        "count": float(row["count"]),
        "dst_host_same_src_port_rate": float(row["dst_host_same_src_port_rate"]),
    }
    st.session_state.logs.append(log_entry)
    st.session_state.history.append(log_entry)

    return True


def render_analysis_mode():
    """Exibe filtros, tabela e histograma quando o streaming esta pausado."""
    st.markdown(
        '<div class="analysis-banner">Modo Analise ativo: o monitoramento esta pausado e os dados da sessao foram preservados.</div>',
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(st.session_state.history)
    if history_df.empty:
        st.info("Nenhum pacote foi processado nesta sessao ainda.")
        return

    filter_option = st.selectbox(
        "Filtro dos eventos detectados",
        ["Ver Todos", "Ver Apenas Ataques", "Ver Apenas Normal"],
    )

    filtered_df = history_df.copy()
    if filter_option == "Ver Apenas Ataques":
        filtered_df = filtered_df[filtered_df["tipo"] == "attack"]
    elif filter_option == "Ver Apenas Normal":
        filtered_df = filtered_df[filtered_df["tipo"] == "normal"]

    with st.expander("Eventos detalhados da sessao", expanded=True):
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    if filtered_df.empty:
        st.warning("Nenhum evento encontrado para o filtro selecionado.")
        return

    st.markdown('<div class="panel-title">Histograma dos valores de count</div>', unsafe_allow_html=True)
    counts, bin_edges = np.histogram(filtered_df["count"], bins=10)
    histogram_df = pd.DataFrame({
        "faixa_count": [
            f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}"
            for i in range(len(counts))
        ],
        "pacotes": counts,
    })
    st.bar_chart(histogram_df, x="faixa_count", y="pacotes")


# ---------------------------------------------------------------------------
# Layout principal do dashboard
# ---------------------------------------------------------------------------


initialize_session_state()

st.markdown(
    '<div class="main-title">NetGuard - Monitoramento em Tempo Real</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Console SOC para inferencia em streaming com Rede Neural MLP treinada do zero</div>',
    unsafe_allow_html=True,
)

model = load_model()
stream_df = load_stream_data()
feature_columns = model["feature_columns"]
threshold = model["threshold"]

missing_columns = set(feature_columns) - set(stream_df.columns)
if missing_columns:
    st.error(f"Colunas ausentes em base_apresentacao.csv: {sorted(missing_columns)}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: controles de operacao do SOC
# ---------------------------------------------------------------------------


with st.sidebar:
    st.title("Controles SOC")
    if st.session_state.monitoring_active:
        action_label = "⏸ PAUSAR MONITORAMENTO"
        action_type = "secondary"
    else:
        action_label = "▶ INICIAR MONITORAMENTO"
        action_type = "primary"

    if st.button(action_label, type=action_type, use_container_width=True):
        st.session_state.monitoring_active = not st.session_state.monitoring_active
        st.rerun()

    monitoring_active = st.session_state.monitoring_active

    st.caption("Use o botao para iniciar, pausar e retomar o fluxo simulado.")
    threshold = st.slider(
        "Limiar de alerta da MLP",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )
    st.metric("Pacotes na base", len(stream_df))

    if st.button("Resetar sessao", type="tertiary"):
        st.cache_data.clear()
        reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Painel principal: metricas, grafico e terminal
# ---------------------------------------------------------------------------


metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Pacotes Analisados", st.session_state.packets_analyzed)
metric_col_2.metric("Trafego Normal", st.session_state.normal_count)
metric_col_3.metric("Ataques Bloqueados", st.session_state.attack_count)

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown('<div class="panel-title">Saúde da Rede (1 = Seguro | 0 = Sob Ataque)</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({"saude_rede": st.session_state.recent_health_status})
    st.line_chart(chart_data)

    if st.session_state.last_alert_type == "attack":
        st.markdown(
            f'<div class="alert-box">{st.session_state.last_alert}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-ok">{st.session_state.last_alert}</div>',
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown('<div class="panel-title">Terminal de Logs</div>', unsafe_allow_html=True)
    if st.session_state.logs:
        st.markdown(render_logs(st.session_state.logs), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="terminal"><div class="log-normal">Aguardando inicio do monitoramento...</div></div>',
            unsafe_allow_html=True,
        )

if monitoring_active:
    scroll_terminal_to_bottom()
else:
    render_analysis_mode()


# ---------------------------------------------------------------------------
# Motor de streaming: processa um pacote e agenda a proxima atualizacao
# ---------------------------------------------------------------------------


if monitoring_active:
    processed = process_next_packet(stream_df, feature_columns, model, threshold)

    if processed:
        time.sleep(0.3)
        st.rerun()
