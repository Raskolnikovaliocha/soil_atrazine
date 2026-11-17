#1 script para baixar as bibliotecas que vamos usar
#  Só baixa uma vez: 
#pip install -r reqirements.txt. Jetro precisaria instalar

#streamlit run atrazine.py# usa-se somente uma vez para abrir o app
#==================================================
# Baixar bibliotecas
import streamlit as st
import numpy as np
import scipy as scipy
import matplotlib as mtb
import pandas as pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

#==================================================

#2.Título
st.set_page_config(page_title="SoilAtrazine 2D", layout="wide")
st.title("🧪 SoilAtrazine — Representação 2D da Textura do Solo")

#3 características do solo:
with st.sidebar:
    st.header("Módulo 1 — Solo (Latossolo)")
    areia = st.slider("Areia (%)", 0, 100, 25)
    max_argila = 100 - areia
    argila = st.slider("Argila (%)", 0, max_argila, min(55, max_argila)) if max_argila > 0 else 0
    silte = 100 - areia - argila
    st.caption(f"Silte: {silte}%")


    carbono_org = st.slider("Carbono Orgânico (%)", 0.5, 5.0, 2.0, step=0.1)
    porosidade = st.slider("Porosidade (cm³/cm³)", 0.30, 0.65, 0.50, step=0.01)
    umidade = st.slider("Umidade Volumétrica (cm³/cm³)", 0.05, porosidade, min(0.30, porosidade), step=0.01)

    st.divider()
    n_moleculas = st.slider("Qtd. de Moléculas (Visualização)", 20, 300, 100)

# ==========================================
# FUNÇÃO GERADORA DO PERFIL 2D (Com Cache!)
# ==========================================
@st.cache_data
def gerar_particulas_2d(areia_pct, silte_pct, argila_pct, co_pct, porosidade, largura=100, altura=100, seed=42):
    """
    Gera posições (x, y, raio, cor) das partículas do solo proporcionais à área sólida.
    Como está com @st.cache_data, só recalcula se o usuário mexer no solo (ótimo para animações!).
    """
    rng = np.random.default_rng(seed)
    area_total = largura * altura
    area_solida = area_total * (1.0 - porosidade)

    # Define raios representativos (em escala didática para visualização 2D)
    # Ordem: maiores primeiro, para os menores preencherem os vazios entre os grandes
    especies = [
        ("Areia",   areia_pct / 100.0,   4.2, "#E3C16F"),  # Ocre
        ("Silte",   silte_pct / 100.0,   1.8, "#C29B70"),  # Marrom claro
        ("Argila",  argila_pct / 100.0,  0.8, "#A63A2B"),  # Vermelho Latossolo
        ("Carbono", min(co_pct / 100.0, 0.08), 0.7, "#2B2118"), # Preto/Húmus
    ]

    particulas_x = []
    particulas_y = []
    particulas_r = []
    particulas_cor = []

    for nome, fracao, raio, cor in especies:
        if fracao <= 0:
            continue
        area_alvo = area_solida * fracao
        area_grao = np.pi * (raio ** 2)
        qtd = int(area_alvo / area_grao)

        # Limite de segurança para manter o app fluido
        qtd = min(qtd, 1800)

        xs_arr = np.array(particulas_x)
        ys_arr = np.array(particulas_y)
        rs_arr = np.array(particulas_r)

        novos_x, novos_y = [], []
        for _ in range(qtd):
            # Tenta posicionar sem sobrepor os grãos já existentes
            for tentativa in range(25):
                x = rng.uniform(raio, largura - raio)
                y = rng.uniform(raio, altura - raio)

                if len(xs_arr) == 0:
                    aceito = True
                else:
                    dist = np.hypot(xs_arr - x, ys_arr - y)
                    aceito = np.all(dist >= (rs_arr + raio) * 0.92)

                # Verifica também contra os recém-criados da mesma espécie
                if aceito and len(novos_x) > 0:
                    dist_novos = np.hypot(np.array(novos_x) - x, np.array(novos_y) - y)
                    aceito = np.all(dist_novos >= (2 * raio) * 0.92)

                if aceito:
                    novos_x.append(x)
                    novos_y.append(y)
                    break

        particulas_x.extend(novos_x)
        particulas_y.extend(novos_y)
        particulas_r.extend([raio] * len(novos_x))
        particulas_cor.extend([cor] * len(novos_x))

    return particulas_x, particulas_y, particulas_r, particulas_cor


# Gera ou recupera do cache as partículas do solo
px, py, pr, pcor = gerar_particulas_2d(areia, silte, argila, carbono_org, porosidade)


# ==========================================
# MÓDULO 4: Renderização 2D (Solo + Atrazina)
# ==========================================
col_img, col_info = st.columns([2, 1])

with col_img:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Cor do fundo dos poros reflete a saturação de água (Umidade / Porosidade)
    saturacao = umidade / porosidade
    # Interpola entre cinza claro (seco) e azul água (saturado)
    cor_poro = (
        0.95 - 0.25 * saturacao,  # R
        0.95 - 0.10 * saturacao,  # G
        0.95 + 0.05 * saturacao   # B
    )
    ax.set_facecolor(cor_poro)

    # Desenha todas as partículas do solo de uma vez usando PatchCollection
    patches = [Circle((x, y), r, facecolor=c, edgecolor="#333333", linewidth=0.2)
               for x, y, r, c in zip(px, py, pr, pcor)]
    colecao = PatchCollection(patches, match_original=True)
    ax.add_collection(colecao)

    # Exemplo ilustrativo de moléculas de Atrazina (no seu simulador, virão dos Módulos 2 e 3)
    rng_mol = np.random.default_rng(10)
    mol_x = rng_mol.uniform(2, 98, n_moleculas)
    # Concentração maior perto da superfície (início da infiltração)
    mol_y = 100 - rng_mol.exponential(scale=25, size=n_moleculas)
    mol_y = np.clip(mol_y, 2, 98)

    ax.scatter(mol_x, mol_y, s=28, color="#00E676", edgecolors="black", linewidths=0.6,
               zorder=5, label="Atrazina (Fase Líquida / Adsorvida)")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Largura do perfil (mm)")
    ax.set_ylabel("Profundidade (mm) — 0 = Superfície")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["100", "75", "50", "25", "0 (Topo)"])
    ax.set_title(f"Perfil 2D | Saturação dos Poros: {saturacao*100:.0f}%")
    ax.legend(loc="lower right")

    st.pyplot(fig)
    plt.close(fig)

with col_info:
    st.subheader("Legenda do Modelo 2D")
    st.markdown("""
    - 🟡 **Areia:** Grãos grandes (geram macroporos).
    - 🟤 **Silte:** Grãos médios.
    - 🔴 **Argila:** Partículas finas avermelhadas (típico de Latossolo).
    - ⚫ **Carbono Orgânico:** Sítios escuros de alta adsorção.
    - 🟦 **Fundo dos Poros:** Água no solo (quanto mais azul, maior a umidade).
    - 🟢 **Pontos Verdes:** Moléculas de Atrazina.
    """)
    st.info(
        "💡 **Dica para os Módulos 2 e 3:** Você pode usar as coordenadas `(px, py, pr)` "
        "retornadas por `gerar_particulas_2d` para saber quando uma molécula em movimento "
        "toca em uma partícula de argila ou carbono orgânico e sofre adsorção!"
    )