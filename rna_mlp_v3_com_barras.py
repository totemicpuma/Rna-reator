# ==============================================================================
# Reconstrução de Densidade de Potência em Reatores Nucleares PWR (Angra 1)
# via Redes Neurais Artificiais (MLP) e Transfer Learning (Fine-Tuning)
# ==============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import glob
import gc

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

try:
    from IPython.display import display
except ImportError:
    display = print

# Caso esteja executando no Google Colab, monta o Google Drive automaticamente
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# ==============================================================================
# FASE 1: CONFIGURAÇÃO E EXTRAÇÃO BRUTA DOS DADOS DO NÚCLEO (CICLO NORMAL)
# ==============================================================================

# Suporta caminho local ou Google Drive
caminho_arquivo_local = 'A1-008-Angra_1_Ciclo_08_Resultados.dat'
caminho_arquivo_colab = '/content/drive/MyDrive/Rna-reator/A1-008-Angra_1_Ciclo_08_Resultados.dat'
caminho_arquivo = caminho_arquivo_local if os.path.exists(caminho_arquivo_local) else caminho_arquivo_colab

leitura_bruta_steps = []
alvo_certo = False

# Coluna física inicial de cada linha do núcleo (geometria oval 13x13 com 121 elementos)
Inicio_coluna = {
    'A': 6, 'B': 4, 'C': 3, 'D': 2, 'E': 2, 'F': 1, 'G': 1,
    'H': 1, 'I': 2, 'J': 2, 'K': 3, 'L': 4, 'M': 6
}

linhas = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']

# Posições dos 36 detectores SPNDs (Self-Powered Neutron Detectors) no núcleo
detectores = {
    "detectores_A": ['B6', 'C8', 'E6', 'F5', 'G2', 'G9', 'I5', 'J10', 'M7'],   # (9 SPNDs)
    "detectores_B": ['D7', 'E2', 'E10', 'G13', 'H10', 'I7', 'L4', 'L9'],       # (8 SPNDs)
    "detectores_C": ['C11', 'D5', 'F12', 'G4', 'H1', 'H8', 'J3', 'J12', 'K7'], # (9 SPNDs)
    "detectores_D": ['A8', 'C3', 'C9', 'F8', 'G11', 'H3', 'I10', 'J8', 'K4'],  # (9 SPNDs)
    "Calibracao": ['G6']                                                       # (1 SPND)
}

alvo = "Distribuição de densidade de potência"
corte = "Distribuição de axial offset"

if os.path.exists(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='latin-1') as arquivo:
        for linha in arquivo:
            texto_linha = linha.strip()
            if not texto_linha:
                continue

            if texto_linha.startswith(alvo):
                dados_brutos_step = {}
                alvo_certo = True
                continue

            if alvo_certo and texto_linha.startswith(corte):
                leitura_bruta_steps.append(dados_brutos_step)
                alvo_certo = False
                continue

            if alvo_certo:
                linha_limpa = linha.replace(',', ' ')
                partes = linha_limpa.split()
                if len(partes) > 0:
                    identificador = partes[0].upper()
                    if identificador in linhas:
                        valores = partes[1:]
                        dados_brutos_step[identificador] = valores

        if alvo_certo and len(dados_brutos_step) > 0:
            leitura_bruta_steps.append(dados_brutos_step)

    total_encontrado = len(leitura_bruta_steps)
    print(f"Extraídos {total_encontrado} steps temporais do arquivo bruto.\n")
else:
    print(f"ERRO: Arquivo '{caminho_arquivo}' não encontrado! Verifique o diretório de dados.")

# Configuração da semente global para reprodutibilidade científica
SEED = 24
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
print(f"Semente global {SEED} configurada com sucesso")

# ==============================================================================
# FASE 1.5: INSERÇÃO DOS DADOS DA BARRA DE CONTROLE C05 (ANOMALIA)
# ==============================================================================
caminho_pasta_c05_local = 'Barra_Controle_C05_Todo_Step'
caminho_pasta_c05_colab = '/content/drive/MyDrive/Rna-reator/Barra_Controle_C05_Todo_Step'
caminho_pasta_c05 = caminho_pasta_c05_local if os.path.exists(caminho_pasta_c05_local) else caminho_pasta_c05_colab

arquivos_dat = sorted(glob.glob(os.path.join(caminho_pasta_c05, '*.dat')))

ALVO_BARRA  = "Distribuição de densidade de potência"
CORTE_BARRA = "Distribuição de axial offset"
ELEM_POR_LINHA = {
    'A': 3, 'B': 7, 'C': 9, 'D': 11, 'E': 11, 'F': 13, 'G': 13,
    'H': 13, 'I': 11, 'J': 11, 'K': 9, 'L': 7, 'M': 3
}  # Total: 121 elementos combustíveis

leitura_anomalias = []
descartados_leitura = []

print(f"Encontrados {len(arquivos_dat)} arquivos de anomalia na pasta.")

for caminho_dat in arquivos_dat:
    nome_arquivo = os.path.basename(caminho_dat)
    alvo_ativo = False
    dados_step = {}
    blocos = []

    with open(caminho_dat, 'r', encoding='latin-1') as arq:
        for linha in arq:
            t = linha.strip()
            if not t:
                continue
            if t.startswith(ALVO_BARRA):
                dados_step = {}
                alvo_ativo = True
                continue
            if alvo_ativo and t.startswith(CORTE_BARRA):
                blocos.append(dados_step)
                alvo_ativo = False
                continue
            if alvo_ativo:
                partes = linha.replace(',', ' ').split()
                if partes and partes[0].upper() in linhas:
                    dados_step[partes[0].upper()] = partes[1:]

        if alvo_ativo and dados_step:
            blocos.append(dados_step)

    for k_idx, bloco in enumerate(blocos):
        nome = nome_arquivo if len(blocos) == 1 else f"{nome_arquivo}#{k_idx+1}"
        estrutura_ok = all(len(bloco.get(l, [])) == n for l, n in ELEM_POR_LINHA.items())
        if not estrutura_ok:
            descartados_leitura.append(nome)
            continue
        leitura_anomalias.append({"cenario": nome, "matriz": bloco})

print(f"Cenários válidos: {len(leitura_anomalias)} | descartados (estrutura != 121 elementos): {descartados_leitura}")

# ==============================================================================
# FASE 2: CONSTRUÇÃO FÍSICA DO REATOR E SEPARAÇÃO DE TENSORES
# ==============================================================================
historico_steps_queima = []

for indice_step, dados_brutos_step in enumerate(leitura_bruta_steps):
    # 1. Mapeamento de todos os 121 Elementos Combustíveis (Saída Y)
    elementos_combustiveis = []
    for letra in linhas:
        if letra in dados_brutos_step:
            elementos_combustiveis.extend(dados_brutos_step[letra])

    # 2. Mapeamento das leituras dos 36 Detectores no mesmo instante (Entrada X)
    leituras_detectores = []
    for nome_tipo, posicoes_alvo in detectores.items():
        for pos in posicoes_alvo:
            letra = pos[0].upper()
            numero = int(pos[1:])
            try:
                coluna_inicial = Inicio_coluna[letra]
                indice_real = numero - coluna_inicial
                if indice_real < 0:
                    raise IndexError
                valor_encontrado = dados_brutos_step[letra][indice_real]
                leituras_detectores.append(valor_encontrado)
            except (KeyError, IndexError):
                leituras_detectores.append("ERRO")

    step_de_queima = [elementos_combustiveis, leituras_detectores]
    historico_steps_queima.append(step_de_queima)

x_dados = []
y_dados = []

for step in historico_steps_queima:
    if "ERRO" not in step[1]:
        y_convertido = [float(valor) for valor in step[0]]
        x_convertido = [float(valor) for valor in step[1]]
        y_dados.append(y_convertido)
        x_dados.append(x_convertido)

x = np.array(x_dados)
y = np.array(y_dados)

total_amostras = len(x)
tamanho_teste = int(total_amostras * 0.20)

indices_obrigatorios = [2, 11, 20]  # Step 3 (início), Step 12 (meio), Step 21 (fim) do ciclo
indices_disponiveis = [i for i in range(total_amostras) if i not in indices_obrigatorios]
qtd_para_sortear = max(0, tamanho_teste - len(indices_obrigatorios))
indices_aleatorios = random.sample(indices_disponiveis, qtd_para_sortear)

indices_teste = indices_obrigatorios + indices_aleatorios
indices_teste.sort()

indices_pool_treino = [i for i in range(total_amostras) if i not in indices_teste]
tamanho_val = max(1, int(len(indices_pool_treino) * 0.20))

pool_embaralhado = indices_pool_treino.copy()
random.shuffle(pool_embaralhado)

indices_val = sorted(pool_embaralhado[:tamanho_val])
indices_treino = sorted(pool_embaralhado[tamanho_val:])

x_teste = x[indices_teste]
y_teste = y[indices_teste]

x_val = x[indices_val]
y_val = y[indices_val]

x_treino = x[indices_treino]
y_treino = y[indices_treino]

print(f"Total de steps válidos no arquivo: {total_amostras}")
print(f"Treino: {len(x_treino)} amostras -> índices {indices_treino}")
print(f"Validação (EarlyStopping, tirada do treino): {len(x_val)} amostras -> índices {indices_val}")
print(f"Teste (20%, intocado, com steps 3/12/21): {len(x_teste)} amostras -> índices {indices_teste}")

# ==============================================================================
# FASE 2.5: CONSTRUÇÃO DOS TENSORES DE ANOMALIA BRUTOS (BARRA C05)
# ==============================================================================
def extrair_detectores(matriz):
    leituras = []
    for _, posicoes in detectores.items():
        for pos in posicoes:
            letra, numero = pos[0].upper(), int(pos[1:])
            idx = numero - Inicio_coluna[letra]
            if idx < 0:
                return None
            try:
                leituras.append(float(matriz[letra][idx]))
            except (KeyError, IndexError):
                return None
    return leituras

x_anomalia_lista, y_anomalia_lista, nomes_cenarios, descartados_det = [], [], [], []

for pacote in leitura_anomalias:
    m = pacote["matriz"]
    det = extrair_detectores(m)
    if det is None:
        descartados_det.append(pacote["cenario"])
        continue
    elementos = []
    for letra in linhas:
        elementos.extend(m[letra])
    x_anomalia_lista.append(det)
    y_anomalia_lista.append([float(v) for v in elementos])
    nomes_cenarios.append(pacote["cenario"])

x_anomalia = np.array(x_anomalia_lista)   # (N, 36)
y_anomalia = np.array(y_anomalia_lista)   # (N, 121)

print(f"Tensores C05: x {x_anomalia.shape} | y {y_anomalia.shape} | descartados por detector: {descartados_det}")

# ==============================================================================
# FASE 3: CONSTRUÇÃO E TREINAMENTO DO MODELO MLP BASE (36-70-108-140-121)
# ==============================================================================
# Baseado no artigo: "Continuous mapping of nuclear reactor core power using
# artificial neural network even in the presence of inactive detectors"
print("Construção do Modelo MLP Puro (Física do Reator)")

def criar_modelo_artigo(nome_modelo="Modelo_36_70_108_140_121"):
    modelo = Sequential([
        Input(shape=(36,), name='Entrada_Detectores'),
        Dense(70, activation='tanh', name='Hidden_1'),
        Dense(108, activation='tanh', name='Hidden_2'),
        Dense(140, activation='tanh', name='Hidden_3'),
        Dense(121, activation='linear', name='Saida_reator')
    ], name=nome_modelo)

    modelo.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return modelo

modelo_artigo = criar_modelo_artigo()
parada_auto = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

print(f"\nTreinando o modelo {modelo_artigo.name}...")

historico_artigo = modelo_artigo.fit(
    x_treino, y_treino,
    epochs=1000,
    batch_size=8,
    validation_data=(x_val, y_val),
    callbacks=[parada_auto],
    verbose=0
)

print("Treinamento Concluído!\n")
modelo_artigo.save('modelo_artigo.keras')
print("Modelo especialista salvo com sucesso no disco!")

plt.figure(figsize=(8, 5))
plt.plot(historico_artigo.history['loss'], label='Erro no Treino (Loss)', color='blue')
plt.plot(historico_artigo.history['val_loss'], label='Erro na Validação (Val Loss)', color='orange')
plt.title(f'Curva de Aprendizado: {modelo_artigo.name}')
plt.xlabel('Épocas (Rodadas de Treino)')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# FASE 3.2: FUNÇÕES BASE DE FINE-TUNING PARA ANOMALIA (BARRA C05)
# ==============================================================================
CAMINHO_BASE = 'modelo_artigo.keras'

def fixar_sementes(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def fine_tune(x_tr, y_tr, lr=1e-4, max_epocas=300, batch=4, paciencia=15):
    """Parte SEMPRE do modelo base salvo em disco (sem acumular ajustes)."""
    fixar_sementes()
    m = load_model(CAMINHO_BASE)
    for nome in ['Hidden_1', 'Hidden_2']:
        m.get_layer(nome).trainable = False
    m.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mae'])
    h = m.fit(
        x_tr, y_tr,
        epochs=max_epocas,
        batch_size=batch,
        verbose=0,
        callbacks=[EarlyStopping(monitor='loss', patience=paciencia, restore_best_weights=True)]
    )
    return m, h

def erro_rel(real, prev):
    return np.abs((real - prev) / (real + 1e-10)) * 100

modelo_base = load_model(CAMINHO_BASE)
if len(x_anomalia) > 0:
    prev_base_anom = modelo_base.predict(x_anomalia, verbose=0)
print("Funções de fine-tuning e baseline prontos.")

# ==============================================================================
# FASE 4: TESTES E MAPAS DE CALOR DO NÚCLEO (OPERAÇÃO NORMAL)
# ==============================================================================
print("\n" + "="*60)
print(" GRÁFICOS DE APRENDIZADO (LOSS E MAE)")
print("="*60)

fig, (ax_loss, ax_mae) = plt.subplots(2, 1, figsize=(10, 8))
epochs = range(1, len(historico_artigo.history['loss']) + 1)

ax_loss.plot(epochs, historico_artigo.history['loss'], label='Perda (Treino)', color='blue')
ax_loss.plot(epochs, historico_artigo.history['val_loss'], label='Perda (Validação)', color='orange')
ax_loss.set_title(f'Curva de Perda (MSE): {modelo_artigo.name}', fontsize=12)
ax_loss.set_ylabel('MSE')
ax_loss.legend()
ax_loss.grid(True)

ax_mae.plot(epochs, historico_artigo.history['mae'], label='MAE (Treino)', color='blue')
ax_mae.plot(epochs, historico_artigo.history['val_mae'], label='MAE (Validação)', color='orange')
ax_mae.set_title(f'Erro Médio Absoluto (MAE): {modelo_artigo.name}', fontsize=12)
ax_mae.set_xlabel('Épocas')
ax_mae.set_ylabel('MAE')
ax_mae.legend()
ax_mae.grid(True)

plt.tight_layout()
plt.savefig('Grafico_Aprendizado_Modelo_Original.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print(" GERANDO PREVISÕES DO MODELO E MAPAS DE CALOR")
print("="*60)

previsoes_artigo = modelo_artigo.predict(x_teste)

nomes_steps = ["Step 3 (Início do Ciclo)", "Step 12 (Meio do Ciclo)", "Step 21 (Fim do Ciclo)"]
elementos_por_linha = {
    'A': 3, 'B': 7, 'C': 9, 'D': 11, 'E': 11, 'F': 13, 'G': 13,
    'H': 13, 'I': 11, 'J': 11, 'K': 9, 'L': 7, 'M': 3
}

def plotar_mapa_radial(valores_reais, valores_previstos, titulo, nome_arquivo):
    mapa_erro = np.full((13, 13), np.nan)
    mapa_textos = np.full((13, 13), "", dtype=object)

    idx = 0
    for i, letra in enumerate(linhas):
        coluna_inicial_0 = Inicio_coluna[letra] - 1
        for j in range(elementos_por_linha[letra]):
            col_atual = coluna_inicial_0 + j
            real = valores_reais[idx]
            prev = valores_previstos[idx]
            erro_relativo = np.abs((real - prev) / (real + 1e-10)) * 100
            mapa_erro[i, col_atual] = erro_relativo
            mapa_textos[i, col_atual] = f"{real:.2f}\n{prev:.2f}\n{erro_relativo:.1f}%"
            idx += 1

    plt.figure(figsize=(10, 7.5))
    ax = sns.heatmap(
        mapa_erro, annot=mapa_textos, fmt="", cmap="Reds",
        linewidths=1, linecolor='gray', cbar_kws={'label': 'Erro Relativo (%)'}
    )

    plt.title(f"{titulo}", fontsize=15, pad=20)
    plt.yticks(ticks=np.arange(13)+0.5, labels=linhas, rotation=0)
    plt.xticks(ticks=np.arange(13)+0.5, labels=range(1, 14))
    ax.xaxis.tick_top()

    plt.tight_layout(pad=2.0)
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    plt.show()

resultados_finais = {'Medio': [], 'Maximo': [], 'Minimo': []}
posicoes_obrigatorias = {idx: indices_teste.index(idx) for idx in indices_obrigatorios}
nomes_por_indice = {2: "Step 3 (Início do Ciclo)", 11: "Step 12 (Meio do Ciclo)", 20: "Step 21 (Fim do Ciclo)"}

erros_totais_m1 = []

for i in range(len(x_teste)):
    reais = y_teste[i]
    prev_1 = previsoes_artigo[i]
    erro_1 = np.abs((reais - prev_1) / (reais + 1e-10)) * 100

    idx_real = indices_teste[i]
    if idx_real in indices_obrigatorios:
        erros_totais_m1.extend(erro_1)
        nome_step = nomes_por_indice[idx_real]
        print(f"\n\n\n{'='*70}")
        print(f"ANALISANDO: {nome_step.upper()} (índice real {idx_real})")
        print(f"{'='*70}\n")

        plotar_mapa_radial(
            reais, prev_1,
            f"{nome_step}\nModelo: {modelo_artigo.name}",
            f"Mapa_Erro_Original_Step_{idx_real}.png"
        )

        m1_medio, m1_max, m1_min = np.mean(erro_1), np.max(erro_1), np.min(erro_1)
        resultados_finais['Medio'].append(m1_medio)
        resultados_finais['Maximo'].append(m1_max)
        resultados_finais['Minimo'].append(m1_min)
        print(f"📊 {modelo_artigo.name} | Erro Médio: {m1_medio:.3f}% | Erro Máximo: {m1_max:.3f}%")

print("\n\n\n" + "="*60)
print(" HISTOGRAMA DE ERRO RELATIVO (%) - STEPS DE TESTE")
print("="*60)

plt.figure(figsize=(8, 5))
sns.histplot(erros_totais_m1, bins=30, kde=True, color='indigo')
plt.title(f'Distribuição de Erros: {modelo_artigo.name}')
plt.xlabel('Erro Relativo (%)')
plt.ylabel('Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("Histograma_Erros_Modelo_Original.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n\n" + "="*60)
print(" TABELA RESUMO DE ERROS (%) - MODELO 36_70_108_140_121")
print("="*60)

df_resultados = pd.DataFrame({
    'Ciclo de Queima': nomes_steps,
    'Erro Médio': resultados_finais['Medio'],
    'Erro Max': resultados_finais['Maximo'],
    'Erro Min': resultados_finais['Minimo']
}).round(3)
display(df_resultados)

# ==============================================================================
# FASE 3.3: VALIDAÇÃO LEAVE-ONE-OUT (LOO k=0 e k=1) E TESTE CEGO C05
# ==============================================================================
if len(x_anomalia) > 0:
    def validar_loo(vizinhos_excluidos=0):
        n = len(x_anomalia)
        linhas_res = []
        for i in range(n):
            excluir = set(range(max(0, i - vizinhos_excluidos), min(n, i + vizinhos_excluidos + 1)))
            idx_tr = [j for j in range(n) if j not in excluir]
            m, h = fine_tune(x_anomalia[idx_tr], y_anomalia[idx_tr])
            e_ft   = erro_rel(y_anomalia[i], m.predict(x_anomalia[i:i+1], verbose=0)[0])
            e_base = erro_rel(y_anomalia[i], prev_base_anom[i])
            linhas_res.append({
                'cenario': nomes_cenarios[i],
                'medio_FT': e_ft.mean(),   'max_FT': e_ft.max(),
                'medio_base': e_base.mean(), 'max_base': e_base.max(),
                'epocas': len(h.history['loss'])
            })
            del m
            gc.collect()
        return pd.DataFrame(linhas_res)

    print("Iniciando bateria de testes estatísticos (Leave-One-Out)...")
    df_loo_k0 = validar_loo(vizinhos_excluidos=0)
    df_loo_k1 = validar_loo(vizinhos_excluidos=1)

    def resumo(df, rotulo):
        return {
            'validação': rotulo,
            'medio_FT (média)': df['medio_FT'].mean(),
            'max_FT (pior caso)': df['max_FT'].max(),
            'medio_base (média)': df['medio_base'].mean(),
            'max_base (pior caso)': df['max_base'].max()
        }

    print("\n=== LOO k=0 (treina em todos os outros steps) ===")
    display(df_loo_k0.round(3))
    print("\n=== LOO k=1 (exclui também os steps vizinhos do treino) ===")
    display(df_loo_k1.round(3))
    print("\n=== RESUMO ESTATÍSTICO ===")
    display(pd.DataFrame([resumo(df_loo_k0, 'LOO k=0'), resumo(df_loo_k1, 'LOO k=1')]).round(3))

    # ==========================================================================
    # FASE 5: TESTE CEGO, MAPA RADIAL DE ANOMALIA E PROVA DE RETENÇÃO
    # ==========================================================================
    NOME_TESTE = 'A1-008-C05-07.dat'
    VIZINHOS_EXCLUIDOS = 1

    try:
        i_teste = next(i for i, n in enumerate(nomes_cenarios) if NOME_TESTE in n)
        excluir = set(range(max(0, i_teste - VIZINHOS_EXCLUIDOS), min(len(x_anomalia), i_teste + VIZINHOS_EXCLUIDOS + 1)))
        idx_tr = [j for j in range(len(x_anomalia)) if j not in excluir]

        modelo_ft, historico_ft = fine_tune(x_anomalia[idx_tr], y_anomalia[idx_tr])
        print(f"Fine-tuning: {len(idx_tr)} steps no treino | teste cego: {nomes_cenarios[i_teste]} | épocas: {len(historico_ft.history['loss'])}")
        modelo_ft.save('modelo_finetuned_c05.keras')

        # 1) Teste cego C05 + mapa radial
        prev = modelo_ft.predict(x_anomalia[i_teste:i_teste+1], verbose=0)[0]
        e = erro_rel(y_anomalia[i_teste], prev)
        print(f"\nAnomalia C05 ({nomes_cenarios[i_teste]}) - Erro Médio: {e.mean():.3f}% | Erro Máximo: {e.max():.3f}%")

        plotar_mapa_radial(
            y_anomalia[i_teste], prev,
            f"Anomalia C05 ({nomes_cenarios[i_teste]})\nModelo Fine-Tuned",
            "Mapa_Erro_FineTuning_C05.png"
        )

        # 2) Curva de aprendizado do fine-tuning
        plt.figure(figsize=(8, 4))
        plt.plot(historico_ft.history['loss'], color='blue')
        plt.title('Curva de Aprendizado — Fine-Tuning C05 (Treino)')
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()

        # 3) Esquecimento: operação normal antes x depois
        linhas_esq = []
        for idx_real in indices_obrigatorios:
            p = indices_teste.index(idx_real)
            r = y_teste[p]
            e_b = erro_rel(r, modelo_base.predict(x_teste[p:p+1], verbose=0)[0])
            e_f = erro_rel(r, modelo_ft.predict(x_teste[p:p+1], verbose=0)[0])
            linhas_esq.append({
                'Step': idx_real + 1,
                'Médio base': e_b.mean(), 'Máx base': e_b.max(),
                'Médio FT': e_f.mean(),   'Máx FT': e_f.max()
            })

        print("\nOperação normal (sem barra) — Base x Fine-Tuned (Prova de Retenção):")
        display(pd.DataFrame(linhas_esq).round(3))

    except StopIteration:
        print(f"AVISO: O arquivo '{NOME_TESTE}' não foi encontrado na lista. Arquivos disponíveis: {nomes_cenarios}")
