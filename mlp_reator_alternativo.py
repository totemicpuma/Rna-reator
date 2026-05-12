#treino da IA seguindo o caminho do trabalho Continuous mapping of nuclear reactor core power using artificial neural
#network even in the presence of inactive detectors 
"""
MLP - Mapeamento de Potência do Núcleo do Reator (Modelo Alternativo Otimizado)
============================================================
Diferenças estruturais implementadas para otimização:
  - Normalização StandardScaler em X e Y
  - BatchNormalization entre camadas ocultas
  - EarlyStopping + ReduceLROnPlateau
  - Seed global fixo para reprodutibilidade
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
#  SEMENTE GLOBAL – garante reprodutibilidade (resultados iguais sempre que rodar)
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Caminho para a leitura do arquivo dat. Lembrar que o arquivo deve estar na mesma pasta do código
caminho_arquivo = 'A1-008-Angra_1_Ciclo_08_Resultados.dat' 

# Lista temporária para guardar a leitura bruta do arquivo de texto
leitura_bruta_steps = []

# Nossa "tomada": quando o alvo for achado, o switch fica verdadeiro até encontrar o corte, invertendo a lógica
alvo_certo = False

# Dicionário que avisa ao código em qual coluna física cada linha realmente começa (formato oval do núcleo)
Inicio_coluna = {
    'A': 6, 'B': 4, 'C': 3, 'D': 2, 'E': 2, 'F': 1, 'G': 1,
    'H': 1, 'I': 2, 'J': 2, 'K': 3, 'L': 4, 'M': 6
}

# Linhas de A a M do núcleo do reator
linhas = ['A','B','C','D','E','F','G','H','I','J','K','L','M']

# Posições dos detectores no núcleo, de acordo com o artigo que usamos
detectores = {
    "detectores_A": ['B6','C8','E6','F5','G2','G9','I5','J10','A7'], 
    "detectores_B": ['D7','E2','E10','G13','H10','I7','L4','L9'],   
    "detectores_C": ['C11','D5','F12','G4','H1','H8','J3','J12','K7'],
    "detectores_D": ['A8', 'C3','C9','F8','G11','H3','I10','J8','K4'],
    "Calibracao": ['G6'] 
}

alvo = "Distribuição de densidade de potência" # Onde o código precisa começar a leitura de dados
corte = "Distribuição de axial offset" # Onde o código precisa terminar a leitura de dados


# ─────────────────────────────────────────────
# FASE 1: EXTRAÇÃO BRUTA DO ARQUIVO DE TEXTO
# ─────────────────────────────────────────────
with open(caminho_arquivo, 'r', encoding='latin-1') as arquivo: 
    for linha in arquivo: # Laço de repetição, lendo linha por linha do arquivo
        texto_linha = linha.strip() # Limpa a linha de caracteres invisíveis

        if not texto_linha: # Se não tiver texto, apenas pula a linha
            continue
            
        if texto_linha.startswith(alvo): # O programa encontrou nosso alvo e inicia a coleta de dados
            dados_brutos_step = {} # Cria uma "gaveta" vazia para guardar as linhas lidas neste exato momento   
            alvo_certo = True # Liga a tomada!
            continue  
            
        if alvo_certo and texto_linha.startswith(corte):
            leitura_bruta_steps.append(dados_brutos_step) # Guarda a leitura atual no nosso histórico bruto
            alvo_certo = False # Desliga a tomada
            continue
    
        # Se a nossa tomada estiver ligada, começa a limpar os números e a separar tudo
        if alvo_certo:
            linha_limpa = linha.replace(',', ' ') # Troca possíveis vírgulas por espaço
            partes = linha_limpa.split() # Corta a linha inteira em pedaços usando os espaços
        
            if len(partes) > 0: # Verifica se a linha lida não está totalmente vazia
                identificador = partes[0].upper() # Pega a 1ª letra da linha (ex: 'A')
            
                # Verifica se a letra lida faz parte das linhas reais do reator
                if identificador in linhas:
                    valores = partes[1:] # Pega apenas os números (ignora a letra)
                    dados_brutos_step[identificador] = valores # Guarda os números na gaveta dessa letra

    # Segurança: se o arquivo acabar do nada, não deixa de salvar a última leitura
    if alvo_certo and len(dados_brutos_step) > 0:
        leitura_bruta_steps.append(dados_brutos_step)

total_encontrado = len(leitura_bruta_steps)
print(f"Extraídos {total_encontrado} steps temporais do arquivo bruto.\n")


# VISUALIZAÇÃO DA SAÍDA BRUTA (Como o dado está agora na memória):
# A variável 'leitura_bruta_steps' é uma LISTA DE DICIONÁRIOS.
# [
#   { 'A': ['1.05', '1.08'...], 'B': ['0.98', '1.02'...], ... }, <-- Step 1 (Bruto)
#   { 'A': ['1.04', '1.07'...], 'B': ['0.97', '1.01'...], ... }  <-- Step 2 (Bruto)
# ]


# ─────────────────────────────────────────────
# FASE 2: CONSTRUÇÃO FÍSICA DO REATOR 
# ─────────────────────────────────────────────
# Esta é a matriz final e definitiva que representa o ciclo de vida físico do reator
historico_steps_queima = [] 

# Passa por cada step lido no arquivo bruto, um por um
for dados_brutos_step in leitura_bruta_steps: 
    
    # 1. Mapeamento de todos os 121 Elementos Combustíveis
    elementos_combustiveis = [] 
    for letra in linhas: # Vai de A até M, na ordem certa
        if letra in dados_brutos_step: # Se a letra existir na leitura atual...
            elementos_combustiveis.extend(dados_brutos_step[letra]) # Despeja todos os números juntos numa fila única

    # 2. Mapeamento das leituras dos 36 Detectores no mesmo instante        
    leituras_detectores = []
    for nome_tipo, posicoes_alvo in detectores.items(): # Passa pelos grupos de detectores
        for pos in posicoes_alvo: # Passa por cada detector separadamente (ex: 'B6')
            letra = pos[0].upper() # Pega só a letra ('B')
            numero = int(pos[1:]) # Pega só o número ('6') e converte de texto para número da matemática
            
            # Aqui é o coração da lógica de busca do formato oval
            try:
                coluna_inicial = Inicio_coluna[letra] # Olha no mapa onde essa letra começa no reator
                indice_real = numero - coluna_inicial # Faz uma conta para encontrar o índice no valor do reator
                                                      # ex: B6, inicio_coluna = 4, 6 - 4 = 2, índice 2, [0,1,2]
                
                if indice_real < 0: # Se der negativo, o detector está fora do núcleo
                    raise IndexError
                
                # Só os valores
                valor_encontrado = dados_brutos_step[letra][indice_real] # Pega o número exato do detector
                leituras_detectores.append(valor_encontrado) # Salva o número na lista de detectores
            
            # Se der erro (a linha sumiu ou o detector caiu num espaço vazio)
            except (KeyError, IndexError):
                leituras_detectores.append("ERRO") # Coloca um "ERRO" para a lista manter sempre as 36 posições

    # 3. Consolidação do Step de Queima
    # Empacotamos as duas listas. O índice [0] será o Reator, o índice [1] serão os Detectores.
    step_de_queima = [elementos_combustiveis, leituras_detectores]
    
    # Salva esse pacote formatado no histórico final do ciclo
    historico_steps_queima.append(step_de_queima)

# conversor para o modelo de IA
x_dados = []
y_dados = []

for step in historico_steps_queima:
    # filtra steps corrompidos (onde o dectetor marcou "ERRO")
    if "ERRO" not in step[1]:
        y_convertido = [float(valor) for valor in step[0]]
        x_convertido = [float(valor) for valor in step[1]]

        y_dados.append(y_convertido)
        x_dados.append(x_convertido)

# transformando em tensores NumPy
x = np.array(x_dados)
y = np.array(y_dados)


# ─────────────────────────────────────────────
# FASE 3: SEPARAÇÃO DOS STEPS DE TESTES
# ─────────────────────────────────────────────
# vamos trabalhar com o step 3, 12, 21, para abrangir o inicio, o meio e o fim do ciclo
# como o python trabalha com índice começando por 0, nossas steps serão [2,11,20]
indices_teste = [2, 11, 20]

# Cria uma lista com todos os outros steps que sobraram para a rede treinar
indices_treino = [i for i in range(len(x)) if i not in indices_teste]

# Separa as matrizes usando os índices que definimos
x_teste = x[indices_teste]
y_teste = y[indices_teste]

x_treino = x[indices_treino]
y_treino = y[indices_treino]

print(f"Steps usados para Teste: 3, 12, 21")
print(f"Total de amostras para treino: {len(x_treino)}")
print(f"Total de amostras para teste: {len(x_teste)}\n")


# ─────────────────────────────────────────────
#  FASE 4: NORMALIZAÇÃO DOS DADOS (Diferencial Otimizado)
# ─────────────────────────────────────────────
# StandardScaler torna a média=0 e o desvio padrão=1, o que facilita muito o aprendizado da função tanh
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_treino_norm = scaler_x.fit_transform(x_treino)
y_treino_norm = scaler_y.fit_transform(y_treino)

# Apenas transforma (sem re-ajustar) para não vazar informação de treino para os dados de teste
x_teste_norm  = scaler_x.transform(x_teste)
y_teste_norm  = scaler_y.transform(y_teste)


# ─────────────────────────────────────────────
# FASE 5: ARQUITETURA DA REDE ALTERNATIVA
# ─────────────────────────────────────────────
print("Construção do modelo alternativo (com BatchNorm)...")

# O BatchNormalization reescala as ativações internamente.
# Isso gera um gradiente mais estável e impede que os pesos explodam.
modelo_alt = Sequential([

    # Camada de Entrada (36 nós - leituras dos detectores normalizadas)
    Input(shape=(36,), name='Entrada_Detectores'),

    # Camada oculta 1 (72 nós com tanh + BatchNorm)
    Dense(72,  activation='tanh', name='Hidden_1'),
    BatchNormalization(name='BN_1'),

    # Camada oculta 2 (108 nós com tanh + BatchNorm)
    Dense(108, activation='tanh', name='Hidden_2'),
    BatchNormalization(name='BN_2'),

    # Camada oculta 3 (144 nós com tanh + BatchNorm)
    Dense(144, activation='tanh', name='Hidden_3'),
    BatchNormalization(name='BN_3'),

    # Camada oculta 4 de compressão simétrica (108 nós)
    Dense(108, activation='tanh', name='Hidden_4'),
    BatchNormalization(name='BN_4'),

    # Camada de Saída (121 nós, mapeamento linear das densidades de potência normalizadas)
    Dense(121, activation='linear', name='Saida_Reator'),
])

print("Compilando o modelo...")
modelo_alt.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['mae']
)

# Exibe o resumo no terminal
modelo_alt.summary()


# ─────────────────────────────────────────────
# FASE 6: CALLBACKS DE TREINAMENTO INTELIGENTE
# ─────────────────────────────────────────────
callbacks = [
    # EarlyStopping: Para o treino se o erro não melhorar por 150 épocas consecutivas e resgata o melhor peso.
    EarlyStopping(
        monitor='val_loss',
        patience=150,
        restore_best_weights=True,   
        verbose=1
    ),
    # ReduceLROnPlateau: Reduz a velocidade de aprendizado (freio automático) se o erro estagnar.
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=80,
        min_lr=1e-6,
        verbose=1
    ),
    # Salva silenciosamente o melhor modelo encontrado no disco
    ModelCheckpoint(
        filepath='melhor_modelo_reator.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    ),
]


# ─────────────────────────────────────────────
# FASE 7: TREINAMENTO
# ─────────────────────────────────────────────
print("\nIniciando o treinamento (com parada inteligente)...")

# Historico treino guarda as notas do "simulado a cada rodada"
historico = modelo_alt.fit(
    x_treino_norm, y_treino_norm,
    epochs=700,
    batch_size=16,
    validation_data=(x_teste_norm, y_teste_norm),
    callbacks=callbacks,
    verbose=1
)

print("\nTreinamento Concluído!")


# ─────────────────────────────────────────────
# FASE 8: VISUALIZAÇÃO DO APRENDIZADO
# ─────────────────────────────────────────────
# Gera um gráfico mostrando como o erro caiu ao longo das épocas
plt.figure(figsize=(10, 6))
plt.plot(historico.history['loss'], label='Erro no Treino (Loss)', color='steelblue')
plt.plot(historico.history['val_loss'], label='Erro na Validação (Val Loss)', color='tomato')
plt.title('Curva de Aprendizado da Rede Neural – MLP Alternativa')
plt.xlabel('Épocas (Rodadas de Treino)')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()

print("\nSalvando e abrindo o gráfico de aprendizado...")
plt.savefig('curva_aprendizado_alternativa.png', dpi=150)
plt.show()


# ─────────────────────────────────────────────
# FASE 9: ANÁLISE DE PRECISÃO FÍSICA E MAPAS DE CALOR
# ─────────────────────────────────────────────
# Quantidade de elementos combustíveis reais em cada linha de A até M (Angra 1)
elementos_por_linha = {
    'A': 3,  'B': 7,  'C': 9,  'D': 11, 'E': 11,
    'F': 13, 'G': 13, 'H': 13, 'I': 11, 'J': 11,
    'K': 9,  'L': 7,  'M': 3
}

# Função para montar e exibir o mapa do núcleo
def plotar_mapa_radial(valores_reais, valores_previstos, titulo):
    # Cria matrizes 13x13 vazias (preenchidas com NaN - Not a Number para a água)
    mapa_erro  = np.full((13, 13), np.nan)
    mapa_textos = np.full((13, 13), "", dtype=object)
    
    idx = 0 # Ponteiro para varrer a lista de 121 elementos

    # Reconstrói a matriz física 2D baseada no dicionário Inicio_coluna
    for i, letra in enumerate(linhas):
        col_0 = Inicio_coluna[letra] - 1 # Ajusta de 1-13 para índice 0-12 do Python
        qtd   = elementos_por_linha[letra]
        
        for j in range(qtd):
            real  = valores_reais[idx]
            prev  = valores_previstos[idx]
            erro  = np.abs((real - prev) / (real + 1e-10)) * 100
            
            # Salva o erro para definir a cor do quadrado
            mapa_erro[i, col_0 + j]   = erro
            
            # Formata o texto em 3 andares: Real \n Previsto \n Erro%
            mapa_textos[i, col_0 + j] = f"{real:.2f}\n{prev:.2f}\n{erro:.1f}%"
            idx += 1

    # Desenhando o Gráfico
    plt.figure(figsize=(14, 11))
    
    # Mapa de calor (cmap="Reds" deixa vermelho onde o erro é maior)
    ax = sns.heatmap(mapa_erro, annot=mapa_textos, fmt="", cmap="Reds",
                     linewidths=1, linecolor='gray',
                     cbar_kws={'label': 'Erro Relativo (%)'})
                     
    plt.title(f"Mapa do Núcleo: {titulo}\n(Real | RNA | Erro %)", fontsize=16, pad=20)
    plt.yticks(ticks=np.arange(13)+0.5, labels=linhas, rotation=0, fontsize=12)
    plt.xticks(ticks=np.arange(13)+0.5, labels=range(1, 14), fontsize=12)
    
    # Coloca o Eixo X na parte de cima igual à imagem de referência
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.savefig(f"mapa_{titulo.replace(' ','_')}.png", dpi=150)
    plt.show()

print("\n==================================================")
print("GERANDO MAPAS DE CALOR DO NÚCLEO")
print("==================================================")

# A rede devolve valores normalizados! Precisamos reverter isso para a escala real física (Desnormalizar)
previsoes_norm = modelo_alt.predict(x_teste_norm)
previsoes      = scaler_y.inverse_transform(previsoes_norm)

nomes_steps = ["Step 3 (Início do Ciclo)", "Step 12 (Meio do Ciclo)", "Step 21 (Fim do Ciclo)"]

# Loop para gerar a análise e os mapas dos 3 steps
for i in range(len(x_teste)):
    reais    = y_teste[i]
    previstos = previsoes[i]
    erros = np.abs((reais - previstos) / (reais + 1e-10)) * 100

    print(f"\n--- {nomes_steps[i]} ---")
    print(f"Erro Médio    : {np.mean(erros):.3f}%")
    print(f"Erro Máximo   : {np.max(erros):.3f}%")
    print(f"Desvio Padrão : {np.std(erros):.3f}%")

    # Chama a função que desenha a imagem na tela
    plotar_mapa_radial(reais, previstos, nomes_steps[i])