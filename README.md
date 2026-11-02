# ⚛️ Reconstrução de Densidade de Potência em Reatores Nucleares PWR (Angra 1) via Redes Neurais Artificiais (MLP) e Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-MLP%20%26%20Fine--Tuning-D00000?logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Pesquisa-F%C3%ADsica%20de%20Reatores%20%26%20IA-success)

---

## 📌 Visão Geral do Projeto

Este projeto investiga e implementa a **reconstrução contínua da distribuição radial de densidade de potência** no núcleo de um reator nuclear do tipo **PWR (*Pressurized Water Reactor*)**, utilizando dados reais/simulados do **Ciclo 08 da Usina Nuclear de Angra 1**.

A partir das leituras esparsas de apenas **36 detectores neutrônicos internos (SPNDs — *Self-Powered Neutron Detectors*)**, uma Rede Neural Artificial do tipo **Perceptron Multicamadas (MLP)** estima simultaneamente a potência relativa em todos os **121 elementos combustíveis** que compõem o núcleo do reator.

Além da operação nominal de queima do ciclo, o projeto introduz uma abordagem de **Transfer Learning (*Fine-Tuning*)** para adaptar o modelo especialista a cenários de **anomalia operacional severa** — especificamente a inserção da **barra de controle na posição C05** —, avaliando tanto a precisão da adaptação via validação **Leave-One-Out (LOO)** quanto a mitigação do **esquecimento catastrófico** (*Catastrophic Forgetting*).

> **Referência Científica Base:**
> *Continuous mapping of nuclear reactor core power using artificial neural network even in the presence of inactive detectors*.

---

## 🔬 Física e Geometria do Núcleo PWR (Angra 1)

O núcleo do reator Angra 1 possui uma geometria quase-cilíndrica (ovalada) disposta em uma grade **13 × 13** (linhas `A` até `M` e colunas `1` até `13`), totalizando **121 elementos combustíveis**:

| Linha do Núcleo | Coluna Física Inicial | Quantidade de Elementos Combustíveis |
| :---: | :---: | :---: |
| **A** | 6 | 3 |
| **B** | 4 | 7 |
| **C** | 3 | 9 |
| **D** | 2 | 11 |
| **E** | 2 | 11 |
| **F** | 1 | 13 |
| **G** | 1 | 13 |
| **H** | 1 | 13 |
| **I** | 2 | 11 |
| **J** | 2 | 11 |
| **K** | 3 | 9 |
| **L** | 4 | 7 |
| **M** | 6 | 3 |
| **Total** | — | **121 Elementos (Saída `Y`)** |

### Mapeamento dos 36 Detectores SPNDs (Entrada `X`)

Os 36 sensores estão divididos em 4 bancos de instrumentação nuclear mais 1 detector central de calibração:

- **`detectores_A` (9 SPNDs):** `B6, C8, E6, F5, G2, G9, I5, J10, M7`
- **`detectores_B` (8 SPNDs):** `D7, E2, E10, G13, H10, I7, L4, L9`
- **`detectores_C` (9 SPNDs):** `C11, D5, F12, G4, H1, H8, J3, J12, K7`
- **`detectores_D` (9 SPNDs):** `A8, C3, C9, F8, G11, H3, I10, J8, K4`
- **`Calibracao` (1 SPND):** `G6`

---

## 🧠 Arquitetura da Rede Neural (`Modelo_36_70_108_140_121`)

O modelo base reproduz a topologia física progressiva proposta na literatura, mapeando o espaço de 36 sensores para os 121 elementos combustíveis:

```text
[Entrada: 36 SPNDs]
        │
        ▼
[Hidden_1: Dense(70, activation='tanh')]   🔒 Congelada durante o Fine-Tuning
        │
        ▼
[Hidden_2: Dense(108, activation='tanh')]  🔒 Congelada durante o Fine-Tuning
        │
        ▼
[Hidden_3: Dense(140, activation='tanh')]  🔓 Ajustada no Fine-Tuning
        │
        ▼
[Saida_reator: Dense(121, activation='linear')] 🔓 Ajustada no Fine-Tuning
```

### Hiperparâmetros de Treinamento (Modelo Base)
- **Semente Global Fixa (`SEED = 24`):** Garante reprodutibilidade estrita em `NumPy`, `TensorFlow` e `Python random`.
- **Otimizador:** `Adam`
- **Função de Perda (Loss):** Erro Quadrático Médio (`MSE`)
- **Métrica Monitorada:** Erro Médio Absoluto (`MAE`)
- **Regularização Temporal:** `EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)`
- **Divisão de Dados Sem Vazamento (*Zero Data Leakage*):**
  - **Teste (20% intocado):** Inclui obrigatoriamente o **Step 3** (Início do Ciclo), **Step 12** (Meio do Ciclo) e **Step 21** (Fim do Ciclo), além de amostragem complementar aleatória.
  - **Validação (20% do pool de treino):** Separada exclusivamente para o `EarlyStopping`.
  - **Treino (80% do pool de treino):** Dados efetivos de ajuste de pesos.

---

## ⚙️ Metodologia para Anomalia Operacional (Barra de Controle C05)

Quando uma barra de controle é inserida na posição **C05**, ocorre uma depressão severa e localizada no fluxo neutrônico que altera o perfil radial do núcleo. Para evitar treinar uma rede do zero com poucos dados de anomalia, aplica-se **Transfer Learning (*Fine-Tuning*)**:

1. **Preservação do Conhecimento Físico Global:** Carrega-se o modelo especialista base (`modelo_artigo.keras`) e congelam-se as camadas iniciais (`Hidden_1` e `Hidden_2`), que codificam a simetria global do reator.
2. **Adaptação Local de Baixa Taxa de Aprendizado:** Apenas a camada `Hidden_3` e a camada `Saida_reator` permanecem treináveis, utilizando `Adam(learning_rate=1e-4)`, `batch_size=4` e `EarlyStopping(patience=15)`.
3. **Validação Estatística Leave-One-Out (LOO):**
   - **`LOO k=0`:** Para cada step de anomalia $i$, o modelo realiza fine-tuning em todos os demais cenários e é avaliado no cenário $i$ isolado.
   - **`LOO k=1` (Rigoroso):** Exclui do treino o cenário de teste $i$ e também seus vizinhos temporais imediatos ($i-1$ e $i+1$), impedindo interpolação temporal trivial entre steps adjacentes.
4. **Teste Cego e Prova de Retenção (*Catastrophic Forgetting*):**
   - Geração do mapa de calor radial 13×13 para o cenário cego `A1-008-C05-07.dat`.
   - Avaliação comparativa nos steps normais (Steps 3, 12 e 21) antes e após o fine-tuning para mensurar a retenção da operação nominal.

---

## 📊 Artefatos e Visualizações Geradas pelo Pipeline

Ao executar o script `rna_mlp_v3_com_barras.py`, os seguintes modelos e gráficos em alta resolução (`300 DPI`) são exportados automaticamente:

- `modelo_artigo.keras` — Pesos treinados da rede especialista para operação nominal.
- `modelo_finetuned_c05.keras` — Pesos ajustados via Transfer Learning para a anomalia da barra C05.
- `Grafico_Aprendizado_Modelo_Original.png` — Curvas de convergência de `MSE` e `MAE` (Treino vs. Validação).
- `Mapa_Erro_Original_Step_2.png` — Mapa de calor radial 13×13 do núcleo no **Step 3 (Início do Ciclo)**.
- `Mapa_Erro_Original_Step_11.png` — Mapa de calor radial 13×13 do núcleo no **Step 12 (Meio do Ciclo)**.
- `Mapa_Erro_Original_Step_20.png` — Mapa de calor radial 13×13 do núcleo no **Step 21 (Fim do Ciclo)**.
- `Histograma_Erros_Modelo_Original.png` — Distribuição de frequência do erro relativo percentual (%) com curva KDE.
- `Mapa_Erro_FineTuning_C05.png` — Mapa de calor radial 13×13 no teste cego com inserção da barra de controle C05.

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Instale as dependências em um ambiente Python 3.10+:

```bash
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib
```

### 2. Dados de Entrada (`.dat`)
O script detecta automaticamente os arquivos tanto no diretório local quanto no Google Colab (`/content/drive/MyDrive/Rna-reator/`):
- `A1-008-Angra_1_Ciclo_08_Resultados.dat` (Ciclo normal de queima)
- Pasta `Barra_Controle_C05_Todo_Step/*.dat` (Cenários com inserção da barra C05)

### 3. Execução
```bash
python rna_mlp_v3_com_barras.py
```

---

## 👤 Autor

**Isaac Salles** (`@totemicpuma`)
- Pesquisa e Desenvolvimento em Inteligência Artificial aplicada à Engenharia e Física de Reatores Nucleares.
