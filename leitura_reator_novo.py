#caminho para a leitura do arquivo dat, Lembrar do arquivo está na mesma pasta que o código
caminho_arquivo = 'A1-008-Angra_1_Ciclo_08_Resultados.dat' 

#lista em que toda step de queima fica (Dicionários puros)
todos_steps = []
#nossa "tomada", quando o alvo for achado, o switch fica verdadeiro até encontrar
#o corte, invertendo a logica
alvo_certo = False

Inicio_coluna = {
    'A': 6, 'B': 4, 'C': 3, 'D': 2, 'E': 2, 'F': 1, 'G': 1,
    'H': 1, 'I': 2, 'J': 2, 'K': 3, 'L': 4, 'M': 6
}
#linhas de A a M do núcleo do reator
linhas = ['A','B','C','D','E','F','G','H','I','J','K','L','M']

# Posições dos detectores no núcleo, de acordo com o artigo que a gente usou
detectores = {
    "detectores_A": ['B6','C8','E6','F5','G2','G9','I5','J10','A7'], 
    "detectores_B": ['D7','E2','E10','G13','H10','I7','L4','L9'],   
    "detectores_C": ['C11','D5','F12','G4','H1','H8','J3','J12','K7'],
    "detectores_D": ['A8', 'C3','C9','F8','G11','H3','I10','J8','K4'],
    "Calibracao": ['G6'] 
}

alvo = "Distribuição de densidade de potência" # onde o código precisa começar a leitura de dados
corte = "Distribuição de axial offset" # onde o código precisa terminar a leitura de dados

with open(caminho_arquivo, 'r', encoding='latin-1') as arquivo: #caminho para abertura do arquivo
    for linha in arquivo: # laço de repetição, lendo linha por linha do arquivo
        texto_linha = linha.strip() # limpa a linha de caracteres invisíveis

        if not texto_linha:#se não tiver texto, só pula a linha
            continue
            
        if texto_linha.startswith(alvo):# o programa encontro nosso alvo, e inicia a coleta de dados
            Step_queima = {} #Os dados   
            alvo_certo = True 
            continue  
            
        if alvo_certo and texto_linha.startswith(corte):
            todos_steps.append(Step_queima) # A step de queima é 
            alvo_certo = False
            continue
    
        if alvo_certo:
            linha_limpa = linha.replace(',', ' ')
            partes = linha_limpa.split()
        
            if len(partes) > 0:
                identificador = partes[0].upper()
            
                if identificador in linhas:
                    valores = partes[1:]
                    Step_queima[identificador] = valores

    if alvo_certo and len(Step_queima) > 0:
        todos_steps.append(Step_queima)

total_encontrado = len(todos_steps)
print(f"Encontrados {total_encontrado} steps.\n")

# =========================================================
# NOVA LÓGICA: MATRIZ DE DADOS
# =========================================================
matriz_todos_steps = [] # Esta será a sua matriz mestre

for indice_step, Step_queima_atual in enumerate(todos_steps):
    
    lista_reator = [] 
    for letra in linhas:
        if letra in Step_queima_atual:
            lista_reator.extend(Step_queima_atual[letra])
            
    lista_detectores = []
    for nome_tipo, posicoes_alvo in detectores.items():
        for pos in posicoes_alvo:
            letra = pos[0].upper()
            numero = int(pos[1:])
            
            try:
                coluna_inicial = Inicio_coluna[letra]
                indice_real = numero - coluna_inicial
                
                if indice_real < 0:
                    raise IndexError
                
                #só os valores
                valor_encontrado = Step_queima_atual[letra][indice_real]
                lista_detectores.append(valor_encontrado)
            except (KeyError, IndexError):
                lista_detectores.append("ERRO")

    # Aqui nós empacotamos as duas listas e guardamos na matriz mestre.
    # O índice [0] será sempre o Reator, o índice [1] será sempre os Detectores.
    matriz_todos_steps.append([lista_reator, lista_detectores])

print("Matriz construída com sucesso!\n")

# =========================================================
# EXEMPLOS DE COMO SELECIONAR E EXTRAIR DADOS DA MATRIZ
# =========================================================

# Exemplo 1: Pegando todos os dados do Step 3 (Índice 2)
# step_3_completo = matriz_todos_steps[2]

# Exemplo 2: Pegando APENAS os 121 elementos do Step 5 (Índice 4, Posição 0)
print("--- TESTE DE ACESSO: Elementos do Reator no Step 5 ---")
elementos_step_5 = matriz_todos_steps[0][0][0]
print(elementos_step_5)
print(f"Total extraído: {len(elementos_step_5)} posições\n")

# Exemplo 3: Pegando APENAS os 36 detectores do Step 1 (Índice 0, Posição 1)
print("--- TESTE DE ACESSO: Detectores no Step 1 ---")
detectores_step_1 = matriz_todos_steps[0][0]
print(detectores_step_1)
print(f"Total extraído: {len(detectores_step_1)} posições\n")

# Exemplo 4: Pegando APENAS O VALOR DO PRIMEIRO DETECTOR (B6) DO STEP 1
# Índice 0 (Step 1) -> Índice 1 (Detectores) -> Índice 0 (Primeiro detector da lista)
print("--- TESTE DE ACESSO: Primeiro detector (B6) do Step 1 ---")
detector_B6_step_1 = matriz_todos_steps[0][1][0]
print(f"Valor: {detector_B6_step_1}") 