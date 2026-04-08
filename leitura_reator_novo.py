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


# FASE 1: EXTRAÇÃO BRUTA DO ARQUIVO DE TEXTO
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




# FASE 2: CONSTRUÇÃO FÍSICA DO REATOR 
# Esta é a matriz final e definitiva que representa o ciclo de vida físico do reator
historico_steps_queima = [] 

# Passa por cada step lido no arquivo bruto, um por um
for indice_step, dados_brutos_step in enumerate(leitura_bruta_steps): 
    
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

print("Histórico de Steps de Queima construído com sucesso!\n")


# VISUALIZAÇÃO DA SAÍDA FINAL:
# A variável 'historico_steps_queima' é uma LISTA DE LISTAS.
# [
#   [ ['1.05', '1.08'... 121 itens], ['1.08', '0.98'... 36 itens] ], <-- Step 1: [Combustíveis, Detectores]
#   [ ['1.04', '1.07'... 121 itens], ['1.07', '0.97'... 36 itens] ]  <-- Step 2: [Combustíveis, Detectores]
# ]




# EXEMPLOS DE EXTRAÇÃO DE DADOS
# historico_steps_queima[Numero_do_Step][0 para Combustíveis ou 1 para Detectores]

print("ANALISANDO O STEP DE QUEIMA 1")

combustiveis_step_1 = historico_steps_queima[0][0]

print(f"Total de Elementos Combustíveis mapeados: {len(combustiveis_step_1)} posições")
print(f"dados do elemento combustível: {combustiveis_step_1}") 

detectores_step_1 = historico_steps_queima[0][1] 
print(f"Total de Detectores mapeados: {len(detectores_step_1)} posições\n")

print("--- TESTE DE ACESSO PONTUAL ---")
# Pegando apenas o valor do primeiro detector no Step 1
primeiro_detector = historico_steps_queima[0][1][0] 
print(f"Leitura do primeiro detector no Step 1: {primeiro_detector}")