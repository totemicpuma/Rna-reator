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
    for linha in arquivo: # Laço de repetição, lendo linha por linha do arquivo de cima para baixo
        texto_linha = linha.strip() # Limpa a linha de caracteres invisíveis (espaços extras nas pontas)

        if not texto_linha: # Se não tiver texto, só pula a linha e vai para a próxima
            continue
            
        if texto_linha.startswith(alvo): # O programa encontrou nosso alvo e inicia a coleta de dados
            dados_brutos_step = {} # Cria uma "gaveta" vazia para guardar os dados desse momento   
            alvo_certo = True # Liga a tomada!
            continue  
            
        if alvo_certo and texto_linha.startswith(corte):
            leitura_bruta_steps.append(dados_brutos_step) # Guarda a leitura na nossa lista principal
            alvo_certo = False # Desliga a tomada, o bloco de dados acabou
            continue
    
        # Se a tomada estiver ligada, começa a organizar os números
        if alvo_certo:
            linha_limpa = linha.replace(',', ' ') # Troca qualquer vírgula por espaço
            partes = linha_limpa.split() # Corta a linha inteira em vários pedaços usando os espaços
        
            if len(partes) > 0: # Garante que sobrou algum texto na linha
                identificador = partes[0].upper() # Pega a 1ª letra da linha (ex: 'A')
            
                # Verifica se a letra lida faz parte das linhas reais do reator
                if identificador in linhas:
                    valores = partes[1:] # Pega todos os números da frente, ignorando a letra
                    dados_brutos_step[identificador] = valores # Guarda os números dentro da "gaveta" dessa letra

    # Segurança: se o arquivo acabar de repente e a tomada estiver ligada, salva a última step
    if alvo_certo and len(dados_brutos_step) > 0:
        leitura_bruta_steps.append(dados_brutos_step)

total_encontrado = len(leitura_bruta_steps)

# O terminal avisa apenas que a leitura foi concluída para não travar a tela
print(f"Extraídos {total_encontrado} steps temporais do arquivo bruto.")
nome_txt = "Relatorio_Steps_Reator.txt"
print(f"Gerando o arquivo '{nome_txt}'... Aguarde.")



# FASE 2: CONSTRUÇÃO DO REATOR E ESCRITA NO ARQUIVO TXT


# Abre (ou cria) um arquivo .txt em modo de escrita ('w' de write)
with open(nome_txt, 'w', encoding='utf-8') as arquivo_txt:
    
    # Passa por todos os dados coletados, step por step
    for indice_step, dados_brutos_step in enumerate(leitura_bruta_steps):
        
        # 1. Mapeamento de todos os 121 Elementos Combustíveis
        elementos_combustiveis = [] 
        for letra in linhas: # Vai de A até M na ordem certa
            if letra in dados_brutos_step:
                # Pega os números daquela letra e despeja todos numa fila indiana única
                elementos_combustiveis.extend(dados_brutos_step[letra])
                
        # 2. Mapeamento das leituras dos 36 Detectores no mesmo instante
        leituras_detectores = []
        for nome_tipo, posicoes_alvo in detectores.items(): # Abre os grupos de detectores
            for pos in posicoes_alvo: # Olha um detector por vez (ex: 'B6')
                letra = pos[0].upper() # Separa a letra ('B')
                numero = int(pos[1:]) # Separa o número ('6') e transforma em valor matemático
                
                # Aqui é a lógica para encontrar a posição certa no formato oval do reator
                try:
                    coluna_inicial = Inicio_coluna[letra] # Olha no mapa onde essa linha começa de verdade
                    
                    # Faz a conta para achar o índice. Ex: B6, incio_coluna = 4, logo: 6 - 4 = 2. Índice 2.
                    indice_real = numero - coluna_inicial 
                    
                    if indice_real < 0: # Se der negativo, (fora do núcleo)
                        raise IndexError
                    
                    # Só os valores
                    valor_encontrado = dados_brutos_step[letra][indice_real] # Vai na gaveta certa e pega o número,
                                                                             #Ex: B6, olha o dicionário B, e vai no 3 elemento da lista
                    leituras_detectores.append(valor_encontrado) # Salva o número na lista
                    
                # Se a linha não existir ou a conta cair fora do núcleo, joga para o erro
                except (KeyError, IndexError):
                    leituras_detectores.append("ERRO") # Grava "ERRO" para manter sempre os 36 espaços preenchidos

        # 3. Enviando os prints direto para o arquivo txt (usando o comando file=arquivo_txt)
        print(f"\n--- STEP {indice_step + 1} ---", file=arquivo_txt)
        print(f"Elementos combustíveis no reator", file=arquivo_txt)
        print(f"Elementos ({len(elementos_combustiveis)} posições):", file=arquivo_txt)
        print(elementos_combustiveis, file=arquivo_txt) # Imprime a fila indiana dos 121 elementos
        
        print(f"\nDetectores ({len(leituras_detectores)} posições):", file=arquivo_txt)
        print(leituras_detectores, file=arquivo_txt) # Imprime a fila indiana dos 36 detectores
        print("--------------------------------------------------", file=arquivo_txt)

    # Quando acabar de passar por todos os steps, avisa no txt que acabou
    print("\n=== FIM DA LEITURA ===", file=arquivo_txt)

