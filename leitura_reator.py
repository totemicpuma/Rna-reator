#caminho para a leitura do arquivo dat, Lembrar do arquivo está na mesma pasta que o código
caminho_arquivo = 'A1-008-Angra_1_Ciclo_08_Resultados.dat' 

#lista em que toda step de queima fica
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
            
            #aqui é 
            try:
                coluna_inicial = Inicio_coluna[letra]
                indice_real = numero - coluna_inicial
                
                if indice_real < 0:
                    raise IndexError
                
                #só os valores
                valor_encontrado = Step_queima_atual[letra][indice_real]
                lista_detectores.append(valor_encontrado)
                #lista_detectores.append(f"{pos}: {valor_encontrado}") 
            except (KeyError, IndexError):
                lista_detectores.append("ERRO")
                #lista_detectores.append(f"{pos}: ERRO")


    print(f"\n--- STEP {indice_step + 1} ---")
    print(f"\nElementos combustíveis no reator")
    print(f"Elementos ({len(lista_reator)} posições):")
    print(lista_reator)
    
    print(f"Detectores ({len(lista_detectores)} posições):")
    print(lista_detectores)

print("\n=== FIM DA LEITURA ===")