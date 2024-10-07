import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter


def load_imagem(imagem_path):
    # Função para carregar a imagem.
    imagem = cv2.imread(imagem_path)
    if imagem is None:
        print("Erro: Não foi possível carregar a imagem.")
        return
    else:
        return imagem
    
def load_imagem_grayscale(imagem_path,nova_imagem_path):
    # Função para carregar a imagem em grayscale.
    imagem_grayscale = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    if imagem_grayscale is None:
        print("Erro: Não foi possível carregar a imagem.")
        return
    else:
        cv2.imwrite(nova_imagem_path, imagem_grayscale)
        print(f"Imagem em grayscale salva como {nova_imagem_path}")
        return imagem_grayscale
    
def adicionar_ruido(imagem, nova_imagem_path, proporcao_ruido):
    # Adiciona ruído de sal e pimenta à imagem.
    # Copiar a imagem original para não modificar diretamente
    imagem_ruidosa = np.copy(imagem)
    # Número de pixels a serem afetados
    num_pixels = int(proporcao_ruido * imagem.size)
    # Gerar ruído 'sal' (pixels brancos)
    for _ in range(num_pixels // 2):  # Metade para o sal
        x = random.randint(0, imagem.shape[0] - 1)
        y = random.randint(0, imagem.shape[1] - 1)
        if len(imagem.shape) == 2:  # Imagem em escala de cinza
            imagem_ruidosa[x, y] = 255
        else:  # Imagem colorida
            imagem_ruidosa[x, y] = [255, 255, 255]
    # Gerar ruído 'pimenta' (pixels pretos)
    for _ in range(num_pixels // 2):  # Metade para a pimenta
        x = random.randint(0, imagem.shape[0] - 1)
        y = random.randint(0, imagem.shape[1] - 1)
        if len(imagem.shape) == 2:  # Imagem em escala de cinza
            imagem_ruidosa[x, y] = 0
        else:  # Imagem colorida
            imagem_ruidosa[x, y] = [0, 0, 0]
    cv2.imwrite(nova_imagem_path, imagem_ruidosa)
    print(f"Imagem com ruído salva como {nova_imagem_path}")
    return imagem_ruidosa

def reduzir_quantizacao(imagem, nova_imagem_path, num_cores):
    # Reduz a quantização da imagem para o número de cores especificado usando K-means.
    dados = imagem.reshape((-1, 3))
    dados = np.float32(dados)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, rotulos, centros = cv2.kmeans(dados, num_cores, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)
    imagem_quantizada = centros[rotulos.flatten()]
    imagem_quantizada = imagem_quantizada.reshape(imagem.shape)
    cv2.imwrite(nova_imagem_path, imagem_quantizada)
    print(f"Imagem quantizada salva como {nova_imagem_path}")
    return imagem_quantizada, rotulos.reshape(imagem.shape[:2]), centros
    
def obter_vizinhos(imagem, x, y, tamanho=3):
    # Função auxiliar para obter os vizinhos de um pixel (x, y) com uma janela quadrada de tamanho especificado.
    altura, largura = imagem.shape
    vizinhos = []
    # Definir o deslocamento da janela em relação ao centro (x, y)
    offset = tamanho // 2
    # Iterar pela janela 3x3 ao redor do pixel (x, y)
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            # Verificar se o vizinho está dentro da imagem
            if 0 <= x + i < altura and 0 <= y + j < largura:
                vizinhos.append(imagem[x + i, y + j])
    return vizinhos

def alterar_brilho(imagem, valor_brilho, nova_imagem_path): 
    # Conversão para o formato float para evitar erros ao adicionar o valor de brilho
    imagem_float = imagem.astype(np.float32)
    # Ajustar o brilho (somar o valor do brilho)
    imagem_brilho = cv2.add(imagem_float, valor_brilho)
    # Garantir que os valores estejam no intervalo correto (0-255)
    imagem_brilho = np.clip(imagem_brilho, 0, 255)
    # Converter de volta para uint8
    imagem_brilho = imagem_brilho.astype(np.uint8)
    # Salvar a nova imagem
    cv2.imwrite(nova_imagem_path, imagem_brilho)
    print(f"Imagem com brilho alterado salva como {nova_imagem_path}")

def imagem_negativa(imagem, nova_imagem_path):
    # Transformar a imagem em negativa
    imagem_negativa = 255 - imagem
    # Salvar a nova imagem
    cv2.imwrite(nova_imagem_path, imagem_negativa)
    print(f"Imagem negativa salva como {nova_imagem_path}")

def gerar_histograma(imagem, arquivo_texto):
    # Separar os canais RGB
    canais = cv2.split(imagem)
    # Criar um vetor para armazenar o histograma concatenado
    histograma_concatenado = np.array([])
    # Para cada canal (B, G, R)
    for i, canal in enumerate(canais):
        # Calcular o histograma para cada canal com 256 bins
        histograma = cv2.calcHist([canal], [0], None, [256], [0, 256])
        # Achatar o histograma em um vetor unidimensional
        histograma = histograma.flatten()
        # Concatenar o histograma no vetor final
        histograma_concatenado = np.concatenate((histograma_concatenado, histograma))
    # Salvar o histograma concatenado em um arquivo texto
    np.savetxt(arquivo_texto, histograma_concatenado, fmt='%d')
    
    print(f"Histograma concatenado salvo em {arquivo_texto}")

def visualizar_histograma(arquivo_texto):
    # Carregar o histograma do arquivo texto
    histograma_concatenado = np.loadtxt(arquivo_texto, dtype=int)
    # Cada canal tem 256 valores no histograma
    tamanho_canal = 256
    # Separar os histogramas de cada canal (B, G, R)
    histograma_b = histograma_concatenado[:tamanho_canal]
    histograma_g = histograma_concatenado[tamanho_canal:2*tamanho_canal]
    histograma_r = histograma_concatenado[2*tamanho_canal:]
    # Gerar o eixo x que representa os níveis de intensidade (0 a 255)
    x = np.arange(tamanho_canal)
    # Plotar os histogramas
    plt.figure(figsize=(10, 6))
    # Histograma do canal azul (B)
    plt.plot(x, histograma_b, color='blue', label='Canal Azul (B)')
    # Histograma do canal verde (G)
    plt.plot(x, histograma_g, color='green', label='Canal Verde (G)')
    # Histograma do canal vermelho (R)
    plt.plot(x, histograma_r, color='red', label='Canal Vermelho (R)')
    # Configurar o gráfico
    plt.title('Histograma dos Canais RGB')
    plt.xlabel('Níveis de Intensidade')
    plt.ylabel('Número de Pixels')
    plt.legend()
    # Exibir o gráfico
    plt.show()

def gerar_histograma_local(imagem, arquivo_texto, num_particoes):
    # Obter as dimensões da imagem
    altura, largura, _ = imagem.shape
    # Definir o número mínimo de partições como 3
    if num_particoes < 3:
        num_particoes = 3
    # Calcular o tamanho das partições
    tamanho_altura = altura // num_particoes
    tamanho_largura = largura // num_particoes
    # Criar um vetor para armazenar o histograma concatenado
    histograma_concatenado = np.array([])
    # Percorrer as partições da imagem
    for i in range(num_particoes):
        for j in range(num_particoes):
            # Definir os limites da partição
            y_inicio = i * tamanho_altura
            y_fim = (i + 1) * tamanho_altura if (i + 1) < num_particoes else altura
            x_inicio = j * tamanho_largura
            x_fim = (j + 1) * tamanho_largura if (j + 1) < num_particoes else largura
            # Extrair a partição da imagem
            particao = imagem[y_inicio:y_fim, x_inicio:x_fim]
            # Separar os canais RGB da partição
            canais = cv2.split(particao)
            # Para cada canal (B, G, R)
            for canal in canais:
                # Calcular o histograma com 256 bins para cada canal
                histograma = cv2.calcHist([canal], [0], None, [256], [0, 256])
                # Achatar o histograma em um vetor unidimensional
                histograma = histograma.flatten()
                # Concatenar o histograma no vetor final
                histograma_concatenado = np.concatenate((histograma_concatenado, histograma))
    # Salvar o histograma concatenado em um arquivo texto
    np.savetxt(arquivo_texto, histograma_concatenado, fmt='%d')
    
    print(f"Histograma local concatenado salvo em {arquivo_texto}")

def visualizar_histogramas_locais(arquivo_texto, num_particoes):
    # Carregar o histograma concatenado do arquivo texto
    histograma_concatenado = np.loadtxt(arquivo_texto, dtype=int)
    # Cada canal tem 256 valores no histograma
    tamanho_canal = 256
    histograma_por_particao = tamanho_canal * 3  # 3 canais (R, G, B) por partição
    # Quantidade total de partições
    num_total_particoes = num_particoes * num_particoes
    # Criar subplots para exibir os histogramas de cada partição
    fig, axs = plt.subplots(num_particoes, num_particoes, figsize=(15, 15))
    # Iterar sobre as partições e plotar os histogramas
    for i in range(num_total_particoes):
        # Calcular o índice da partição
        linha = i // num_particoes
        coluna = i % num_particoes
        # Extrair o histograma da partição atual
        inicio = i * histograma_por_particao
        fim = (i + 1) * histograma_por_particao
        histograma_particao = histograma_concatenado[inicio:fim]
        # Separar os histogramas de cada canal (B, G, R)
        histograma_b = histograma_particao[:tamanho_canal]
        histograma_g = histograma_particao[tamanho_canal:2*tamanho_canal]
        histograma_r = histograma_particao[2*tamanho_canal:]
        # Gerar o eixo x que representa os níveis de intensidade (0 a 255)
        x = np.arange(tamanho_canal)
        # Plotar o histograma de cada canal na partição
        axs[linha, coluna].plot(x, histograma_b, color='blue', label='Canal Azul (B)')
        axs[linha, coluna].plot(x, histograma_g, color='green', label='Canal Verde (G)')
        axs[linha, coluna].plot(x, histograma_r, color='red', label='Canal Vermelho (R)')
        # Configurações dos gráficos
        axs[linha, coluna].set_title(f'Partição ({linha+1}, {coluna+1})')
        axs[linha, coluna].set_xlabel('Níveis de Intensidade')
        axs[linha, coluna].set_ylabel('Número de Pixels')
        axs[linha, coluna].legend()
    # Ajustar layout dos subplots
    plt.tight_layout()
    plt.show()

def transformada_expansao_contraste(imagem, nova_imagem_path, new_max, new_min):
    # Encontra os valores mínimo e máximo originais da imagem
    min_val, max_val = imagem.min(), imagem.max()
    # Calcula os fatores de escala
    a = (new_max - new_min) / (max_val - min_val)
    b = new_min - a * min_val
    # Aplica a transformação linear
    imagem_transformada = a * imagem + b
    # Trunca os valores para o intervalo [0, 255]
    imagem_transformada = np.clip(imagem_transformada, 0, 255)
    imagem_transformada = imagem_transformada.astype(np.uint8)
    cv2.imwrite(nova_imagem_path, imagem_transformada)
    print(f"Imagem tranformada por expansão de contraste salva como {nova_imagem_path}")

def transformada_compressao_expansao(imagem, nova_imagem_path, gamma):
    # Definir a constante de normalização (c)
    c = 255.0 / (255.0 ** gamma)
    # Aplicar a transformação de compressão/expansão a todos os pixels
    imagem_transformada = c * (imagem ** gamma)
    cv2.imwrite(nova_imagem_path, imagem_transformada)
    print(f"Imagem tranformada por compressão e expansão salva como {nova_imagem_path}")

def transformada_dente_de_serra(imagem, nova_imagem_path, L):
    # Aplicar a transformação dente de serra a todos os pixels
    imagem_transformada = np.mod(imagem, L)
    cv2.imwrite(nova_imagem_path, imagem_transformada)
    print(f"Imagem tranformada por dente de serra salva como {nova_imagem_path}")

def transformada_logaritmica(imagem, nova_imagem_path,c):
    # Aplicar a transformação logarítmica
    imagem_transformada = c * (np.log(imagem.astype(float) + 1))
    # Specifica o tipo para conversão de float para inteiro
    imagem_transformada = np.array(imagem_transformada, dtype = np.uint8) 
    cv2.imwrite(nova_imagem_path, imagem_transformada)
    print(f"Imagem tranformada por logarítmo salva como {nova_imagem_path}")


def filtro_media(imagem, nova_imagem_path, tamanho_vizinhanca):
    altura, largura = imagem.shape
    imagem_filtrada = np.zeros_like(imagem)
    for i in range(altura):
        for j in range(largura):
            vizinhos = obter_vizinhos(imagem, i, j, tamanho_vizinhanca)
            # Aplicar a média dos valores dos vizinhos
            imagem_filtrada[i, j] = np.mean(vizinhos)
    cv2.imwrite(nova_imagem_path, imagem_filtrada)
    print(f"Imagem filtrada por media salva como {nova_imagem_path}")

def filtro_k_vizinhos_proximos(imagem, nova_imagem_path, k, tamanho_vizinhanca):
    altura, largura = imagem.shape
    imagem_filtrada = np.zeros_like(imagem)
    for i in range(altura):
        for j in range(largura):
            vizinhos = obter_vizinhos(imagem, i, j, tamanho_vizinhanca)
            vizinhos_ordenados = sorted(vizinhos)
            # Selecionar os K vizinhos mais próximos e tirar a média
            k_vizinhos = vizinhos_ordenados[:k]
            imagem_filtrada[i, j] = np.mean(k_vizinhos)
    cv2.imwrite(nova_imagem_path, imagem_filtrada)
    print(f"Imagem filtrada por k vizinhos salva como {nova_imagem_path}")

def filtro_mediana(imagem, nova_imagem_path, tamanho_vizinhanca):
    altura, largura = imagem.shape
    imagem_filtrada = np.zeros_like(imagem)
    for i in range(altura):
        for j in range(largura):
            vizinhos = obter_vizinhos(imagem, i, j, tamanho_vizinhanca)
            # Aplicar a mediana dos valores dos vizinhos
            imagem_filtrada[i, j] = np.median(vizinhos)
    cv2.imwrite(nova_imagem_path, imagem_filtrada)
    print(f"Imagem filtrada por mediana salva como {nova_imagem_path}")

def filtro_moda(imagem, nova_imagem_path, tamanho_vizinhanca):
    altura, largura = imagem.shape
    imagem_filtrada = np.zeros_like(imagem)
    for i in range(altura):
        for j in range(largura):
            vizinhos = obter_vizinhos(imagem, i, j, tamanho_vizinhanca)
            # Calcular a moda dos valores dos vizinhos
            moda = Counter(vizinhos).most_common(1)[0][0]
            imagem_filtrada[i, j] = moda
    cv2.imwrite(nova_imagem_path, imagem_filtrada)
    print(f"Imagem filtrada por moda salva como {nova_imagem_path}")

def deteccao_bordas(imagem_quantizada, nova_imagem_path):
    # Aplica o gradiente de Roberts manualmente para detectar bordas em uma imagem em escala de cinza
    imagem_cinza = cv2.cvtColor(imagem_quantizada, cv2.COLOR_BGR2GRAY)
    # Definir os kernels de Roberts
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=float)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=float)
    # Obter as dimensões da imagem
    altura, largura = imagem_cinza.shape
    # Inicializar as imagens para os gradientes
    gradiente_x = np.zeros_like(imagem_cinza, dtype=float)
    gradiente_y = np.zeros_like(imagem_cinza, dtype=float)
    # Aplicar convolução manualmente (sem funções prontas)
    for i in range(altura - 1):
        for j in range(largura - 1):
            # Aplicar o kernel de Roberts para x e y
            gx = (imagem_cinza[i, j] * kernel_roberts_x[0, 0] + 
                  imagem_cinza[i, j+1] * kernel_roberts_x[0, 1] + 
                  imagem_cinza[i+1, j] * kernel_roberts_x[1, 0] + 
                  imagem_cinza[i+1, j+1] * kernel_roberts_x[1, 1])
            gy = (imagem_cinza[i, j] * kernel_roberts_y[0, 0] + 
                  imagem_cinza[i, j+1] * kernel_roberts_y[0, 1] + 
                  imagem_cinza[i+1, j] * kernel_roberts_y[1, 0] + 
                  imagem_cinza[i+1, j+1] * kernel_roberts_y[1, 1])
            gradiente_x[i, j] = gx
            gradiente_y[i, j] = gy
    # Calcular a magnitude do gradiente (bordas)
    magnitude = np.sqrt(gradiente_x**2 + gradiente_y**2)
    # Normalizar para o intervalo [0, 255] e converter para uint8
    bordas = np.uint8(255 * magnitude / np.max(magnitude))
    cv2.imwrite(nova_imagem_path, bordas)
    print(f"Imagem das bordas salva como {nova_imagem_path}")
    return bordas

def extrair_propriedades_bic(imagem, bordas, rotulos, borda_path, interior_path, histograma_path, num_cores):
    # Definir um limiar para classificar bordas (ajuste conforme necessário)
    limiar_borda = np.max(bordas) * 0.1
    mascara_borda = bordas > limiar_borda
    # Inicializar os histogramas de borda e interior
    hist_borda = np.zeros(num_cores, dtype=int)
    hist_interior = np.zeros(num_cores, dtype=int)
    # Contar os pixels de borda e interior para cada cor
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            indice_cor = rotulos[i, j]
            if mascara_borda[i, j]:
                hist_borda[indice_cor] += 1
            else:
                hist_interior[indice_cor] += 1
    # Gerar as imagens com os pixels de borda e interior
    imagem_borda = np.full(imagem.shape, 255, dtype=np.uint8)  # Imagem branca
    imagem_interior = np.full(imagem.shape, 255, dtype=np.uint8)  # Imagem branca
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if mascara_borda[i, j]:
                imagem_borda[i, j] = imagem[i, j]  # Cor original para os pixels de borda
            else:
                imagem_interior[i, j] = imagem[i, j]  # Cor original para os pixels de interior
    # Salvar os histogramas em um arquivo texto
    with open(histograma_path, 'w') as f:
        f.write("Histograma de Borda:\n")
        f.write(", ".join(map(str, hist_borda)) + "\n")
        f.write("Histograma de Interior:\n")
        f.write(", ".join(map(str, hist_interior)) + "\n")
    # Salvar as imagens de borda e interior
    print(f"Imagem das bordas por bic salva como {borda_path}")
    cv2.imwrite(borda_path, imagem_borda)
    print(f"Imagem do interior por bic salva como {interior_path}")
    cv2.imwrite(interior_path, imagem_interior)
    return hist_borda, hist_interior

def visualizar_histogramas_bic(hist_borda, hist_interior, num_cores):
    # Definir os índices das cores
    cores = range(num_cores)
    # Criar a figura e os eixos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Plotar o histograma de borda
    ax1.bar(cores, hist_borda, color='blue', alpha=0.7)
    ax1.set_title("Histograma de Borda")
    ax1.set_xlabel("Índice de Cor")
    ax1.set_ylabel("Frequência")
    # Plotar o histograma de interior
    ax2.bar(cores, hist_interior, color='green', alpha=0.7)
    ax2.set_title("Histograma de Interior")
    ax2.set_xlabel("Índice de Cor")
    ax2.set_ylabel("Frequência")
    # Ajustar o layout
    plt.tight_layout()
    # Mostrar o gráfico
    plt.show()





# Caminhos dos arquivos a serem salvos/acessados.

caminho_imagem = "imagem_original.jpg" 
caminho_imagem_grayscale = "imagem_original_grayscale.jpg"  
caminho_imagem_ruidosa = "imagem_original_ruidosa.jpg"
caminho_imagem_quantizada = "imagem_original_quantizada.jpg"
caminho_imagem_brilho = "imagem_brilho.jpg" 
caminho_imagem_negativa = "imagem_negativa.jpg" 
caminho_histograma = "histograma_global.txt"  
caminho_histograma_local = "histograma_local.txt"  
caminho_transformada_expansao_contraste = 'imagem_expansao_contraste.jpg'
caminho_transformada_compressao_expansao = 'imagem_compressao_expansao.jpg'
caminho_transformada_dente_de_serra = 'imagem_dente_de_serra.jpg'
caminho_transformada_logaritmica = 'imagem_logaritmica.jpg'
caminho_filtro_media = 'imagem_media.jpg'
caminho_filtro_k_vizinhos = 'imagem_k_vizinhos.jpg'
caminho_filtro_mediana = 'imagem_mediana.jpg'
caminho_filtro_moda = 'imagem_moda.jpg'
caminho_bordas = 'imagem_bordas.jpg'
caminho_borda_bic = 'imagem_bordas_bic.jpg'
caminho_interior_bic = 'imagem_interior_bic.jpg'
caminho_histograma_bic = 'bic_histogramas.txt'

# Variaveis das funções

proporcao_ruido = 0.1 # Porcentagem de ruído. Por padrão 0.1 (10%)
num_cores = 256 # Número de cores a ser usado na quantização
valor_brilho = 100  # Valor de brilho a ser adicionado
num_particoes = 3 # Número de partições a ser usado na geração do histograma local
new_max = 200 # Valor máximo a ser usado na transformada de expansão de contraste 
new_min = 50 # Valor mínimo a ser usado na transformada de expansão de contraste 
gamma = 0.5 # Variável de luminancia a ser usado na transformada de compressão e expansão
L = 100 # Limiar a ser utilizado pela transformada dente de serra
c = 25 # Constante da transformada logarítmica
tamanho_vizinhanca = 3 # Quantidade a ser consultado ao visitar os vizinhos do pixel
k = 5 # Variável que especifica a quantidade de pixels a ser comparada na função de filtro dos k vizinhos proximos

# Funções para o carregamento das imagens

imagem = load_imagem(caminho_imagem)
imagem_grayscale = load_imagem_grayscale(caminho_imagem, caminho_imagem_grayscale)
imagem_ruidosa = adicionar_ruido(imagem_grayscale, caminho_imagem_ruidosa, proporcao_ruido)
imagem_quantizada, rotulos, centros = reduzir_quantizacao(imagem, caminho_imagem_quantizada, num_cores)

# Funções alvo

alterar_brilho(imagem, valor_brilho, caminho_imagem_brilho)
imagem_negativa(imagem, caminho_imagem_negativa)
gerar_histograma(imagem, caminho_histograma)
gerar_histograma_local(imagem, caminho_histograma_local, num_particoes)
transformada_expansao_contraste(imagem_grayscale, caminho_transformada_expansao_contraste, new_max, new_min)
transformada_compressao_expansao(imagem_grayscale, caminho_transformada_compressao_expansao, gamma)
transformada_dente_de_serra(imagem_grayscale, caminho_transformada_dente_de_serra, L)
transformada_logaritmica(imagem_grayscale, caminho_transformada_logaritmica,c)
filtro_media(imagem_ruidosa, caminho_filtro_media, tamanho_vizinhanca)
filtro_k_vizinhos_proximos(imagem_ruidosa, caminho_filtro_k_vizinhos, k, tamanho_vizinhanca)
filtro_mediana(imagem_ruidosa, caminho_filtro_mediana, tamanho_vizinhanca)
filtro_moda(imagem_ruidosa, caminho_filtro_moda, tamanho_vizinhanca)
bordas = deteccao_bordas(imagem_quantizada, caminho_bordas)
hist_borda, hist_interior = extrair_propriedades_bic(imagem_quantizada, bordas, rotulos, caminho_borda_bic, caminho_interior_bic, caminho_histograma_bic, num_cores)

# Funções para visualização de histogramas.

#visualizar_histograma(caminho_histograma)
#visualizar_histogramas_locais(caminho_histograma_local, num_particoes)
#visualizar_histogramas_bic(hist_borda, hist_interior, num_cores)