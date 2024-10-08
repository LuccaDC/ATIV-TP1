# Trabalho Prático 1 de Análise e Tratamento de Imagens e Vídeos Digitais

### Universidade Federal do Amazonas
#### Lucca Dourado Cunha - 22051490


Este primeiro trabalho prático tem como objetivo a compreensão sobre algoritmos de manipulação de imagens digitais, por meio da implementação de operações básicas. Abaixo compilei separado por tópicos os resultados que obtive ao aplicar os algoritmos solicitados na imagem de exemplo a seguir:

![Thiago Silva em 2018](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_original.jpg?raw=true)

## Alteração de brilho

Para a alteração de brilho foi adicionado o valor 100 a toda imagem para obter o seguinte resultado:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_brilho.jpg?raw=true)

## Imagem Negativa


![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_negativa.jpg?raw=true)

## Histograma Global

Para visualização dos dados do histograma global uma função foi feita para plotagem dos dados com a utilização da biblioteca do Python matplotlib.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/grafico_histograma_global.png?raw=true)

## Histograma Local

Para visualização dos dados do histograma local também foi feita uma função para plotagem dos dados com a utilização da biblioteca do Python matplotlib. Como parâmetro para a geração do histograma foi escolhido 3x3 para ser o número de partições da imagem.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/grafico_histograma_local.png?raw=true)

## Transformadas Radiométricas

Para os quatro procedimentos a seguir foi utilizado a seguinte imagem, que nada mais é a original em grayscale:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_original_grayscale.jpg?raw=true)

### Expansão de Contraste Linear

Para essa transformada de Expansão de Contraste Linear foi escolhido o intervalo entre 50 e 200 nos seus parâmetros, o que resultou na imagem abaixo:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_expansao_contraste.jpg?raw=true)

### Compressão e Expansão

Nesta transformada de Compressão e Expansão o valor 0.5 foi atribuído a gama na função, gerando a seguinte imagem:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_compressao_expansao.jpg?raw=true)

### Dente de Serra

Para essa imagem abaixo a transformada dente de serra utilizou o valor 100 no parâmetro de período da função:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_dente_de_serra.jpg?raw=true)

### Transformada do Logaritmo

Para essa transformada logarítmica a constante C assumiu o valor 25, e gerou o resultado abaixo:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_logaritmica.jpg?raw=true)

## Filtros Espaciais

Para os quatro procedimentos a seguir foi gerado a seguinte imagem a partir da original com ruído do tipo sal e pimenta em 10% da imagem:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_original_ruidosa.jpg?raw=true)

Além disso como parâmetro consideraremos uma vizinhança de 3x3 para os 4 filtros a seguir.

### Filtro da Média

O Filtro da média não obteve um bom resultado na diminuição de ruído. Ele ficou menos destacado por causa do desfoque da imagem inteira mas ainda é totalmente perceptível.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_media.jpg?raw=true)

### Filtro dos K Vizinhos Mais Próximos

No filtro dos K vizinhos mais próximos utilizamos o valor 5 no parâmetro K. Com isso obtivemos um ótimo resultado ao remover ruídos do tipo sal, porém os do tipo pimenta se mantiveram na imagem final e cresceram levemente.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_k_vizinhos.jpg?raw=true)

### Filtro da Mediana

O filtro da mediana foi o filtro que melhor resolveu o problema do ruído. A imagem final em sua grande parte ficou limpa de ruídos, podemos perceber que restaram apenas alguns escassos pontos isolados, majoritariamente do tipo sal.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_mediana.jpg?raw=true)

### Filtro da Moda

O filtro da moda foi de longe o filtro com o pior resultado. Em partes mais complexas da imagem ele deu uma leve melhorada na quantidade de ruído, como podemos ver no background da imagem. Porém com as partes mais complexas da imagem, como o rosto da pessoa, o ruído se propagou e distorceu a imagem.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_moda.jpg?raw=true)

## Detecção de Bordas

### Quantização da imagem
Para a Detecção de bordas foi utilizada a seguinte imagem quantizada (128 Cores):

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_original_quantizada.jpg?raw=true)

### Técnica de detecção das bordas

Para este trabalho foi utilizado a técnica "Gradiente de Roberts", que a partir da imagem quantizada nos gerou o resultado abaixo:

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_bordas.jpg?raw=true)

## Descritor BIC

### Histogramas

Novamente para visualização dos dados dos histogramas de borda e de interior uma função foi feita para plotagem dos dados com a utilização da biblioteca do Python matplotlib.

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/grafico_histograma_bic.png?raw=true)

### Pixeis de Borda

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_bordas_bic.jpg?raw=true)

### Pixeis de Interior

![](https://github.com/LuccaDC/ATIV-TP1/blob/main/Artefatos/imagem_interior_bic.jpg?raw=true)
