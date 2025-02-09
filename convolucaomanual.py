import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar a imagem (usando matplotlib)
imagem = plt.imread("natureza.jpg")

# 2. Converter para escala de cinza manualmente
if len(imagem.shape) == 3:
    imagem = np.dot(imagem[...,:3], [0.2989, 0.5870, 0.1140])

# 3. Normalizar para [0,1]
imagem = imagem.astype(np.float32) / 255.0

# 4. Criar kernel de média 9x9 manualmente
kernel_size = 9
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

# 5. Aplicar convolução manualmente
def convolucao_manual(imagem, kernel):
    pad = kernel_size // 2
    # Adicionar bordas à imagem
    imagem_padded = np.pad(imagem, pad, mode='constant')
    
    # Criar imagem de saída
    output = np.zeros_like(imagem)
    
    # Percorrer cada pixel da imagem original
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            # Extrair região de interesse
            regiao = imagem_padded[i:i+kernel_size, j:j+kernel_size]
            # Aplicar o kernel e armazenar resultado
            output[i,j] = np.sum(regiao * kernel)
            
    return output

# 6. Aplicar o filtro
imagem_filtrada = convolucao_manual(imagem, kernel)

# 7. Normalização final para exibição
imagem_filtrada = (imagem_filtrada - np.min(imagem_filtrada)) 
imagem_filtrada /= (np.max(imagem_filtrada) - np.min(imagem_filtrada) + 1e-8)

# 8. Exibir resultados
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(imagem, cmap='gray')
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(imagem_filtrada, cmap='gray')
plt.title('Filtro de Média 9x9 Manual')
plt.show()