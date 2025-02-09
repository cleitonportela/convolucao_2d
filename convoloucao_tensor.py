import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 📌 1. Carregar a imagem e converter para escala de cinza
imagem = cv2.imread("cat2.jpg", cv2.IMREAD_GRAYSCALE)

# 📌 2. Normalizar a imagem para o intervalo [0,1] e adicionar dimensões extras para compatibilidade com TensorFlow
imagem = imagem.astype(np.float32) / 255.0
imagem = np.expand_dims(imagem, axis=(0, -1))  # Shape: (1, altura, largura, 1)

# EFEITO BLUR (desfoque 9x9)
kernel_blur = np.ones((9, 9, 1, 1), np.float32) / 81  # Kernel maior


# EFEITO 3D
kernel_3d = np.array([[-2, -1,  0],
                                  [-1,  1,  1],
                                  [  0,  1,  2]], dtype=np.float32).reshape(3, 3, 1, 1)

# Nitidez
kernel_nitidez = np.array([[  0, -1,  0],
                                   [-1,  9, -1],
                                   [  0, -1,  0]], dtype=np.float32).reshape(3, 3, 1, 1)

# Efeito de bordas
kernel_sobel_5x5 = np.array([[-1, -2,  0,  2,  1],
                              [-2, -3,  0,  3,  2],
                              [-3, -4,  0,  4,  3],
                              [-2, -3,  0,  3,  2],
                              [-1, -2,  0,  2,  1]], dtype=np.float32).reshape(5, 5, 1, 1)



kernel = kernel_3d

# 📌 4. Aplicar a convolução 2D usando TensorFlow
imagem_filtrada = tf.nn.conv2d(imagem, filters=kernel, strides=1, padding="SAME")

# 📌 5. Remover dimensões extras para exibição
imagem_filtrada = np.squeeze(imagem_filtrada.numpy())  # Converte de Tensor para NumPy

# 📌 6. Exibir as imagens original e filtrada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(imagem), cmap="gray")
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(imagem_filtrada, cmap="gray")
plt.title("Imagem com Filtro Aplicado")

plt.show()