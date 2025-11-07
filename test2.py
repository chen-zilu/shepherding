import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# 假设 f 是 2D 场（比如灰度图）
# f = plt.imread("lena.png").mean(axis=2)

img_color = data.astronaut()   # 3通道彩色图像
f = color.rgb2gray(img_color)  # 转灰度

# 生成频率坐标
Nx, Ny = f.shape
kx = np.fft.fftfreq(Nx).reshape(-1,1)
ky = np.fft.fftfreq(Ny).reshape(1,-1)

# 2D FFT
F = np.fft.fft2(f)

# 1) 计算梯度
Fx = 1j * (2*np.pi) * kx * F
Fy = 1j * (2*np.pi) * ky * F
grad_x = np.real(np.fft.ifft2(Fx))
grad_y = np.real(np.fft.ifft2(Fy))

# 2) 计算拉普拉斯
Lap = -(2*np.pi)**2 * (kx**2 + ky**2) * F
laplace_f = np.real(np.fft.ifft2(Lap))

# 显示结果
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(f, cmap='gray')
plt.subplot(1,3,2); plt.title("grad_x"); plt.imshow(grad_x, cmap='gray')
plt.subplot(1,3,3); plt.title("Laplacian"); plt.imshow(laplace_f, cmap='gray')
plt.show()
