import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# 读取 skimage 内置的 lena（新版 skimage 用 astronaut，转换为灰度）
img_color = data.astronaut()
img = color.rgb2gray(img_color)

# 1) 二维傅里叶变换
F = np.fft.fft2(img)

# 2) 低频中心化
F_shift = np.fft.fftshift(F)

# 3) 计算幅度谱与相位谱
magnitude = np.log(np.abs(F_shift) + 1e-8)
phase = np.angle(F_shift)

# 4) 逆变换
F_ishift = np.fft.ifftshift(F_shift)
img_recon = np.real(np.fft.ifft2(F_ishift))

# 5) 可视化
plt.figure(figsize=(14,4))

plt.subplot(1,4,1)
plt.title("原始图像 (实域)")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("幅度谱 (频域)")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("相位谱 (频域)")
plt.imshow(phase, cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("逆变换图像（复原）")
plt.imshow(img_recon, cmap='gray')
plt.axis('off')

plt.show()
