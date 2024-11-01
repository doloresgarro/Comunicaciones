import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# Parámetros de la señal
Ac = 1         # Amplitud de la portadora
a = 0.5        # Índice de modulación
fc = 1000      # Frecuencia de la portadora en Hz
fm = 100       # Frecuencia del mensaje en Hz
fs = 10000     # Frecuencia de muestreo
t = np.arange(0, 0.05, 1/fs)  # Vector de tiempo para 50 ms

# Señal de mensaje m(t) = cos(2 * pi * fm * t)
mt = np.cos(2 * np.pi * fm * t)

# Señal modulada AM
am_signal = Ac * (1 + a * mt) * np.cos(2 * np.pi * fc * t)

# Transformada de Fourier (TF) de la señal AM
N = len(am_signal)
yf = fft(am_signal)
xf = fftfreq(N, 1 / fs)

# Densidad Espectral de Potencia (DEP) usando el método de Welch
frequencies, power_density = welch(am_signal, fs, nperseg=1024)

# Crear un solo gráfico con subplots
plt.figure(figsize=(12, 10))

# Gráfico de la señal AM en el tiempo
plt.subplot(3, 1, 1)
plt.plot(t, am_signal, label='Señal AM', color='blue')
plt.plot(t, Ac * (1 + a * mt), 'r--', label='Envolvente superior')
plt.plot(t, -Ac * (1 + a * mt), 'r--', label='Envolvente inferior')
plt.title('Señal de AM con a = 0.5 y m(t) = cos(2π100t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()

# Gráfico de la Transformada de Fourier
plt.subplot(3, 1, 2)
plt.plot(xf[:N // 2], 2.0 / N * np.abs(yf[:N // 2]), color='green')
plt.title('Transformada de Fourier de la señal AM')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.grid()

# Gráfico de la Densidad Espectral de Potencia
plt.subplot(3, 1, 3)
plt.semilogy(frequencies, power_density, color='purple')
plt.title('Densidad Espectral de Potencia de la señal AM')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia')
plt.grid()

# Ajuste final de los subplots
plt.tight_layout()
plt.show()
