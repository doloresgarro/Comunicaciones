# Ejercicio 1b

# Modulacion doble banda lateral con portadora suprimida (DBL PS)
# m(t) = sinc(100t)
# Frecuencia de la portadora fc = 1Khz

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

# Parámetros
fc = 1000           # Frecuencia de la portadora en Hz
fm = 100            # Parámetro de la función sinc (100t en este caso)
Ac = 1              # Amplitud de la portadora
Fs = 10000          # Frecuencia de muestreo en Hz
T = 0.05            # Duración de la señal en segundos
t = np.arange(0, T, 1/Fs)  # Vector de tiempo

# Señal m(t) = sinc(100t) y señal modulada x(t) con DBL-PS
m_t = np.sinc(fm * t)  # Señal sinc(100t)
x_t = m_t * (Ac * np.cos(2 * np.pi * fc * t))  # Modulación DBL-PS

# Creación de figura con 3 subplots
plt.figure()

# Señal modulada x(t)
plt.subplot(3,1,1)
plt.plot(t, x_t)
plt.title('Señal Modulada DBL-PS x(t) con m(t) = sinc(100t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

# Transformada de Fourier de x(t)
X_f = fftshift(fft(x_t)) / len(x_t)
f = np.linspace(-Fs/2, Fs/2, len(X_f))

# Gráfica de la Transformada de Fourier
plt.subplot(3,1,2)
plt.plot(f, np.abs(X_f))
plt.title('Transformada de Fourier de x(t)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('|X(f)|')
plt.grid(True)

# Densidad espectral de potencia
Pxx = np.abs(X_f)**2 / Fs
plt.subplot(3,1,3)
plt.plot(f, Pxx)
plt.title('Densidad Espectral de Potencia de x(t)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia/Frecuencia (dB/Hz)')
plt.grid(True)

# Ajuste de layout y mostrar
plt.tight_layout()
plt.show()
