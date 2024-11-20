import numpy as np
from scipy.signal import decimate
import sounddevice as sd
from filtro_butterworth import aplicar_filtro_butter
from discriminador import demodular_fm
from graficar_espectro import graficar_espectro  # Importar la función para graficar el espectro

# Configuración inicial
fs = 2.048e6  # Frecuencia de muestreo original
fc = 96.5e6  # Frecuencia central de la estación
cutoff_freq_1 = 200e3  # Frecuencia de corte para el primer filtro (200 kHz)
cutoff_freq_2 = 15e3   # Frecuencia de corte para el filtro de audio (15 kHz)
N1 = 10  # Factor de diezmado inicial
N2 = 5   # Factor de diezmado final

# Cargar las muestras capturadas (señal IQ)
x = np.load("muestras_capturadas_96_5.npy")  # Asegúrate de que este archivo exista

# Filtrado paso bajo inicial
y_B1 = aplicar_filtro_butter(x, cutoff_freq_1, fs)

# Diezmado inicial
fs1 = fs / N1
y_N1 = decimate(y_B1, N1, ftype='fir')

# Demodulación de la señal
z_dis = demodular_fm(y_N1, fs1)

# Graficar el espectro de la señal discriminada
graficar_espectro(z_dis, fs1, title="Espectro de la señal discriminada (Z_dis[n])")

# Filtrado paso bajo de audio
z_B2 = aplicar_filtro_butter(z_dis, cutoff_freq_2, fs1)

# Diezmado final
fs2 = fs1 / N2
z_N2 = decimate(z_B2, N2, ftype='fir')

# Graficar el espectro de la señal de audio final
graficar_espectro(z_N2, fs2, title="Espectro de la señal de audio demodulada (Z_out[n])")

# Normalización de la señal
z_out = z_N2 / np.max(np.abs(z_N2))

# Reproducir la señal demodulada
sd.play(z_out, samplerate=fs2)
sd.wait()

# Guardar el audio demodulado en un archivo para análisis posterior
np.save("audio_demodulado.npy", z_out)
