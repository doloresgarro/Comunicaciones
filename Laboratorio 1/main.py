
# Librerías a utilizar
from rtlsdr import RtlSdr
import numpy as np
from graficar_dep import graficar_dep
from filtro_butterworth import aplicar_filtro_butter

# Configuración inicial
fs = 2.048e6      # frecuencia de muestreo
fc = 96.4e6       # frecuencia de la estación de radio
spf = 256 * 64    # muestras por frame
dur = 10          # duración en segundos
frames = int(dur * fs / spf)  # cantidad de frames necesarios

# Crear objeto SDR
sdr = RtlSdr()
sdr.sample_rate = fs
sdr.center_freq = fc
sdr.gain = 'auto'

# Captura de muestras
buffer = []
print(f"Capturando {frames} frames para {dur} segundos de datos...")
for counter in range(frames):
    data = sdr.read_samples(spf)
    buffer.append(data)

sdr.close()

# Concatenar muestras en un vector continuo
x = np.concatenate(buffer)
np.save("muestras_capturadas_96_5.npy", x)
print("Muestras capturadas y guardadas en 'muestras_capturadas_96_5.npy'.")

# Guardar las muestras en un archivo
np.save("muestras_capturadas_96_5.npy", x)
print(f"\nMuestras guardadas en el archivo 'muestras_capturadas.npy'.")

# Graficar la DEP de la señal original
graficar_dep(x, fs, title="DEP de la señal centrada en 96.5 MHz")

# Filtrar la señal
carsson_bw = 200e3  # ancho de banda típico para FM comercial
x_filtrada = aplicar_filtro_butter(x, fs, carsson_bw)

# Graficar la DEP de la señal filtrada
graficar_dep(x_filtrada, fs, title="DEP de la señal filtrada centrada en 96.5 MHz")
