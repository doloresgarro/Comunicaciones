import os
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))


# Librerías a utilizar
from rtlsdr import RtlSdr
import numpy as np
from graficar_dep import graficar_dep
from graficar_espectro import graficar_espectro
from filtro_butterworth import aplicar_filtro_butter
from FM_DEMOD_GarroDolores import demodulador_fm

# Configuración inicial
fs = 2.048e6      # frecuencia de muestreo
fc = 96.5e6       # frecuencia de la estación de radio
#fc = 103.7e6      # frecuencia de la nueva estación de radio
spf = 256 * 64    # muestras por frame
dur = 10          # duración en segundos
frames = int(dur * fs / spf)  # cantidad de frames necesarios
carsson_bw = 200e3  # ancho de banda típico para FM comercial

# Configuración para demodular
B2 = 15e3  # Frecuencia de corte del filtro de audio (15 kHz)
fd = 75e3
N1 = 10 
N2 = 5
# N1 = 8
# N2 = 5



# Crear objeto SDR
sdr = RtlSdr()
sdr.sample_rate = fs # configuro frecuencia de muestreo
sdr.center_freq = fc # configuro frecuencia de estación de radio
sdr.gain = 'auto'    # configuro ganacia automáticamente 


# Captura de muestras
buffer = []
print(f"Capturando {frames} frames para {dur} segundos de datos...")
for counter in range(frames):
    data = sdr.read_samples(spf)
    buffer.append(data)

sdr.close()

# Concatenar muestras en un vector
x = np.concatenate(buffer)
np.save("muestras_capturadas_96_5.npy", x)
print("Muestras capturadas y guardadas en 'muestras_capturadas_96_5.npy'.")


# Guardar las muestras en un archivo
np.save("muestras_capturadas_103_7.npy", x)
print(f"\nMuestras guardadas en el archivo 'muestras_capturadas_103_7.npy'.")


# Graficar la DEP de la señal original
print ("Graficando la DEP de la señal original...")
graficar_dep(x, fs, title="DEP de la señal centrada en 96.5 MHz")

# Graficar el ESPECTRO de la señal original 
print ("Graficando el espectro de la señal original..")
graficar_espectro(x, fs, title="Espectro de la  señal original")

# Filtrar la señal
#x_filtrada = aplicar_filtro_butter(x, fs, carsson_bw)
x_filtrada = aplicar_filtro_butter(x, carsson_bw, fs)

# Graficar la DEP de la señal filtrada
print ("Graficando la DEP de la señal filtrada...")
graficar_dep(x_filtrada, fs, title="DEP de la señal filtrada centrada en 96.5 MHz")

# Graficar el ESPECTRO de la señal filtrada 
print ("Graficando el espectro de la señal filtrada..")
graficar_espectro(x, fs, title="Espectro de la  señal filtrada")

"""
# Graficar la DEP de la señal original
print ("Graficando la DEP de la señal original...")
graficar_dep(x, fs, title="DEP de la señal centrada en 103.7 MHz")

# Graficar el ESPECTRO de la señal original 
print ("Graficando el espectro de la señal original..")
graficar_espectro(x, fs, title="Espectro de la  señal original")

# Filtrar la señal
#x_filtrada = aplicar_filtro_butter(x, fs, carsson_bw)
x_filtrada = aplicar_filtro_butter(x, carsson_bw, fs)

# Graficar la DEP de la señal filtrada
print ("Graficando la DEP de la señal filtrada...")
graficar_dep(x_filtrada, fs, title="DEP de la señal filtrada centrada en 103.7 MHz")

# Graficar el ESPECTRO de la señal filtrada 
print ("Graficando el espectro de la señal filtrada..")
graficar_espectro(x, fs, title="Espectro de la  señal filtrada")
"""

# demodular
# x, B1, B2, N1, N2, fd, fs)
z_out = demodulador_fm(x, carsson_bw, B2, N1, N2, fd, fs)
