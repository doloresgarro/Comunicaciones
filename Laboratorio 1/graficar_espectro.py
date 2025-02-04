import matplotlib.pyplot as plt
from graficar_dep import calcular_dep
import numpy as np

def graficar_espectro(signal, fs, title="Espectro de la señal"):
    """
    Grafica el contenido frecuencial de una señal usando la densidad espectral de potencia

    Parameters:
        signal (numpy array): Señal a analizar.
        fs (float): Frecuencia de muestreo en Hz.
        title (str): Título del gráfico.
    """
    freqs, psd = calcular_dep(signal, fs, nperseg=1024)

    # convierto a escala logarítmica para mejor visualización
    psd_db = 10 * np.log10(psd + 1e-12)  

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_db, color='blue')
    plt.title(title, fontsize=14)
    plt.xlabel("Frecuencia (Hz)", fontsize=12)
    plt.ylabel("Densidad espectral de potencia (dB/Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
