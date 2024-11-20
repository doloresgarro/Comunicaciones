import matplotlib.pyplot as plt
from scipy.signal import welch

def graficar_espectro(signal, fs, title="Espectro de la señal"):
    """
    Graficar el contenido frecuencial de una señal usando la densidad espectral de potencia.
    
    Parameters:
        signal (numpy array): Señal a analizar.
        fs (float): Frecuencia de muestreo en Hz.
        title (str): Título del gráfico.
    """
    freqs, psd = welch(signal, fs, nperseg=1024)
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.title(title, fontsize=14)
    plt.xlabel("Frecuencia (Hz)", fontsize=12)
    plt.ylabel("Densidad espectral de potencia (dB/Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
