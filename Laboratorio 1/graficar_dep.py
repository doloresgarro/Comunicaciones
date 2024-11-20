import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def graficar_dep(signal, fs, title="Densidad Espectral de Potencia"):
    # scipy.signal.welch para estimar la DEP
    freqs, psd = welch(signal, fs=fs, window='hann', nperseg=2048, scaling='density')
    
    # Convertir frecuencias a MHz y potencia a dB
    freqs_mhz = freqs / 1e6
    psd_db = 10 * np.log10(psd)
    
    # Configurar el gráfico
    plt.figure(figsize=(14, 8))
    plt.plot(freqs_mhz, psd_db, color='darkblue', linewidth=1, label="DEP estimada")
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Frecuencia (MHz)", fontsize=14, labelpad=10)
    plt.ylabel("Potencia (dB/Hz)", fontsize=14, labelpad=10)
    plt.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
    plt.legend(fontsize=13, loc='upper right', frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    plt.show()
