import matplotlib.pyplot as plt
import numpy as np

def calcular_dep(signal, fs, nperseg=2048):

    num_segments = len(signal) // nperseg # segmentos en los que se divide la señal
    psd_accum = np.zeros(nperseg // 2)


    for i in range(num_segments):
        segment = signal[i * nperseg:(i + 1) * nperseg] * np.hanning(nperseg)
        spectrum = np.fft.fft(segment) # calcula la FFT
        psd_accum += np.abs(spectrum[:nperseg // 2]) ** 2

    psd = psd_accum / num_segments
    freqs = np.fft.fftfreq(nperseg, 1/fs)[:nperseg // 2]
    return freqs, psd


def graficar_dep(signal, fs, title="Densidad Espectral de Potencia"):
    freqs, psd = calcular_dep(signal, fs)
    psd_db = 10 * np.log10(psd)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs / 1e6, psd_db, color='blue')
    plt.title(title)
    plt.xlabel("Frecuencia (MHz)")
    plt.ylabel("Potencia (dB/Hz)")
    plt.grid()
    plt.show()

