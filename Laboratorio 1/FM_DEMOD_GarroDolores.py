import numpy as np
from scipy.signal import decimate
import sounddevice as sd
from filtro_butterworth import aplicar_filtro_butter
from discriminador import demodular_fm
from graficar_espectro import graficar_espectro 

def demodulador_fm(x, B1, B2, N1, N2, fd, fs):
    """
    Procesa un demodulador FM utilizando las muestras capturadas.

    Parameters:
        x (numpy array): Vector de muestras capturadas (IQ).
        B1 (float): Frecuencia de corte para el primer filtro pasa bajos (Hz).
        B2 (float): Frecuencia de corte para el filtro de audio (Hz).
        N1 (int): Factor de diezmado inicial.
        N2 (int): Factor de diezmado final.
        fd (float): Desviación de frecuencia típica (Hz).
        fs (float): Frecuencia de muestreo original (Hz).
    Returns:
        z_out (numpy array): Señal de audio final normalizada.
    """

    # PASO 1: filtrado pasa bajos inicial
    y_B1 = aplicar_filtro_butter(x, B1, fs)

    # PASO 2: diezmado inicial
    fs1 = fs / N1
    y_N1 = decimate(y_B1, N1, ftype='fir')

    # PASO 3: demodulación de la señal
    z_dis = demodular_fm(y_N1, fs1, fd)

    # Graficar el espectro de la señal discriminada
    graficar_espectro(z_dis, fs1, title="Espectro de la señal discriminada (Z_dis[n])")

    # PASO 4: filtrado de audio
    z_B2 = aplicar_filtro_butter(z_dis, B2, fs1)

    # PASO 5: segundo diezmado 
    fs2 = fs1 / N2
    z_N2 = decimate(z_B2, N2, ftype='fir')

    # Graficar el espectro de la señal de audio final
    graficar_espectro(z_N2, fs2, title="Espectro de la señal de audio demodulada (Z_out[n])")

    # PASO 6: Normalización de la señal
    if np.max(np.abs(z_N2)) != 0:
        z_out = z_N2 / np.max(np.abs(z_N2))
    else: z_out = z_N2 # si no incluyo esto puede tirar error

    # chequeo si valores son distintos de 0 
    print(f"Min: {np.min(z_out)}, Max: {np.max(z_out)}")


    # Reproducir audio
    sd.play(z_out, samplerate=fs2)
    sd.wait()

    # Guardar el audio demodulado en un archivo para análisis posterior
    np.save("audio_demodulado.npy", z_out)

    return z_out 

   
