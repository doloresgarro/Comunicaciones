# from scipy.signal import butter, lfilter

# def aplicar_filtro_butter(signal, fs, cutoff, order=5):
#     """
#     Aplica un filtro Butterworth pasa-bajos a la señal.
    
#     Parameters:
#         signal (numpy array): Vector de muestras complejas (IQ).
#         fs (float): Frecuencia de muestreo en Hz.
#         cutoff (float): Frecuencia de corte del filtro en Hz.
#         order (int): Orden del filtro.
    
#     Returns:
#         numpy array: Señal filtrada.
#     """
#     nyquist = fs / 2  # Frecuencia de Nyquist
#     normal_cutoff = cutoff / nyquist  # Frecuencia de corte normalizada
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     filtered_signal = lfilter(b, a, signal)

from scipy.signal import butter, lfilter

def aplicar_filtro_butter(data, cutoff, fs, order=5):
    """
    Aplica un filtro Butterworth de orden dado a los datos proporcionados.

    Parameters:
        data (numpy array): Señal de entrada.
        cutoff (float): Frecuencia de corte (Hz).
        fs (float): Frecuencia de muestreo (Hz).
        order (int): Orden del filtro.

    Returns:
        numpy array: Señal filtrada.
    """
    nyquist = fs / 2  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Normalización de la frecuencia de corte
    if not 0 < normal_cutoff < 1:
        raise ValueError("La frecuencia de corte debe estar entre 0 y la frecuencia de Nyquist (fs/2).")
    
    # Diseño del filtro
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Aplicar el filtro
    y = lfilter(b, a, data)
    return y
