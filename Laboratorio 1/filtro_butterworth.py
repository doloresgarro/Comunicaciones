from scipy.signal import butter, lfilter

def aplicar_filtro_butter(data, cutoff, fs, orden=5):
    """
    Aplica un filtro Butterworth de orden 5 a los datos proporcionados.

    Parameters:
        data (numpy array): Señal de entrada.
        cutoff (float): Frecuencia de corte (Hz).
        fs (float): Frecuencia de muestreo (Hz).
        orden (int): Orden del filtro.

    Returns:
        numpy array: Señal filtrada.
    """
    nyquist = fs / 2  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Normalización de la frecuencia de corte
    if not 0 < cutoff < fs / 2:
     raise ValueError(f"Frecuencia de corte inválida: {cutoff}. Debe estar entre 0 y {fs / 2} Hz.")

    # Diseño del filtro
    b, a = butter(orden, normal_cutoff, btype='low', analog=False)
    
    # Aplicar el filtro
    y = lfilter(b, a, data)
    return y
