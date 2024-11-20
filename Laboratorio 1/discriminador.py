import numpy as np

def demodular_fm(signal, fs):
    """
    Realiza la demodulación de FM utilizando un discriminador basado en la derivada de la fase.

    Parameters:
        signal (numpy array): Señal compleja en banda base (IQ).
        fs (float): Frecuencia de muestreo (Hz).

    Returns:
        numpy array: Señal demodulada (audio).
    """
    # Calcular la fase de la señal
    phase = np.unwrap(np.angle(signal))
    
    # Calcular la derivada discreta de la fase
    delta_phase = np.diff(phase)
    
    # Estimar el mensaje demodulado
    demodulated_signal = delta_phase / (2 * np.pi)
    
    # Ajustar la longitud de la señal
    return demodulated_signal
