#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 09:28:46 2026

@author: heitor
"""

import matplotlib.pyplot as plt
import numpy as np

# Exemplo: Limitar a 4 subplots por figura
dados = range(10) # 10 gráficos totais
n_por_fig = 4

for i in range(0, len(dados), n_por_fig):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) # Cria nova figura para cada grupo
    for j, ax in enumerate(axs.flat):
        idx = i + j
        if idx < len(dados):
            ax.plot(np.random.rand(10))
            ax.set_title(f'Gráfico {idx+1}')
        else:
            ax.axis('off') # Esconde subplots vazios
    plt.tight_layout()
plt.show()

import pandas as pd

d = {"col1": [1, 2, 5], "col2": [3, 4, 6], "col3":[7, 8, 9]}
df = pd.DataFrame(data=d, index=[1, 2, 3])
df2 = pd.DataFrame(data=d, index=[3])
print(df)
print(df2)

import pandas as pd
data = {"City": ["NY", "LA"], "Population": [8_399_000, 3_990_000]}
df = pd.DataFrame(data)

# Insert a row in the middle
df.loc[0.5] = ["Chicago", 2_705_000]  # Add at a floating-point index
print(df)
df = df.sort_index().reset_index(drop=True)
print(df)

a = [10, 20, 30, 40, 50]

# Find the index of the element 30 in the list 'a'
idx = a.index(45)
print(idx)

df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=["A2", "A4", "A1", "d"])
df.loc["A3"] = 5
df.sort_index(key=lambda x: x.str.lower())

b_vector = [True, False, False]
print(np.any(b_vector))

fig, axs = plt.subplots(1,1)
axs.plot(time[ascent],convolution[ascent])
fig.show()

fig, axs = plt.subplots(1,3)
for idx in range(0, len(step)):
    time_spline = t_spline + time[step[idx][1]]
    axs[idx].scatter(time_spline[idx_picos], interpolation[idx][idx_picos], label='Picos para encontrar param')
    axs[idx].plot(time_spline, interpolation[idx], color='tab:red', label='cubicspline')
fig.show()

#%%%
# WAVELET TRANSFORM

import pywt
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate a sample signal (e.g., a signal with two different frequency bursts)
duration = 1
fs = 1000  # Sampling frequency in Hz
t = np.linspace(0, duration, fs, endpoint=False)
# Create a signal with a 20 Hz burst and a 60 Hz burst
signal = np.sin(2 * np.pi * 20 * t) * (t > 0.2) * (t < 0.4) + \
         np.sin(2 * np.pi * 60 * t) * (t > 0.6) * (t < 0.8)

# 2. Define frequencies and scales
# The relationship between scales and frequencies is complex, often requires careful handling
# You can use the scale2frequency function if needed.
widths = np.arange(1, 128) # Varying widths (scales)

# 3. Perform the Continuous Wavelet Transform
# 'morl' refers to the Morlet wavelet
cwtmatr, freqs = pywt.cwt(signal, widths, 'morl')

# 4. Plot the results (magnitude of the transform)
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(cwtmatr), extent=[0, duration, freqs[-1], freqs[0]], aspect='auto',
           cmap='coolwarm', origin='upper')
plt.title("Morlet Wavelet Transform (CWT) Scalogram")
plt.ylabel("Frequency (approx. related to scale)")
plt.xlabel("Time (s)")
plt.colorbar(label="Magnitude")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.show()

#%%%
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Parameters ---
wavelet_name = 'cmor1.5-1.0' 
level = 10 

# --- Generate the wavelet function and x-values ---
[psi, xval] = pywt.ContinuousWavelet(wavelet_name).wavefun(level)
real_psi = np.real(psi)

# --- Find peaks using Prominence ---
# Calculate the maximum amplitude in the signal
max_amp = np.max(real_psi)

# Set prominence to 10% of the maximum amplitude.
# This ignores tiny ripples and floating point noise, but keeps the main lobes.
min_prominence = max_amp * 0.05 

# The function returns the peak indices, and a dictionary of properties (which we catch in 'props')
peaks, props = find_peaks(real_psi, prominence=min_prominence)

# --- Plot the real and imaginary parts ---
plt.figure()
plt.plot(xval, real_psi, label="real")
plt.plot(xval, np.imag(psi), label="imag")

# Scatter plot the filtered peaks
plt.scatter(xval[peaks], real_psi[peaks], color='red', zorder=5, label='Prominent Peaks')

plt.title(f'Complex Morlet Wavelet "{wavelet_name}"\n(Prominence Filtered)')
plt.xlabel('Time (xval)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

#%%

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Filter parameters
order = 5  # Filter order
cutoff_freq_hz = 30  # Cutoff frequency in Hz
sampling_rate_hz = 100  # Sampling frequency in Hz

# Normalize cutoff frequency to Nyquist frequency (0 to 1, where 1 is Nyquist)
# For digital filters, Wn is in the same units as fs
Wn = cutoff_freq_hz / (sampling_rate_hz / 2.0)

# Design the digital low-pass Bessel filter
# Returns numerator (b) and denominator (a) polynomials
b, a = signal.bessel(order, Wn, btype='low', analog=False, output='ba')

# Optional: Use second-order sections (sos) for better numerical stability
# sos = signal.bessel(order, Wn, btype='low', analog=False, output='sos')

from scipy.signal import lfilter, filtfilt

# Example: Generate a noisy signal
t = np.linspace(0, 0.25, sampling_rate_hz, endpoint=False)
# Signal with 50 Hz and 1500 Hz components
input_signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 40 * t)

# Apply the filter
# Using filtfilt for zero phase shift
filtered_signal = filtfilt(b, a, input_signal)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t, input_signal, 'b-', label='Original Signal')
plt.plot(t, filtered_signal, 'g-', linewidth=2, label='Filtered Signal (Bessel)')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.title('Bessel Filter Application in Python')
plt.legend()
plt.grid(True)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def gerar_sistema_segunda_ordem(zeta, wn):
    """
    Gera uma função de transferência de segunda ordem.
    
    Parâmetros:
    zeta (float): Coeficiente de amortecimento
    wn (float): Frequência natural não amortecida (rad/s)
    
    Retorna:
    scipy.signal.TransferFunction: O sistema contínuo gerado
    """
    # Numerador: wn^2
    numerador = [wn**2]
    
    # Denominador: 1*s^2 + 2*zeta*wn*s + wn^2
    denominador = [1, 2 * zeta * wn, wn**2]
    
    # Cria o objeto da função de transferência
    sistema = signal.TransferFunction(numerador, denominador)
    
    return sistema

# ==========================================
# Exemplo de Uso
# ==========================================

if __name__ == "__main__":
    # Definindo os parâmetros
    zeta_exemplo = 0.3  # Sistema subamortecido (0 < zeta < 1)
    wn_exemplo = 10.0   # rad/s

    # 1. Gerando a função de transferência
    meu_sistema = gerar_sistema_segunda_ordem(zeta_exemplo, wn_exemplo)
    print("Numerador:", meu_sistema.num)
    print("Denominador:", meu_sistema.den)

    # 2. Calculando a resposta ao degrau unitário
    tempo, resposta = signal.step(meu_sistema)

    # 3. Plotando o gráfico da resposta ao degrau
    plt.figure(figsize=(8, 5))
    plt.plot(tempo, resposta, color='blue', linewidth=2, 
             label=f'$\\zeta={zeta_exemplo}$, $\\omega_n={wn_exemplo}$')
    
    # Linha de referência (valor de regime permanente)
    plt.axhline(y=1, color='r', linestyle='--', label='Referência (Degrau)')
    
    # Configurações do gráfico
    plt.title('Resposta ao Degrau - Sistema de 2ª Ordem')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    plt.show()
    
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def gerar_sistema_segunda_ordem(zeta, wn):
    """Gera o sistema de segunda ordem usando a Função de Transferência."""
    numerador = [wn**2]
    denominador = [1, 2 * zeta * wn, wn**2]
    return signal.TransferFunction(numerador, denominador)

if __name__ == "__main__":
    # 1. Definindo o sistema e o sinal de entrada
    zeta = 0.3
    wn = 10.0
    meu_sistema = gerar_sistema_segunda_ordem(zeta, wn)

    tempo = np.linspace(0, 5, 1000)
    sinal_entrada = np.sin(2 * tempo) + 0.5 * np.sin(50 * tempo)

    # 2. Simulando o sistema e capturando a saída e os estados
    tempo_out, sinal_saida, estado_interno = signal.lsim(meu_sistema, U=sinal_entrada, T=tempo)

    # 3. Criando a figura com 2 subgráficos
    plt.figure(figsize=(10, 8))

    # --- GRÁFICO 1: Visão Externa (Entrada e Saída) ---
    plt.subplot(2, 1, 1)
    plt.plot(tempo, sinal_entrada, color='gray', alpha=0.6, label='Sinal de Entrada')
    plt.plot(tempo_out, sinal_saida, color='blue', linewidth=2, label='Sinal de Saída')
    plt.title('Visão Externa: O que entra e o que sai do sistema')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # --- GRÁFICO 2: Visão Interna (Variáveis de Estado) ---
    plt.subplot(2, 1, 2)
    # estado_interno[:, 0] extrai todas as linhas da coluna 0 (Estado 1)
    plt.plot(tempo_out, estado_interno[:, 0], color='orange', label='Estado 1 ($x_1$)')
    # estado_interno[:, 1] extrai todas as linhas da coluna 1 (Estado 2)
    plt.plot(tempo_out, estado_interno[:, 1], color='green', label='Estado 2 ($x_2$)')
    plt.title('Visão Interna: Dinâmica das Variáveis de Estado')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Amplitude dos Estados')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Ajusta o espaçamento para não sobrepor os textos
    plt.tight_layout()
    plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def gerar_sistema_segunda_ordem(zeta, wn):
    """Gera o sistema de segunda ordem original."""
    numerador = [wn**2]
    denominador = [1, 2 * zeta * wn, wn**2]
    return signal.TransferFunction(numerador, denominador)

if __name__ == "__main__":
    # 1. Definindo o sistema original
    zeta = 0.3
    wn = 10.0
    meu_sistema = gerar_sistema_segunda_ordem(zeta, wn)

    # 2. Criando o sistema inverso (Invertendo numerador e denominador)
    num_inverso = meu_sistema.den
    den_inverso = meu_sistema.num
    sistema_inverso = signal.TransferFunction(num_inverso, den_inverso)

    # 3. Multiplicando os dois sistemas (Original * Inverso)
    # Na SciPy, multiplicamos os polinômios usando np.polymul
    num_combinado = np.polymul(meu_sistema.num, sistema_inverso.num)
    den_combinado = np.polymul(meu_sistema.den, sistema_inverso.den)
    sistema_combinado = signal.TransferFunction(num_combinado, den_combinado)

    # 4. Calculando o Diagrama de Bode de ambos
    w_orig, mag_orig, fase_orig = signal.bode(meu_sistema)
    
    # Forçamos o sistema combinado a usar as mesmas frequências do original
    w_comb, mag_comb, fase_comb = signal.bode(sistema_combinado, w=w_orig) 

    # 5. Plotando os gráficos comparativos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Gráfico de Magnitude ---
    ax1.semilogx(w_orig, mag_orig, color='blue', linewidth=2, label='Sistema Original $H(s)$')
    ax1.semilogx(w_comb, mag_comb, color='red', linestyle='--', linewidth=3, label='Combinado $H(s) \\cdot H^{-1}(s)$')
    
    ax1.set_title('Diagrama de Bode - Original vs. Combinado (Cancelamento)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend()

    # --- Gráfico de Fase ---
    ax2.semilogx(w_orig, fase_orig, color='green', linewidth=2, label='Sistema Original')
    ax2.semilogx(w_comb, fase_comb, color='orange', linestyle='--', linewidth=3, label='Combinado')
    
    ax2.set_xlabel('Frequência (rad/s)')
    ax2.set_ylabel('Fase (graus)')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
#%%

import numpy as np
import vitaldb
import pandas as pd
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 100
dt = 1.0 / SAMPLE_RATE

# IDs que possuem SNUADC/ART
# Baixa a lista completa de faixas (tracks) disponíveis na API
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Filtra apenas as linhas onde o nome da faixa é 'SNUADC/ART'
snuadc_art_cases = df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid'].unique()

file_number = snuadc_art_cases[3]
folder_name = "resultados_detector"
print("Executando flush detector da sessão ",file_number)

### SINAIS
track_names = ['SNUADC/ART']
try:
    vf = vitaldb.VitalFile(file_number, track_names)
except Exception as e:
    print("The error is: ",e)
    print("This caseID doesn't have SNUADC/ART")
    #return

samples = vf.to_numpy(track_names, dt)
signal_raw = samples[:,0]
signal_raw_filtered = signal_raw[ ~np.isnan(signal_raw)]

# From time-series input:
y = signal_raw_filtered
#y, sr = librosa.load(librosa.ex('trumpet'))

# Set your window size
n_fft = 2 * SAMPLE_RATE  # 200 samples for a 2-second window

# Optionally, you can set hop_length to control the overlap.
# Default is n_fft // 4 (which would be 50 samples, or 0.5 seconds).
hop_length = n_fft // 4 

# 1. From time-series input:
cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)

# 2. From spectrogram input (Make sure STFT uses the same n_fft!):
S, phase = librosa.magphase(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)

# 3. Using reassigned spectrogram:
freqs, times, D = librosa.reassigned_spectrogram(y, sr=SAMPLE_RATE, fill_nan=True, n_fft=n_fft, hop_length=hop_length)
librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)

cent_1d = cent[0]
window_size = 5
movingAvg_cent = np.convolve(cent[0], np.ones(window_size)/window_size, mode='same')
# Center=True ensures the average aligns perfectly with the current time step
movingAvg_cent = pd.Series(cent_1d).rolling(window=window_size, min_periods=1, center=True).mean().values

# --- FLUSH DETECTION ---
threshold = 0.5 # Hz
min_duration_frames = 2 # Minimum time steps (2 frames = 1 second at hop=50)

squareWaveLike_indices = []
in_wave = False
start_idx = 0

# Loop through the smoothed moving average
for idx, val in enumerate(movingAvg_cent):
    if val <= threshold and not in_wave:
        # Signal dropped below threshold: start of a flush
        in_wave = True
        start_idx = idx
        
    elif val > threshold and in_wave:
        # Signal went back above threshold: end of a flush
        in_wave = False
        end_idx = idx
        
        # Only save it if the drop lasted longer than our minimum duration
        if (end_idx - start_idx) >= min_duration_frames:
            squareWaveLike_indices.append([start_idx, end_idx])

# Catch the edge case where the file ends while still in a flush
if in_wave:
    squareWaveLike_indices.append([start_idx, len(movingAvg_cent) - 1])

print(f"Detected {len(squareWaveLike_indices)} flush(es).")
        
cent_1d = cent[0]
window_size = 5
movingAvg_cent = np.convolve(cent[0], np.ones(window_size)/window_size, mode='same')
# Center=True ensures the average aligns perfectly with the current time step
movingAvg_cent = pd.Series(cent_1d).rolling(window=window_size, min_periods=1, center=True).mean().values

# Plot the result
times_plot = librosa.times_like(cent, sr=SAMPLE_RATE, hop_length=hop_length)

fig, ax = plt.subplots()

# 1. Change y_axis='log' to y_axis='linear'
# Physiological signals are usually much easier to interpret on a linear frequency scale.
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='linear', x_axis='time', 
                         sr=SAMPLE_RATE, hop_length=hop_length, ax=ax)

# Plot the raw spectral centroid (thinner/transparent so it doesn't overpower the graph)
ax.plot(times_plot, cent_1d, label='Raw Spectral Centroid', color='w', alpha=0.4)

# Plot the moving average centroid (thicker line to stand out)
ax.plot(times_plot, movingAvg_cent, label=f'Moving Average (window={window_size})', color='g', linewidth=2)

# Set the y-axis limits to focus on the 0 to 20 Hz range
ax.set_ylim([0, 20]) 

ax.legend(loc='upper right')
ax.set(title='Linear Power Spectrogram of ABP Signal (0 - 20 Hz)')
plt.show()

# --- PLOTTING FLUSHES ---
if len(squareWaveLike_indices) == 0:
    print(f"No flushes detected for CaseID {file_number}.")
else:
    # We will plot up to 3 flushes per figure, as in your original code
    for i in range(0, len(squareWaveLike_indices), 3):
        
        # Determine how many rows this specific figure needs (up to 3)
        rows_this_fig = min(3, len(squareWaveLike_indices) - i)
        fig, axs = plt.subplots(rows_this_fig, 1, layout='constrained', figsize=(15, 5 * rows_this_fig), squeeze=False)
        
        for j, ax in enumerate(axs.flat):
            idx = i + j
            if idx < len(squareWaveLike_indices):
                # Unpack the start and end indices of the flush
                flush_start_idx, flush_end_idx = squareWaveLike_indices[idx]
                
                # Convert the STFT frame index back to raw signal sample index
                # frame_index * hop_length = raw_sample_index
                raw_start = flush_start_idx * hop_length
                raw_end = flush_end_idx * hop_length
                
                # Create a viewing window (e.g., 10 seconds before and after the flush)
                window_samples = 10 * SAMPLE_RATE
                view_start = max(0, raw_start - window_samples)
                view_end = min(len(signal_raw_filtered), raw_end + window_samples)
                
                # Time array for the raw signal slice
                t_slice = np.arange(view_start, view_end) / SAMPLE_RATE
                signal_slice = signal_raw_filtered[view_start:view_end]
                
                ax.plot(t_slice, signal_slice, label='ABP Signal', linewidth=1, color='b')
                
                # Draw vertical lines marking the exact start and end of the detected flush
                ax.axvline(x=(raw_start / SAMPLE_RATE), color='r', linestyle='--', label='Flush Start')
                ax.axvline(x=(raw_end / SAMPLE_RATE), color='g', linestyle='--', label='Flush End')
                
                ax.set_title(f'Flush {idx + 1} (Duration: {times_plot[flush_end_idx] - times_plot[flush_start_idx]:.2f}s)')
                ax.set_ylabel('Amplitude (mmHg)')
                ax.set_xlabel('Time (s)')
                ax.legend(loc='upper right')
                ax.grid(True)
                
        plt.show()


#%%

t = np.linspace(0,20,100)
x = list(range(20))
y = np.cos(t)
plt.plot(t[x],y[x])
plt.show()
    
    