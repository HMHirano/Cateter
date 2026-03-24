import vitaldb
import matplotlib.pyplot as plt
import numpy as np # Often used with vitaldb
import pandas as pd
from scipy import signal
from scipy.signal import argrelextrema
import sys
#import pandas as pd # Also commonly used
#import os
#import csv

'''
array = [x for x in range(10)]
print(array)
array2 = [y for y in range(5,15)]
print(array2)
array3 = [array[z] for z in range(len(array)) if array[z] in array2]
print(array3)
array4 = [array[x] > array2[y] for x in range(len(array)) for y in range(len(array2))]
print(array4)
print(array3[1:3])
'''

FILE_TO_SAVE_CSV = "step_detection_results"
FOLDER_SAVE = "resumo_testes_DB"
SAMPLE_RATE = 100                                   # Taxa de amostragem em Hz
STEP_WIDTH = "definido na função"                   # Largura do degrau em número de amostras
STEP_AMPLITUDE_FACTOR = 0.8                         # Fator para definir a amplitude do degrau em relação ao valor máximo do sinal
STEP_DETECTION_THRESHOLD = "definido na função"     # Limiar para detectar o degrau na convolução
SIGNAL_THRESHOLD = 200

# Descida
DERIVATIVE_THRESHOLD_LOW = 0.1                      # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_HIGH = 0                       # Limiar superior para a derivada

dt = 1.0 / SAMPLE_RATE
track_names = ['SNUADC/ART']
vf = vitaldb.VitalFile(3, track_names)
samples = vf.to_numpy(track_names, dt)
signal_raw = samples[:,0]
signal_raw_filtered = signal_raw[ ~np.isnan(signal_raw)]
print("Signal Filtered")
print(f"Shape: {signal_raw_filtered.shape} Length: {len(signal_raw_filtered)}")
time = [x*dt for x in range(len(signal_raw_filtered))]
time = np.array(time)
period = 1
t = np.linspace(0, period - 0.001, SAMPLE_RATE * period) # quant_pontos =  100pontos/s * tempo da onda(10)
t2 = np.linspace(0.001, period, SAMPLE_RATE * period) # quant_pontos =  100pontos/s * tempo da onda(10)
square_wave_HL = signal.square(2 * np.pi / period * t) * 0.8 * max(signal_raw_filtered)
square_wave_LH = signal.square( (-1) * (2 * np.pi / period * t2)) * 0.8 * max(signal_raw_filtered)
convolution = signal.fftconvolve(signal_raw_filtered, square_wave_HL, mode="same")
convolution2 = signal.fftconvolve(signal_raw_filtered, square_wave_LH, mode="same")
convolution = np.array(convolution / max(convolution))
convolution2 = np.array(convolution2 / max(convolution2))
derivative = np.diff(signal_raw_filtered , prepend=signal_raw_filtered[0])
derivative = derivative / max(derivative)
signal_raw_filtered = signal_raw_filtered / max(signal_raw_filtered)
print("Convolution")
print(f"Sample shape: {convolution.shape} Samples length: {len(convolution)}")

condition_ascent = (convolution2 < -0.35) & (derivative > DERIVATIVE_THRESHOLD_LOW) & (signal_raw_filtered > 0.4)
ascent = np.where(condition_ascent)[0]

eventos = []  # Lista para acumular os dados

# Seleciona a região com um alto valor de convolução da onda quadrada HIGH-LOW
if len(ascent) > 0:
    start_idx = ascent[0]
    previous_idx = ascent[0]
    region_count = 1

    # Começamos do segundo elemento, pois já inicializamos com o primeiro
    for idx in ascent[1:]:
        if (idx - previous_idx) > 1:
            # Se houve um salto no índice, encerramos o evento anterior
            eventos.append({
                "Region": region_count,
                "begin": time[start_idx],
                "end": time[previous_idx] # O fim é o último índice contínuo
            })
            
            # Iniciamos um novo evento
            start_idx = idx
            region_count += 1
            
        previous_idx = idx

    # Adiciona o último evento que ficou pendente após o loop terminar
    eventos.append({
        "Region": region_count,
        "begin": time[start_idx],
        "end": time[previous_idx]
    })

# Cria o DataFrame de uma só vez (muito mais rápido)
df_resultado = pd.DataFrame(eventos)

print(df_resultado)
        
rel_ascent_idx = argrelextrema(-convolution2[ascent], np.greater)[0]
abs_ascent_idx = ascent[rel_ascent_idx]

descent = np.where(convolution2 > 0.35)[0]

rel_descent_idx = argrelextrema(convolution2[descent], np.greater)[0]
abs_descent_idx = descent[rel_descent_idx]

step =[]
for asc in abs_ascent_idx:
    for desc in abs_descent_idx:
        if asc < desc:
            step = step + [[asc,desc]]
            break

# plt.figure()
# plt.plot(time[descent],convolution2[descent])
# plt.vlines(time[step_idx], 0, 1, color ='r', alpha = 0.1)
# plt.show()

print(f"Peaks time: {time[abs_descent_idx]}, Peaks values: {convolution2[abs_descent_idx]}")

if len(step) == 0:
    print("Exiting program due to condition.")
    sys.exit(0) # Exit with status code 0 (success)

fig, axes = plt.subplots(len(step), 1, layout='constrained', figsize=(10, 5*len(step)))

for ax, idx in zip(axes.flatten(), step):
    start_pos = max(0, idx[0] - 1000)
    end_pos = min(len(signal_raw_filtered), idx[1] + 4000)
    ax.plot(time[start_pos:end_pos], signal_raw_filtered[start_pos:end_pos], label=f'Instant of flush {time[idx[0]]:.2f} s', linewidth=1)
    ax.plot(time[start_pos:end_pos], convolution2[start_pos:end_pos], linewidth=1)
    # ax.set_title(f'Segment around detected step at time {time[idx]:.2f} s')
    # ax.set_ylabel('Amplitude')
    ax.grid()
fig.show()

fig, (sqwv1, sqwv2) = plt.subplots(2,1,layout='constrained')
sqwv1.plot([x for x in range(len(square_wave_HL))],square_wave_HL)
sqwv1.set_xlabel('Time (s)')
sqwv1.set_title('Square Wave HL')
sqwv1.grid()
sqwv2.plot(t2,square_wave_LH)
sqwv2.set_xlabel('Time (s)')
sqwv2.set_title('Square Wave LH')
sqwv2.grid()
fig.show()

fig2, (sig, sig2) = plt.subplots(2,1, layout='constrained')
sig.plot(time, signal_raw_filtered)
sig.plot(time, convolution)
sig.plot(time, derivative)
sig.set_title("Raw signal and Convolution with L-H")
sig.vlines(time[descent], 0, 1, color='r', alpha=0.1)
sig.grid()
sig2.plot(time, signal_raw_filtered, linewidth=1)
sig2.plot(time, convolution2)
sig2.plot(time, derivative, color='r')
sig2.set_title("Raw signal and Convolution with H-L")
sig2.vlines(time[descent], 0, 1, color='tab:green', alpha=0.1)
sig2.vlines(time[ascent], 0, 1, color='tab:olive', alpha=0.1)
sig2.vlines(time[abs_ascent_idx], 0, 1, color='m', alpha=1)
sig2.grid()
fig2.show()

