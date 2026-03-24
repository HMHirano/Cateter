import vitaldb
import matplotlib.pyplot as plt
import numpy as np # Often used with vitaldb
import pandas as pd
from scipy import signal
import os
import json
from scipy.signal import find_peaks
import csv
from scipy.signal import lfilter, filtfilt
import librosa

### CONSTANTE

FILE_TO_SAVE_CSV = "step_detection_results"
FOLDER_SAVE = "resumo_testes_DB"
SAMPLE_RATE = 100                                   # Taxa de amostragem em Hz
STEP_WIDTH = "definido na função"                   # Largura do degrau em número de amostras
STEP_AMPLITUDE_FACTOR = 0.8                         # Fator para definir a amplitude do degrau em relação ao valor máximo do sinal
STEP_DETECTION_THRESHOLD = "definido na função"     # Limiar para detectar o degrau na convolução
SIGNAL_THRESHOLD = 200
DERIVATIVE_THRESHOLD_ASCENT = 180 / (0.5 * SAMPLE_RATE)    # Limiar inferior para a derivada, equivale a 180mmHg/0.5s
#DERIVATIVE_THRESHOLD_ASCENT = 0.03                      # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_DESCENT = -0.1                       # Limiar superior para a derivada
dt = 1.0 / SAMPLE_RATE


### INICIALIZAÇÃo

# IDs que possuem SNUADC/ART
# Baixa a lista completa de faixas (tracks) disponíveis na API
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Filtra apenas as linhas onde o nome da faixa é 'SNUADC/ART'
snuadc_art_cases = df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid'].unique()

# Exibe a quantidade e os primeiros 20 IDs como exemplo
print(f"Total de casos com SNUADC/ART: {len(snuadc_art_cases)}")
print(f"Exemplos de IDs: {snuadc_art_cases[:20]}")

# Se quiser salvar em um arquivo:
pd.DataFrame(snuadc_art_cases, columns=['caseid']).to_csv('casos_art.csv')

data_ids = []
    
# for file_number in snuadc_art_cases[4]:
#     print("Executando flush detector da sessão ",file_number)
#     idx,sub,des,asc,desc,conv = flush_detector(file_number, "resultados_detector")
#     data_ids.append(idx)

file_number = snuadc_art_cases[27]
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

### CONVOLUÇÃO
time = [x*dt for x in range(len(signal_raw_filtered))]
time = np.array(time)
period = 1
t = np.linspace(0.001, period, SAMPLE_RATE * period) # quant_pontos =  100pontos/s * tempo da onda(10)

# Caso não haja o sinal no banco
try:
    max(signal_raw_filtered)
except Exception as e:
    print("The error is: ",e)
    print("This caseID doesn't have SNUADC/ART")
    #return

square_wave_LH = signal.square( (-1) * (2 * np.pi / period * t)) * 0.8 * max(signal_raw_filtered)
convolution = signal.fftconvolve(signal_raw_filtered, square_wave_LH, mode="same")
derivative = np.diff(signal_raw_filtered , prepend=signal_raw_filtered[0])

def normalize_robust_and_saturation(sig):
    """
    Normaliza o sinal utilizando o percentil 99.5 dos valores absolutos.
    Ideal para sinais que oscilam entre valores positivos e negativos, como derivadas.
    Melhor para os cálculos ao desconsiderar outliers e parte dos ruídos.
    """
    max_val = np.percentile(np.abs(sig), 95)
    sat_idx_pos = np.where(sig > max_val)[0]
    sat_idx_neg = np.where(sig < -max_val)[0]
    sig[sat_idx_pos] = max_val
    sig[sat_idx_neg] = -max_val
    return sig / max_val if max_val != 0 else sig

### NORMALIZATION
signal_raw_filtered = np.array(signal_raw_filtered / max(signal_raw_filtered))
convolution = np.array(convolution / max(convolution)) #normalize_robust_and_saturation(convolution)
norm_der = np.array(derivative / max(derivative)) #normalize_robust_and_saturation(derivative)

#%%
### ESPECTROGRAMA DO SINAL
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

# Plot the result
times_plot = librosa.times_like(cent, sr=SAMPLE_RATE, hop_length=hop_length)

fig, ax = plt.subplots()

# 1. Change y_axis='log' to y_axis='linear'
# Physiological signals are usually much easier to interpret on a linear frequency scale.
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='linear', x_axis='time', 
                         sr=SAMPLE_RATE, hop_length=hop_length, ax=ax)

ax.plot(times_plot, cent.T, label='Spectral centroid', color='w')

# 2. Set the y-axis limits to focus on the 0 to 20 Hz range
ax.set_ylim([0, 20]) 

ax.legend(loc='upper right')
ax.set(title='Linear Power Spectrogram of ABP Signal (0 - 20 Hz)')
plt.show()


#%%
### CONDIÇÕES
#Subida da pressão
condition_ascent = (convolution < -0.35) & (derivative > DERIVATIVE_THRESHOLD_ASCENT) & (signal_raw_filtered > 0.4)
ascent = np.where(condition_ascent)[0]
abs_ascent_idx = []
if len(ascent) != 0:
    idx_ant = ascent[0]
    for idx in ascent:
        if idx-idx_ant > 5:
            abs_ascent_idx.append(idx_ant)
        elif idx == ascent[-1]:
            abs_ascent_idx.append(idx_ant)
        idx_ant = idx
# rel_ascent_idx = argrelextrema(-convolution[ascent], np.greater)[0]
# abs_ascent_idx = ascent[rel_ascent_idx]

# Descida da pressão
condition_descent = (convolution > 0.2) & (norm_der < DERIVATIVE_THRESHOLD_DESCENT)
descent = np.where(condition_descent)[0]
#rel_descent_idx = argrelextrema(convolution[descent], np.greater)[0]
rel_descent_idx = find_peaks(convolution[descent])[0]
abs_descent_idx = descent[rel_descent_idx]

try:
    with open(f"{folder_name}/derivative.json", "r") as f:
        deri = json.load(f)
except Exception:
    print("No saved data")
    deri = {}
else:
    print("Derivatives dictionary exists and was used")
    

try:
    with open(f"{folder_name}/resultados.csv", "r") as f:
        resultados = pd.read_csv(f, index_col=0)
except Exception:
    print("resultados.csv não existe")
    resultados = pd.DataFrame(columns=["num_flush-like","num_true-flush","num_false_flush"])
else:
    print("resultados.csv existe")


#%%
# Pareia subida com descida, demarcando início e fim do flush
step =[]
condition = []
der_values = []
for asc in abs_ascent_idx:
    for desc in abs_descent_idx:
        if asc < desc:
            print(f"Comparação entre asc:{asc} e desc:{desc}")
            if (desc-asc) > 1*SAMPLE_RATE and (desc-asc) < 45*SAMPLE_RATE: # flush deve durar mais que 3s e menos que 30s
                #condition_stability = np.where(convolution[asc:desc] < 0.6*max(convolution[asc:desc]))[0]
                #print(f"max: {max(convolution[asc:desc])}")
                #condition.append(condition_stability)
                step = step + [[asc,desc]]
                der_values.append(derivative[asc])
                break
            else:
                if (desc-asc) < 1*SAMPLE_RATE:
                    print(f"desc-asc não é maior que 1s: {desc-asc}")
                elif (desc-asc) > 45*SAMPLE_RATE:
                    print(f"desc-asc maior que 45s: {desc-asc}")
                break
        
# Filtra oscilações acentuadas no meio do flush ou ruído simplesmente
step = [ele for ele in step if not np.any(signal_raw_filtered[ele[0]:ele[1]] < 0.3)]

data = {"num_flush-like": len(abs_descent_idx), "num_true-flush": len(step), "num_false_flush":len(abs_descent_idx) - len(step)}
try:
    resultados.loc[f"caseID_{file_number}"] = data
    print(resultados)
except Exception:
    resultados.loc[f"caseID_{file_number}"] = data
    resultados = resultados.sort_index(key=lambda x: x.str.lower())
    print(resultados)
    
resultados.to_csv(f'{folder_name}/resultados.csv')


# DETECTOR DE SINAIS CARDIOGÊNICOS


# VALORES SALVOS DE DERIVADA
deri[f"CASEID_{file_number}"] = [float(value) for value in der_values]
with open(f"{folder_name}/derivative.json", "w") as f:
    json.dump(deri, f)

#%%

# IDENTIFICAÇÃO DOS PARÃMETROS
# Pontos de máximo após a descida
# picos = []
# vales = []
# ksi = []
# ksi_Mp = []
# freq = []
# freq_ts = []
# idx_max = []
# t_spline = np.arange(0,0.5,1 / (SAMPLE_RATE * 5))
# interpolation = []
# # Filter parameters
# order = 5  # Filter order
# cutoff_freq_hz = 20  # Cutoff frequency in Hz
# sampling_rate_hz = SAMPLE_RATE  # Sampling frequency in Hz
# Wn = cutoff_freq_hz / (sampling_rate_hz / 2.0)
# b, a = signal.bessel(order, Wn, btype='low', analog=False, output='ba')
# for sub,des in step:
#     t = time[des:des+50]    
#     resp = signal_raw_filtered[des:des+50]
#     cs = CubicSpline(t, resp)
#     inter = cs(t_spline + time[des])
#     interpolation.append(inter)
#     idx_picos = find_peaks(inter, prominence=0.05)[0] # indices no signal_raw_filtered
#     idx_vales = find_peaks(-inter, prominence=0.05)[0]
#     #if inter[idx_picos[0]] > inter[idx_picos[0]]: # resposta amortecida
#     A1 = (inter[idx_picos[0]] - inter[idx_vales[1]]) * max(signal_raw_filtered)
#     A2 = (inter[idx_picos[1]] - inter[idx_vales[1]]) * max(signal_raw_filtered)
#     ksi.append((-1) * math.log((A1/A2)/np.sqrt((math.pi ** 2) + (math.log(A2/A1)) ** 2)))
#     freq.append(1 / ((idx_vales[1] - idx_vales[0]) / (SAMPLE_RATE * 5))) # Vezes 5 por causa do t_spline
#     #else: # resposta não-amortecida
#     ln_Mp = math.log(abs(inter[idx_vales[0]] - inter[idx_vales[-1]]) * max(signal_raw_filtered)) # Pensando que é uma descida, ordem dos termos inverte
#     ksi_Mp.append( -ln_Mp / (ln_Mp**2 + math.pi**2))
#     freq_ts.append(3 / (ksi_Mp[-1] * (idx_vales[-1] / (SAMPLE_RATE * 5))))
#     picos.append(idx_picos)
#     vales.append(idx_vales)
# print("Ksi medidos", ksi)
# print("Frequências medidas", freq)
# print("Ksi medidos com Mp", ksi_Mp)
# print("Frequências medidas com tempo de subida", freq_ts)
    
#%%

### GRÁFICOS
if not os.path.exists(folder_name):
    os.makedirs(folder_name, exist_ok=True)
    
max_conv = np.where(convolution == max(convolution))[0]

# OVERVIEW
fig, sig = plt.subplots()
sig.plot(time, signal_raw_filtered, label="SNUADC/ART", linewidth=1)
sig.plot(time, convolution, label="convolution", linewidth=1)
sig.plot(time, norm_der, label="derivative", color='r', linewidth=1, alpha=0.5)
sig.set_title(f"Raw signal and Convolution from caseID_{file_number}")
sig.vlines(time[descent], 0, 1, label="descent", color='orchid', alpha=0.1)
sig.vlines(time[ascent], 0, 1, label="ascent", color='tab:olive', alpha=0.2)
sig.scatter(time[max_conv],convolution[max_conv],color='r')
#sig.vlines(time[abs_descent_idx], 0, 1, color='tab:cyan', alpha=1)
sig.legend(loc='upper right')
sig.grid()
fig.show()

file_name = f"SNUADC_ART_{file_number}"
# save_path = folder_name + '/' + f'{file_name}_overview.png'
# fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

#FLUSHES
if len(step) == 0:
    print("There was any flush")
    print(f"CaseID {file_number} existe")

n_per_fig = 3   
for i in range(0, len(step), 3):
    if (len(step) - i) < 3:
        fig, axs = plt.subplots((len(step) - i), 1, layout='constrained', figsize=(20, 15), squeeze=False)
    else:
        fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(20, 15), squeeze=False)
    for j, ax in enumerate(axs.flat):
        idx = i + j
        if idx < len(step):
            start_pos = max(0, step[idx][1] - 20*SAMPLE_RATE)
            end_pos = min(len(signal_raw_filtered), step[idx][1] + 20*SAMPLE_RATE)
            t = time[start_pos:end_pos]
            ax.plot(time[start_pos:end_pos], signal_raw_filtered[start_pos:end_pos], label='signal', linewidth=1)
            ax.plot(time[start_pos:end_pos], convolution[start_pos:end_pos], label='convolution', linewidth=1)
            ax.plot(time[start_pos:end_pos], norm_der[start_pos:end_pos], label='norm_der', linewidth=1)
            ax.vlines(time[step[idx][1]],0,1)
            # time_spline = t_spline + time[step[idx][1]]
            # time_resp = time[step[idx][1]:step[idx][1]+50]
            # ax.scatter(time_spline[picos[idx]], interpolation[idx][picos[idx]], label='Picos')
            # ax.scatter(time_spline[vales[idx]], interpolation[idx][vales[idx]], label='Vales', color='tab:olive')
            # ax.plot(time_spline, interpolation[idx], color='tab:red', label='cubicspline')
            ax.set_title(f'Flush {i + j} at time {time[step[idx][0]]:.2f} s max derivative: {max(derivative):.2f}')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')
            ax.grid()
            save_flush = folder_name + '/' + f'{file_name}_flushtest_{idx}.csv'
            data = [t,signal_raw_filtered[start_pos:end_pos]]
            with open(save_flush, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)   
    figure_number = i // 3
    fig.show()
    save_figures = folder_name + '/' + f'{file_name}_flushes_{figure_number}.png'
    fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)
    
    


print(f"CaseID {file_number} existe")
    


