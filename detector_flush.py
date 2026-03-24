#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:21:03 2026

@author: heitor
"""

import vitaldb
import matplotlib.pyplot as plt
import numpy as np # Often used with vitaldb
import pandas as pd
from scipy import signal
from scipy.signal import argrelextrema
import os
import json

### CONSTANTE

FILE_TO_SAVE_CSV = "step_detection_results"
FOLDER_SAVE = "resumo_testes_DB"
SAMPLE_RATE = 100                                   # Taxa de amostragem em Hz
STEP_WIDTH = "definido na função"                   # Largura do degrau em número de amostras
STEP_AMPLITUDE_FACTOR = 0.8                         # Fator para definir a amplitude do degrau em relação ao valor máximo do sinal
STEP_DETECTION_THRESHOLD = "definido na função"     # Limiar para detectar o degrau na convolução
SIGNAL_THRESHOLD = 200
DERIVATIVE_THRESHOLD_LOW = 15                       # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_HIGH = 0                       # Limiar superior para a derivada
dt = 1.0 / SAMPLE_RATE

def flush_detector(file_number, folder_name):
    ### SINAIS
    track_names = ['SNUADC/ART']
    try:
        vf = vitaldb.VitalFile(file_number, track_names)
    except Exception as e:
        print("The error is: ",e)
        print("This caseID doesn't have SNUADC/ART")
        return
    
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
        return
    
    square_wave_LH = signal.square( (-1) * (2 * np.pi / period * t)) * 0.8 * max(signal_raw_filtered)
    convolution = signal.fftconvolve(signal_raw_filtered, square_wave_LH, mode="same")
    derivative = np.diff(signal_raw_filtered , prepend=signal_raw_filtered[0])
    
    ### NORMALIZATION
    signal_raw_filtered = signal_raw_filtered / max(signal_raw_filtered)
    convolution = np.array(convolution / max(convolution))
    norm_der = derivative / max(derivative)
    
    ### CONDIÇÕES
    # Subida da pressão
    condition_ascent = (convolution < -0.35) & (derivative > DERIVATIVE_THRESHOLD_LOW) & (signal_raw_filtered > 0.4)
    ascent = np.where(condition_ascent)[0]
    abs_ascent_idx = []
    if len(ascent) != 0:
        idx_ant = ascent[0]
        for idx in ascent:
            if idx-idx_ant > 100:
                abs_ascent_idx.append(idx_ant)
            elif idx == ascent[-1]:
                abs_ascent_idx.append(idx_ant)
            idx_ant = idx
    #rel_ascent_idx = argrelextrema(-convolution[ascent], np.greater)[0]
    #abs_ascent_idx = ascent[rel_ascent_idx]
    
    # Descida da pressão
    condition_descent = convolution > 0.35
    descent = np.where(condition_descent)[0]
    rel_descent_idx = argrelextrema(convolution[descent], np.greater)[0]
    abs_descent_idx = descent[rel_descent_idx]
    
    
    # Dicionário salva valores utilizados nas derivadas da subida
    try:
        with open("derivative.json", "r") as f:
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
        resultados = pd.DataFrame(columns=["num_flush-like","num_true-flush","num_false_flush","possiveis_flush"])
    else:
        print("resultados.csv existe")
    
    # Pareia subida com descida, demarcando início e fim do flush
    step =[]
    der_values = []
    for asc in abs_ascent_idx:
        for desc in abs_descent_idx:
            if asc < desc:
                print(f"Comparação entre asc:{asc} e desc:{desc} e dif:{desc-asc}")
                # CONDIÇÃO DA DURAÇÃO DO FLUSH
                if (desc-asc) > 3*SAMPLE_RATE and (desc-asc) < 45*SAMPLE_RATE: # flush deve durar mais que 3s e menos que 12s
                    #if len(step) == 0:
                    step = step + [[asc,desc]]
                    der_values.append(derivative[asc])
                    # Verifica se uma mesma descida é utilizada por duas subidas diferentes
                    # elif asc > step[-1][1]: 
                    #     step = step + [[asc,desc]]
                    #     der_values.append(derivative[asc])
                    break
                else:
                    if (desc-asc) < 3*SAMPLE_RATE:
                        print(f"desc-asc não é maior que 3s: {desc-asc}")
                    elif (desc-asc) > 45*SAMPLE_RATE:
                        print(f"desc-asc maior que 45s: {desc-asc}")
                    break
                
    # Oscilações indesejadas
    # Create a boolean mask to keep track of which rows are valid
    # 1. Ensure step is a NumPy array so it supports Boolean Masking
    step_arr = np.array(step)

    keep_mask = []

    for start, end in step_arr:
        segment = signal_raw_filtered[start:end]
        # Keep if NO values are under 0.3
        is_valid = not np.any(segment < 0.3)
        keep_mask.append(is_valid)

    # 2. Now you can filter using the mask because step_arr is a NumPy object
    filtered_step = step_arr[keep_mask]

    # 3. If you specifically need it back as a list:
    step = filtered_step.tolist()
    
                
    deri[f"CASEID_{file_number}"] = [float(value) for value in der_values]
    with open("derivative.json", "w") as f:
        json.dump(deri, f)
        
    num_flush_like = len(abs_descent_idx)
    num_true_flush = len(step)
    num_false_flush = num_flush_like -  len(abs_ascent_idx)
    poss_flush = len(abs_ascent_idx) - num_true_flush
    data = {"num_flush-like": num_flush_like, "num_true-flush": num_true_flush, "num_false_flush":num_false_flush, "possiveis_flush":poss_flush} 
    try:
        resultados.loc[f"caseID_{file_number}"] = data
        print(resultados)
    except Exception:
        resultados.loc[f"caseID_{file_number}"] = data
        resultados = resultados.sort_index(key=lambda x: x.str.lower())
        print(resultados)
        
    resultados.to_csv(f'{folder_name}/resultados.csv')
        
    # total = []
    # for i in deri:
    #     #print(i)
    #     #print(deri[f"{i}"])
    #     total = np.concatenate((total, deri[f"{i}"]), axis=0)
    #     print(total)
    # media = np.mean(total)
    # print("Média dos valores das derivadas da subida {media:.2f}")

    ### GRÁFICOS
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    
    # OVERVIEW
    fig, sig = plt.subplots(figsize=(20, 15))
    sig.plot(time, signal_raw_filtered, label="SNUADC/ART", linewidth=1)
    sig.plot(time, convolution, label="convolution", linewidth=1)
    sig.plot(time, norm_der, label="derivative", color='r', linewidth=1)
    sig.set_title(f"Raw signal and Convolution from caseID_{file_number}")
    sig.vlines(time[descent], 0, 1, label="descent", color='tab:green', alpha=0.1)
    sig.vlines(time[ascent], 0, 1, label="ascent", color='tab:olive', alpha=0.5)
    sig.vlines(time[abs_descent_idx], 0, 1, color='tab:cyan', alpha=1)
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
        return file_number,abs_ascent_idx,abs_descent_idx,ascent,descent,convolution, resultados
    
    n_per_fig = 3   
    for i in range(0, len(step), n_per_fig):
        if (len(step) - i) < n_per_fig:
            fig, axs = plt.subplots((len(step) - i), 1, layout='constrained', figsize=(20, 15), squeeze=False)
        else:
            fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(20, 15), squeeze=False)
        for j, ax in enumerate(axs.flat):
            idx = i + j
            if idx < len(step):
                start_pos = max(0, step[idx][1] - 20*SAMPLE_RATE)
                end_pos = min(len(signal_raw_filtered), step[idx][1] + 20*SAMPLE_RATE)
                t = time[start_pos:end_pos]
                ax.plot(t, signal_raw_filtered[start_pos:end_pos], label=f'Instant of flush {time[step[idx][0]]:.2f} s', linewidth=1)
                ax.plot(t, convolution[start_pos:end_pos], linewidth=1)
                ax.plot(t, norm_der[start_pos:end_pos], linewidth=1)
                ax.hlines(2 * np.mean(convolution[start_pos:end_pos]), time[start_pos], time[end_pos], label="2 médias")
                ax.set_title(f'Flush {i + j} at time {time[step[idx][0]]:.2f} s max derivative: {max(derivative):.2f}')
                ax.set_ylabel('Amplitude')
                ax.legend(loc='upper right')
                ax.grid()
        figure_number = i // 3
        #fig.show()
        save_figures = folder_name + '/' + f'{file_name}_flushes_{figure_number}.png'
        fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)

    
    print(f"CaseID {file_number} existe")
    return file_number,abs_ascent_idx,abs_descent_idx,ascent,descent,convolution, resultados
    
# IDs que possuem SNUADC/ART
# Baixa a lista completa de faixas (tracks) disponíveis na API
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Filtra apenas as linhas onde o nome da faixa é 'SNUADC/ART'
snuadc_art_cases = df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid'].unique()

# Exibe a quantidade e os primeiros 20 IDs como exemplo
# print(f"Total de casos com SNUADC/ART: {len(snuadc_art_cases)}")
# print(f"Exemplos de IDs: {snuadc_art_cases[:20]}")

# Se quiser salvar em um arquivo:
# pd.DataFrame(snuadc_art_cases, columns=['caseid']).to_csv('casos_art.csv', index=False)
data_ids = []
    
# for file_number in snuadc_art_cases[0:11]:
#     print("Executando flush detector da sessão ",file_number)
#     idx,sub,des,asc,desc,conv, resultados = flush_detector(file_number, "resultados_detector")
#     data_ids.append(idx)

file_number = snuadc_art_cases[0]
print("Executando flush detector da sessão ",file_number)
idx,sub,des,asc,desc,conv, resultados = flush_detector(file_number, "resultados_detector")
data_ids.append(idx)

# # Matriz de confusão
# confusion_matrix = pd.DataFrame(dtype='int', columns=["True","False"], index=["True","False"])
# confusion_matrix.iloc[0,0] = resultados.agg(["sum"])["num_true-flush"]
# confusion_matrix.iloc[0,1] = 7
# confusion_matrix.iloc[1,0] = 10
# confusion_matrix.iloc[1,1] = resultados.agg(["sum"])["num_false_flush"]
    
    
    
