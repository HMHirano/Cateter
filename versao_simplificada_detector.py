#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:57:46 2026

@author: heitor
"""

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
from scipy.signal import find_peaks
import os
import json
import csv

### CONSTANTE

FILE_TO_SAVE_CSV = "step_detection_results"
FOLDER_SAVE = "resumo_testes_DB"
SAMPLE_RATE = 100                                   # Taxa de amostragem em Hz
STEP_WIDTH = "definido na função"                   # Largura do degrau em número de amostras
STEP_AMPLITUDE_FACTOR = 0.8                         # Fator para definir a amplitude do degrau em relação ao valor máximo do sinal
STEP_DETECTION_THRESHOLD = "definido na função"     # Limiar para detectar o degrau na convolução
SIGNAL_THRESHOLD = 200
DERIVATIVE_THRESHOLD_ASCENT = 180 / 0.3             # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_DESCENT = - 180 / 0.3                      # Limiar superior para a derivada
dt = 1.0 / SAMPLE_RATE


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
    period_LH = 1
    period_pulse = 2
    t_LH = np.linspace(0.001, period_LH, SAMPLE_RATE * period_LH) # quant_pontos =  100pontos/s * tempo da onda(1)
    t_pulse = np.linspace(-0.01, (period_pulse / 2) + 0.01, SAMPLE_RATE * 1)
    
    # Caso não haja o sinal no banco
    try:
        max(signal_raw_filtered)
    except Exception as e:
        print("The error is: ",e)
        print("This caseID doesn't have SNUADC/ART")
        return
    
    pulse_wave = signal.square(2 * np.pi / period_pulse * t_pulse) * 70 + 70
    square_wave_LH = signal.square( (-1) * (2 * np.pi / period_LH * t_LH)) * 0.8 * max(signal_raw_filtered)
    convolution = signal.fftconvolve(signal_raw_filtered, square_wave_LH, mode="same")
    convolution_pulse = signal.fftconvolve(signal_raw_filtered, pulse_wave, mode="same")
    derivative = np.diff(signal_raw_filtered , prepend=signal_raw_filtered[0])
    mediaConvoluçãoPulso = signal.fftconvolve(convolution_pulse, np.ones(1 * SAMPLE_RATE) / 1 * SAMPLE_RATE, mode="same")
    
    ### NORMALIZATION
    norm_signal = signal_raw_filtered / max(signal_raw_filtered)
    convolution = np.array(convolution / max(convolution))
    norm_convolution_pulse = np.array(convolution_pulse / max(convolution_pulse))
    mediaConvoluçãoPulso = mediaConvoluçãoPulso / max(mediaConvoluçãoPulso)
    norm_der = normalize_robust_and_saturation(derivative)
    
    detectSimples = []
    subSalva = False
    sub = 0  # Inicializa a variável para evitar erros de referência
    
    # Definindo os limiares da histerese
    THRESHOLD_HIGH = 110 * 140 * 100 # 0.75  # Valor para detectar o início do flush (subida)
    THRESHOLD_LOW = 100 * 140 * 100 # 0.65   # Valor para decretar o fim do flush (descida)
    
    for idx in range(0, len(convolution_pulse)):
        valor_atual = convolution_pulse[idx]
        
        # 1. Detecta a SUBIDA: o sinal cruza o limiar superior
        if (valor_atual >= THRESHOLD_HIGH) and (not subSalva):
            sub = idx
            subSalva = True
            
        # 2. Detecta a DESCIDA: o sinal cai abaixo do limiar inferior
        elif (valor_atual < THRESHOLD_LOW) and subSalva:
            des = idx
            deltaT = (des - sub) / 100
            
            # 3. Valida a janela de tempo (entre 1s e 19s)
            if (deltaT > 1) and (deltaT < 19):
                # 1. Calcula a diferença entre as duas curvas no trecho
                diferenca = norm_convolution_pulse[sub:des] - mediaConvoluçãoPulso[sub:des]
                
                # 2. np.sign transforma a diferença em +1 (positivo), -1 (negativo) ou 0
                # np.diff vê quando esses sinais mudam entre pontos consecutivos
                # Sempre que a diferença for diferente de 0, houve um cruzamento!
                interseccao = np.sum(np.diff(np.sign(diferenca)) != 0)
                intersecPorSegundo = interseccao / (deltaT)
                if intersecPorSegundo < 2:
                    print(f"-> Trecho válido adicionado! Instante: {des} Duração: {deltaT:.2f}s Intersecções por segundo: {intersecPorSegundo}")
                    detectSimples.append([sub, des])
                else:
                    print(f"Muitas intersecções por segundo: {intersecPorSegundo} Instante {des}")
            else:
                print(f"-> Descartado: deltaT = {deltaT/SAMPLE_RATE:.2f}s fora da janela especificada.")
                
            # 4. Reseta a flag para procurar o próximo evento
            subSalva = False
            
    # # Platô
    # condition_plato = convolution_pulse > 0.6
    # plato_idx = np.where(condition_plato)[0]
    # # Verifica a continuidade dos idx
    # inicioPlato = plato_idx[0]
    # idxAnterior = plato_idx[0]
    # plato = []
    # for idx in range(0,len(plato_idx)):
    #     if idx - pla_ant > 1:
    #         plato.append(idx)
    #     pla_ant = idx
    
    # if len(plato) == 0:
    #     return
    
    # Pareia subida com descida, demarcando início e fim do flush
    step =[]
    # for pla in plato:
    #     for desc in abs_descent_idx:
    #         if pla < desc:
    #             print(f"Comparação entre asc:{pla} e desc:{desc} e dif:{desc-pla}")
    #             # CONDIÇÃO DA DURAÇÃO DO FLUSH
    #             if (desc-pla) > 1*SAMPLE_RATE and (desc-pla) < 15*SAMPLE_RATE: # flush deve durar mais que 3s e menos que 12s
    #                 #if len(step) == 0:
    #                 step = step + [[pla,desc]]  
    #                 # Verifica se uma mesma descida é utilizada por duas subidas diferentes
    #                 # elif asc > step[-1][1]: 
    #                 #     step = step + [[asc,desc]]
    #                 #     der_values.append(derivative[asc])
    #                 break
    #             else:
    #                 if (desc-pla) < 1*SAMPLE_RATE:
    #                     print(f"desc-asc não é maior que 1s: {desc-pla}")
    #                 elif (desc-pla) > 15*SAMPLE_RATE:
    #                     print(f"desc-asc maior que 15s: {desc-pla}")
    #                 break
        
    ### GRÁFICOS
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        
    # OVERVIEW
    fig, sig = plt.subplots(figsize=(20, 15))
    sig.plot(time, norm_signal, label="SNUADC/ART", linewidth=1)
    sig.plot(time, convolution, label="convolution", linewidth=1)
    sig.plot(time, norm_convolution_pulse, label="pulse convolution", linewidth=1)
    sig.plot(time, norm_der, label="norm der", linewidth=1, alpha=0.2)
    sig.plot(time, mediaConvoluçãoPulso, label="media Conv Pulso", linewidth=1)
    # sig.vlines(time[plato], 0, 1, label="plato", color='greenyellow', alpha=0.2)
    sig.hlines(180 / max(signal_raw_filtered), time[0], time[-1])
    #sig.vlines(time[abs_descent_idx], 0, 1, label="descida", color='red', alpha=1)
    sig.set_title(f"Raw signal and Convolution from caseID_{file_number}")
    sig.legend(loc='upper right')
    sig.grid()
    fig.show()
    
    file_name = f"SNUADC_ART_{file_number}"
    # save_path = folder_name + '/' + f'{file_name}_overview.png'
    # fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    
    folder_flush_csv = f"{folder_name}/flush_csv"
    
    try:
        os.mkdir(folder_flush_csv)
        print(f"Folder '{folder_name}' created.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")
    except FileNotFoundError:
        print("Parent directory does not exist.")
    
    # #FLUSHES
    # if (len(step) == 0) or (len(step) > 5):
    #     print("There was any flush")
    #     print(f"CaseID {file_number} existe")
    #     return 
    
    # n_per_fig = 3   
    # for i in range(0, len(step), n_per_fig):
    #     if (len(step) - i) < n_per_fig:
    #         fig, axs = plt.subplots((len(step) - i), 1, layout='constrained', figsize=(20, 15), squeeze=False)
    #     else:
    #         fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(20, 15), squeeze=False)
    #     for j, ax in enumerate(axs.flat):
    #         idx = i + j
    #         if idx < len(step):
    #             start_pos = max(0, step[idx][1] - 20*SAMPLE_RATE)
    #             end_pos = min(len(norm_signal), step[idx][1] + 20*SAMPLE_RATE)
                
    #             t_seg = time[start_pos:end_pos]
                
    #             ax.plot(t_seg, norm_signal[start_pos:end_pos], label=f'Instant of flush {time[step[idx][0]]:.2f} s', linewidth=1)
    #             ax.plot(t_seg, convolution[start_pos:end_pos], linewidth=1)
    #             ax.hlines(2 * np.mean(convolution[start_pos:end_pos]), time[start_pos], time[end_pos], label="2 médias")
    #             ax.set_title(f'Flush {i + j} at time {time[step[idx][0]]:.2f} s')
    #             ax.set_ylabel('Amplitude')
    #             ax.legend(loc='upper right')
    #             ax.grid()
    #     # figure_number = i // 3
    #     fig.show()
    #     # save_figures = folder_name + '/' + f'{file_name}_flushes_{figure_number}.png'
    #     # fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)
        
    #     # Salva o CSV do flush específico (salvando o sinal original não normalizado)
    #     folder_flush_csv = "flush_csv"
    #     save_flush = os.path.join(folder_name, folder_flush_csv, f'{file_name}_flushtest_{idx}.csv')
    #     # np.column_stack é mais seguro para juntar arrays 1D como colunas
    #     data_export = np.column_stack((t_seg, norm_signal[start_pos:end_pos])) 
    #     with open(save_flush, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerows(data_export)
            
    #FLUSHES
    if (len(detectSimples) == 0) and (len(detectSimples < 18)):
        print("There was any flush or too many flushes")
        print(f"CaseID {file_number} existe")
        return 
    
    n_per_fig = 3   
    for i in range(0, len(detectSimples), n_per_fig):
        if (len(detectSimples) - i) < n_per_fig:
            fig, axs = plt.subplots((len(detectSimples) - i), 1, layout='constrained', figsize=(20, 15), squeeze=False)
        else:
            fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(20, 15), squeeze=False)
        for j, ax in enumerate(axs.flat):
            idx = i + j
            if idx < len(detectSimples):
                start_pos = max(0, detectSimples[idx][1] - 20*SAMPLE_RATE)
                end_pos = min(len(norm_signal), detectSimples[idx][1] + 20*SAMPLE_RATE)
                
                t_seg = time[start_pos:end_pos]
                
                ax.plot(t_seg, norm_signal[start_pos:end_pos], label='normalizeed signal', linewidth=1)
                ax.plot(t_seg, norm_convolution_pulse[start_pos:end_pos], label='convolution pulse', linewidth=1)
                ax.plot(t_seg, mediaConvoluçãoPulso[start_pos:end_pos],label='media convolução', linewidth=1)
                ax.vlines(time[detectSimples[idx][0]], 0, 1, label='inicio')
                ax.vlines(time[detectSimples[idx][1]], 0, 1, label='fim')                
                ax.hlines(THRESHOLD_HIGH / max(convolution_pulse), t_seg[0], t_seg[-1])
                ax.set_title(f'Flush {i + j} at time {time[detectSimples[idx][0]]:.2f} s')
                ax.set_ylabel('Amplitude')
                ax.legend(loc='upper right')
                ax.grid()
        # figure_number = i // 3
        fig.show()
        # save_figures = folder_name + '/' + f'{file_name}_flushes_{figure_number}.png'
        # fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)
        
        # # Salva o CSV do flush específico (salvando o sinal original não normalizado)
        # folder_flush_csv = "flush_csv"
        # save_flush = os.path.join(folder_name, folder_flush_csv, f'{file_name}_flushtest_{idx}.csv')
        # # np.column_stack é mais seguro para juntar arrays 1D como colunas
        # data_export = np.column_stack((t_seg, norm_signal[start_pos:end_pos])) 
        # with open(save_flush, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data_export) 
    
    
    print(f"CaseID {file_number} processado com sucesso. {len(detectSimples)} flushes-like encontrados.")
    
    return
    
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
#     idx,sub,des,asc,desc,conv = flush_detector(file_number, "resultados_detector")
#     data_ids.append(idx)

file_number = snuadc_art_cases[2]
print("Executando flush detector da sessão ",file_number)
flush_detector(file_number, "resultados_detector")
    
    
    
