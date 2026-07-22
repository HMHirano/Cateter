#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:25:29 2026

@author: heitor
"""

import vitaldb
import matplotlib 
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np # Often used with vitaldb
from scipy import signal
import scipy.ndimage as ndimage
import pandas as pd
import os
import csv
import librosa

SAMPLE_RATE = 100
THRESHOLD_HIGH = 1.0
THRESHOLD_LOW = 0.9

def butterworth_filter(t, sig, N, Wn, mode, fs):
    sos = signal.butter(N, Wn, mode, fs, output='sos')
    filtered = signal.sosfilt(sos,sig)
    fig, axs = plt.subplots()
    axs.plot(t, filtered)
    axs.set_title('After 10 Hz high-pass filter')
    axs.set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()

def flush_detector(file_number, folder_name):
    # SINAIS
    track_names = ['SNUADC/ART']
    try:
        vf = vitaldb.VitalFile(file_number, track_names)
    except Exception as e:
        print("The error is: ", e)
        print("This caseID doesn't have SNUADC/ART")
        return
    
    dt = 1.0 / SAMPLE_RATE
    samples = vf.to_numpy(track_names, dt)
    signal_raw = samples[:, 0]
    signal_raw_filtered = signal_raw[~np.isnan(signal_raw)]
    
    # CONVOLUÇÃO
    
    time = [x*dt for x in range(len(signal_raw_filtered))]
    time = np.array(time)

    # Janela deve ser um pouco maior que um ciclo cardíaco (ex: 1.5 segundos)
    window_size = int(1 * SAMPLE_RATE)
    
    # Aplica o filtro de mínimo
    signal_eroded = ndimage.minimum_filter1d(signal_raw_filtered, size=window_size)
    
    norm_signal = signal_raw_filtered / max(signal_raw_filtered)
    
    #%%
    ### ESPECTROGRAMA DO SINAL
    # From time-series input:
    y = signal_raw_filtered
    #y, sr = librosa.load(librosa.ex('trumpet'))

    # Set your window size
    n_fft = 1 * SAMPLE_RATE  # 200 samples for a 2-second window

    # Optionally, you can set hop_length to control the overlap.
    # Default is n_fft // 4 (which would be 50 samples, or 0.5 seconds).
    hop_length = n_fft // 8 

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

    # Figura com o espectrograma (em cima) acompanhado por um gráfico no
    # domínio do tempo (embaixo), compartilhando o mesmo eixo x (tempo).
    fig, (ax, ax_time) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # 1. Change y_axis='log' to y_axis='linear'
    # Physiological signals are usually much easier to interpret on a linear frequency scale.
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='linear', x_axis='time', 
                             sr=SAMPLE_RATE, hop_length=hop_length, ax=ax)

    # Gráfico acompanhando o espectrograma: sinal ABP no domínio do tempo
    ax_time.plot(time, norm_signal, color='k', linewidth=0.8, label='ABP signal (normalized)')
    ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Amplitude (norm.)')
    ax_time.legend(loc='upper right')
    ax_time.grid()

    # ax.plot(times_plot, cent.T, label='Spectral centroid', color='w')

    # # 2. Set the y-axis limits to focus on the 0 to 20 Hz range
    # ax.set_ylim([0, 20]) 

    # ax.legend(loc='upper right')
    # ax.set(title='Linear Power Spectrogram of ABP Signal (0 - 20 Hz)')
    # plt.show()
    
    #%%
    
    detectSimples = []
    detect_timecentroid = []
    subSalva = False
    
    for idx in range(0,len(cent.T)):
        valor_atual = cent.T[idx]

        # 1. Detecta a SUBIDA: o sinal cruza o limiar superior
        if (valor_atual <= THRESHOLD_LOW) and (not subSalva):
            sub = idx
            subSalva = True

        # 2. Detecta a DESCIDA: o sinal cai abaixo do limiar inferior
        elif (valor_atual >= THRESHOLD_HIGH) and subSalva:
            des = idx
            sub_sinal = librosa.frames_to_samples(sub, hop_length=hop_length)
            des_sinal = librosa.frames_to_samples(des, hop_length=hop_length)
            deltaT = (des_sinal - sub_sinal) / SAMPLE_RATE
            
            if (deltaT > 1) and (deltaT < 19):
                
                condition_sub = np.mean(cent.T[sub-8:sub] > 1.0) and np.mean(cent.T[sub-8:sub] < 2.0)
                condition_des = np.mean(cent.T[des:des + 8] > 1.0) and np.mean(cent.T[des:des + 8] < 2.0)
                
                if condition_sub and condition_des:
                    # --- NOVO FILTRO DE RUÍDO ---
                    detectSimples.append([sub_sinal, des_sinal])
                    detect_timecentroid.append([sub,des])
                    print(f"Trecho adicionado: deltaT {deltaT}")
                
            else:
                print(f"-> Descartado: deltaT = {deltaT:.2f}s fora da janela especificada.")

            subSalva = False
    
    print("-----------------------------------------")
    print(f"Detect Simples: {detectSimples}")
    detectSimples = np.array(detectSimples)
    detect_timecentroid = np.array(detect_timecentroid)
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        
    file_name = f"SNUADC_ART_{file_number}"
    folder_flush_csv = f"{folder_name}/flush_csv"
    
    try:
        os.mkdir(folder_flush_csv)
        print(f"Folder '{folder_name}' created.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")
    except FileNotFoundError:
        print("Parent directory does not exist.")
    
    #FLUSHES
    if (len(detectSimples) == 0) and (len(detectSimples) < 18):
        print("There was any flush or too many flushes")
        print(f"CaseID {file_number} existe")
        return
    
    ax.plot(times_plot, cent.T, label='Spectral centroid', color='w')
    ax.vlines(times_plot[detect_timecentroid], 0, 3, label='start and finish', color='green')
    ax.hlines(cent.T[detect_timecentroid[:, 0]].flatten(), times_plot[detect_timecentroid[:, 0]], times_plot[detect_timecentroid[:, 1]], label='threshold', color='green')

    # 2. Set the y-axis limits to focus on the 0 to 20 Hz range
    ax.set_ylim([0, 20]) 

    ax.legend(loc='upper left')
    ax.set_title('Linear Power Spectrogram of ABP Signal (0 - 20 Hz)')

    # Marca os mesmos inícios/fins de flush no gráfico do domínio do tempo
    ax_time.vlines(time[detectSimples], 0, 1, color='green', label='start and finish')
    ax_time.legend(loc='upper left')
    fig.show()
    
    fig,axs = plt.subplots()
    axs.plot(time, norm_signal, label="filtered")
    # 'cent.T' está no domínio de frames (um valor a cada hop_length amostras),
    # então precisa ser plotado contra 'times_plot' (e não 'time', que está no
    # domínio de amostras) — essa era a causa do ValueError.
    axs.plot(times_plot, cent.T, label="centroide", color="red")
    axs.vlines(time[detectSimples], 0, 1, linewidth=1, color='green')
    axs.vlines(times_plot[detect_timecentroid], 0, 3, label='start and end', color='green')
    # 'detect_timecentroid' guarda índices de frame, então é o índice correto
    # para 'cent.T'; 'detectSimples' guarda índices de amostra e deve ser
    # usado apenas com 'time'/'signal_raw_filtered'.
    axs.hlines(cent.T[detect_timecentroid[:, 0]].flatten(), times_plot[detect_timecentroid[:, 0]], times_plot[detect_timecentroid[:, 1]], label='limiar', color='green')
    axs.hlines(180/max(signal_raw_filtered), 0, time[-1], label="180 mmHg")
    axs.legend(loc='upper left')
    axs.grid()
    fig.show()

    n_per_fig = 1
    for i in range(0, len(detectSimples), n_per_fig):
        if (len(detectSimples) - i) < n_per_fig:
            fig, axs = plt.subplots((len(detectSimples) - i), 1, layout='constrained', figsize=(20, 15), squeeze=False)
        else:
            fig, axs = plt.subplots(n_per_fig, 1, layout='constrained', figsize=(20, 15), squeeze=False)
        for j, ax in enumerate(axs.flat):
            idx = i + j
            if idx < len(detectSimples):
                start_pos = max(0, detectSimples[idx][1] - 20*SAMPLE_RATE)
                end_pos = min(len(signal_raw_filtered), detectSimples[idx][1] + 20*SAMPLE_RATE)

                t_seg = time[start_pos:end_pos]
                t_cent_start = detectSimples[idx][1] - start_pos
                t_cent_end = end_pos - detectSimples[idx][1]
                t_centralizado = range(-t_cent_start, t_cent_end)
        
                ax.plot(t_seg, norm_signal[start_pos:end_pos], label="sinal")
                ax.vlines(time[detectSimples[idx]], 0, 1, label='start and finish', color='red')
                ax.hlines(norm_signal[detectSimples[idx][0]], time[detectSimples[idx][0]], time[detectSimples[idx][1]], label='start and finish', color='red')
                ax.set_title(f'Flush {idx} at time {time[detectSimples[idx][0]]:.2f} s')
                ax.set_ylabel('Amplitude')
                ax.legend(loc='upper left')
                ax.grid()

                 # Salva o CSV do flush específico (salvando o sinal original não normalizado)
                folder_flush_csv = "flush_csv"
                save_flush = os.path.join(folder_name, folder_flush_csv, f'{file_name}_waveform_{idx}_fall_{time[end_pos - 1]}.csv')                # np.column_stack é mais seguro para juntar arrays 1D como colunas
                data_export = np.column_stack((t_centralizado, signal_raw_filtered[start_pos:end_pos]))
                with open(save_flush, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_export)

        figure_number = i // 3
        plt.show()
        save_figures = folder_name + '/' + f'{file_name}_flushes_{figure_number}.png'
        fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)
        # plt.close(fig)
        

    print(f"CaseID {file_number} processado com sucesso. {len(detectSimples)} flushes-like encontrados.")
    
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

for file_number in snuadc_art_cases[0:1]:
    print("Executando flush detector da sessão ", file_number)
    flush_detector(file_number, "resultados_detector_flushlike")

# file_number = snuadc_art_cases[0]
# print("Executando flush detector da sessão ",file_number)
# flush_detector(file_number, "resultados_detector")
    
    