#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:25:29 2026

@author: heitor

Modificado para:
  1) Gerar o gráfico da série temporal de cada trecho que atende as
     condições dos if's (1s < deltaT < 19s), centralizado no evento.
  2) Gerar o espectrograma central desse mesmo trecho (mesma janela de
     tempo usada no gráfico da série temporal).
"""

import vitaldb
import matplotlib
#matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np  # Often used with vitaldb
from scipy import signal
import scipy.ndimage as ndimage
import pandas as pd
import os
import csv

SAMPLE_RATE = 100
THRESHOLD_HIGH = 170
THRESHOLD_LOW = 170


def butterworth_filter(t, sig, N, Wn, mode, fs):
    sos = signal.butter(N, Wn, mode, fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    fig, axs = plt.subplots()
    axs.plot(t, filtered)
    axs.set_title('After 10 Hz high-pass filter')
    axs.set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_time_series_and_spectrogram(t_seg, sig_seg, idx, file_number,
                                      det_start, det_end, start_pos, fs):
    """
    Plota, lado a lado (2 subplots), a série temporal do trecho e o
    espectrograma central desse mesmo trecho.
    """
    fig, (ax_ts, ax_spec) = plt.subplots(2, 1, layout='constrained',
                                          figsize=(12, 10))

    # ---- Série temporal ----
    ax_ts.plot(t_seg, sig_seg, label="sinal")
    ax_ts.vlines([t_seg[0] + (det_start - start_pos) / fs,
                  t_seg[0] + (det_end - start_pos) / fs],
                 0, THRESHOLD_HIGH, label='inicio e fim', color='red')
    ax_ts.hlines(THRESHOLD_HIGH,
                 t_seg[0] + (det_start - start_pos) / fs,
                 t_seg[0] + (det_end - start_pos) / fs,
                 label=f'THRESHOLD = {THRESHOLD_HIGH}', color='red')
    ax_ts.set_title(f'Flush {idx} - série temporal (session ID: {file_number})')
    ax_ts.set_xlabel('Time [s]')
    ax_ts.set_ylabel('Amplitude')
    ax_ts.legend(loc='upper right')
    ax_ts.grid()

    # ---- Espectrograma central ----
    # nperseg maior -> estimativa espectral mais estável/menos ruidosa
    # (janela pequena demais faz o centroide "tremer" em toda a série,
    # já que poucas amostras por janela geram muita variância na estimativa
    # de potência espectral). O step (hop) continua pequeno para manter
    # boa resolução temporal e capturar bem as transições abruptas.
    nperseg = min(128, len(sig_seg))
    hop = max(1, nperseg // 32)          # step pequeno -> alta sobreposição
    noverlap = nperseg - hop
    f, t_spec, Sxx = signal.spectrogram(sig_seg, fs=fs,
                                         nperseg=nperseg,
                                         noverlap=noverlap)
    # Espectro em dB para melhor visualização
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    t_spec_abs = t_seg[0] + t_spec

    pcm = ax_spec.pcolormesh(t_spec_abs, f, Sxx_db, shading='gouraud')

    # ---- Centroide espectral (linha) ----
    # Calcular centroide apenas para o range de frequências visualizado
    # (evita distorção por energia em frequências altas fora da visualização)
    fmax_viz = 20.0  # Hz - máximo para visualização
    idx_fmax = np.searchsorted(f, fmax_viz)
    
    # Limita Sxx ao range desejado
    Sxx_limited = Sxx[:idx_fmax, :]
    f_limited = f[:idx_fmax]
    
    # centroid[k] = soma(f * Sxx[:,k]) / soma(Sxx[:,k]), por janela de tempo
    centroid = np.sum(f_limited[:, None] * Sxx_limited, axis=0) / (np.sum(Sxx_limited, axis=0) + 1e-12)
    ax_spec.plot(t_spec_abs, centroid, color='white', linewidth=1.5,
                 label='Centroide espectral')
    ax_spec.legend(loc='upper right')
    
    # Limita o eixo Y da visualização para maior clareza
    ax_spec.set_ylim([0, fmax_viz])

    ax_spec.set_title(f'Flush {idx} - espectrograma central (session ID: {file_number})')
    ax_spec.set_xlabel('Time [s]')
    ax_spec.set_ylabel('Frequência [Hz]')
    fig.colorbar(pcm, ax=ax_spec, label='Potência [dB]')

    plt.show()
    return fig


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

    time = [x * dt for x in range(len(signal_raw_filtered))]
    time = np.array(time)

    detectSimples = []
    subSalva = False

    for idx in range(0, len(signal_raw_filtered)):
        valor_atual = signal_raw_filtered[idx]

        # 1. Detecta a SUBIDA: o sinal cruza o limiar superior
        if (valor_atual >= THRESHOLD_HIGH) and (not subSalva):
            sub = idx
            subSalva = True

        # 2. Detecta a DESCIDA: o sinal cai abaixo do limiar inferior
        elif (valor_atual < THRESHOLD_LOW) and subSalva:
            des = idx
            deltaT = (des - sub) / 100

            if (deltaT > 1) and (deltaT < 19):
                # --- NOVO FILTRO DE RUÍDO ---
                detectSimples.append([sub, des])
            else:
                print(f"-> Descartado: deltaT = {deltaT:.2f}s fora da janela especificada.")

            subSalva = False

    print("-----------------------------------------")
    print(f"Detect Simples: {detectSimples}")
    detectSimples = np.array(detectSimples)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    file_name = f"SNUADC_ART_{file_number}"
    folder_flush_csv = f"{folder_name}/flush_csv"

    os.makedirs(folder_flush_csv, exist_ok=True)

    # FLUSHES
    if (len(detectSimples) == 0):
        print("There was any flush or too many flushes")
        print(f"CaseID {file_number} existe")
        return

    # Visão geral com todos os eventos marcados
    fig, axs = plt.subplots()
    axs.plot(time, signal_raw_filtered, label="filtered")
    axs.vlines(time[detectSimples], 0, 1, linewidth=1, color='r')
    axs.legend()
    axs.grid()
    plt.show()

    # Para cada trecho que atendeu a condição do if (1s < deltaT < 19s),
    # gera o gráfico da série temporal + o espectrograma central do trecho
    for idx in range(len(detectSimples)):
        start_pos = max(0, detectSimples[idx][1] - 60 * SAMPLE_RATE)
        end_pos = min(len(signal_raw_filtered), detectSimples[idx][1] + 60 * SAMPLE_RATE)

        t_seg = time[start_pos:end_pos]
        sig_seg = signal_raw_filtered[start_pos:end_pos]

        plot_time_series_and_spectrogram(
            t_seg, sig_seg, idx, file_number,
            detectSimples[idx][0], detectSimples[idx][1],
            start_pos, SAMPLE_RATE
        )

        #  # Salva o CSV do flush específico (salvando o sinal original não normalizado)
        # save_flush = os.path.join(folder_flush_csv, f'{file_name}_flushes_{idx}_fall_{time[end_pos - 1]}.csv')
        # data_export = np.column_stack((range(start_pos, end_pos), sig_seg))
        # with open(save_flush, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data_export)

    print(f"CaseID {file_number} processado com sucesso. {len(detectSimples)} flushes-like encontrados.")


# IDs que possuem SNUADC/ART
# Baixa a lista completa de faixas (tracks) disponíveis na API
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Filtra apenas as linhas onde o nome da faixa é 'SNUADC/ART'
snuadc_art_cases = df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid'].unique()

# Se quiser salvar em um arquivo:
pd.DataFrame(snuadc_art_cases, columns=['caseid']).to_csv('casos_art.csv', index=True)
data_ids = []

for file_number in snuadc_art_cases[0:1]:
    print("Executando flush detector da sessão ", file_number)
    flush_detector(file_number, "resultados_detector_flushlike")
