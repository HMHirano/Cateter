#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:07:17 2026

@author: heitor
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vitaldb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import os
import json
import csv

### CONSTANTES
FILE_TO_SAVE_CSV = "step_detection_results"
FOLDER_SAVE = "resumo_testes_DB"
SAMPLE_RATE = 100
MIN_FLUSH_DURATION = 3 * SAMPLE_RATE
MAX_FLUSH_DURATION = 45 * SAMPLE_RATE
DERIVATIVE_THRESHOLD_LOW = 15
dt = 1.0 / SAMPLE_RATE

# ==========================================
# FUNÇÕES DE PROCESSAMENTO DE SINAL
# ==========================================

def normalize_robust(sig):
    """Normaliza o sinal utilizando o percentil 99.5 para evitar que outliers anulem os dados reais."""
    max_val = np.percentile(sig, 99.5)
    return sig / max_val if max_val != 0 else sig

def match_ascent_descent(abs_ascent_idx, abs_descent_idx, derivative, sig_norm):
    """Pareia subida com descida usando busca vetorizada."""
    step = []
    der_values = []
    
    asc_arr = np.array(abs_ascent_idx)
    desc_arr = np.array(abs_descent_idx)
    
    for asc in asc_arr:
        # Encontra a primeira descida após a subida
        idx_desc = np.searchsorted(desc_arr, asc)
        
        if idx_desc < len(desc_arr):
            desc = desc_arr[idx_desc]
            duration = desc - asc
            
            if MIN_FLUSH_DURATION <= duration <= MAX_FLUSH_DURATION:
                segment = sig_norm[asc:desc]
                # Verifica se há vales indesejados no meio do flush
                if not np.any(segment < 0.3):
                    step.append([int(asc), int(desc)])
                    der_values.append(derivative[asc])
                    
    return step, der_values

def normalize_robust_abs(sig):
    """
    Normaliza o sinal utilizando o percentil 99.5 dos valores absolutos.
    Ideal para sinais que oscilam entre valores positivos e negativos, como derivadas.
    Melhor para os cálculos ao desconsiderar outliers e parte dos ruídos.
    """
    max_val = np.percentile(np.abs(sig), 95)
    return sig / max_val if max_val != 0 else sig

def process_flush_signal(signal_raw):
    """Processa o sinal bruto: filtra, normaliza, faz a convolução e encontra os índices dos flushes."""
    # 1. Remove NaNs
    sig_no_nan = signal_raw[~np.isnan(signal_raw)]
    time = np.arange(len(sig_no_nan)) * dt
    
    # 2. APLICAÇÃO DO FILTRO PASSA-BAIXAS (Butterworth)
    # Frequência de corte (ex: 5 Hz) - ajustável conforme a necessidade do sinal arterial
    cutoff_hz = 20 
    nyq = 0.5 * SAMPLE_RATE
    normal_cutoff = cutoff_hz / nyq
    
    # Cria um filtro Butterworth de ordem 4
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # Aplica o filtro usando filtfilt (para não dar atraso/deslocamento de fase no sinal)
    sig_filtered = signal.filtfilt(b, a, sig_no_nan)
    
    # 3. Normalização e Derivada (agora usando o sinal filtrado!)
    sig_norm = normalize_robust(sig_filtered)
    
    # Derivada sofre muito menos com ruídos agora
    derivative = np.diff(sig_filtered, prepend=sig_filtered[0]) 
    norm_der = normalize_robust_abs(derivative)
    
    # Convolução (Onda Quadrada)
    period = 1
    t = np.linspace(0.001, period, SAMPLE_RATE * period)
    template_amp = 0.8 * np.percentile(sig_filtered, 95)
    square_wave_LH = signal.square(-2 * np.pi / period * t) * template_amp
    
    convolution = signal.fftconvolve(sig_filtered, square_wave_LH, mode="same")
    conv_norm = convolution / max(convolution)
    
    # Condições de Subida
    condition_ascent = (conv_norm < -0.35) & (derivative > DERIVATIVE_THRESHOLD_LOW) & (sig_norm > 0.4)
    ascent_indices = np.where(condition_ascent)[0]
    
    abs_ascent_idx = []
    if len(ascent_indices) > 0:
        idx_ant = ascent_indices[0]
        for idx in ascent_indices:
            if idx - idx_ant > 100:
                abs_ascent_idx.append(idx_ant)
            idx_ant = idx
        abs_ascent_idx.append(idx_ant)
        
    # Condições de Descida
    condition_descent = conv_norm > 0.35
    descent_indices = np.where(condition_descent)[0]
    
    # Usando find_peaks para as descidas (mais estável que argrelextrema)
    abs_descent_idx, _ = signal.find_peaks(conv_norm, height=0.35)
    
    # Pareamento
    step, der_values = match_ascent_descent(abs_ascent_idx, abs_descent_idx, derivative, sig_norm)
    
    return time, sig_filtered, sig_norm, derivative, norm_der, conv_norm, ascent_indices, descent_indices, abs_ascent_idx, abs_descent_idx, step, der_values

# ==========================================
# FUNÇÃO PRINCIPAL (ORQUESTRADOR E GRÁFICOS)
# ==========================================

def flush_detector_pipeline(file_number, folder_name):
    ### EXTRAÇÃO DOS SINAIS VITALDB
    track_names = ['SNUADC/ART']
    try:
        vf = vitaldb.VitalFile(file_number, track_names)
        samples = vf.to_numpy(track_names, dt)
        signal_raw = samples[:, 0]
    except Exception as e:
        print(f"Erro ao carregar o caso {file_number}: {e}")
        return
    
    if len(signal_raw[~np.isnan(signal_raw)]) == 0:
        print(f"Sinal vazio para o caseID {file_number}")
        return
        
    ### PROCESSAMENTO MATEMÁTICO
    (time, sig_filtered, sig_norm, derivative, norm_der, conv_norm, 
     ascent_indices, descent_indices, abs_ascent_idx, abs_descent_idx, step, der_values) = process_flush_signal(signal_raw)

    ### SALVAMENTO DE DADOS (JSON e CSV)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    # JSON das derivadas
    try:
        with open("derivative.json", "r") as f:
            deri = json.load(f)
    except Exception:
        deri = {}
        
    deri[f"CASEID_{file_number}"] = [float(val) for val in der_values]
    with open("derivative.json", "w") as f:
        json.dump(deri, f)
        
    # CSV de Resultados Resumidos
    csv_path = f"{folder_name}/resultados.csv"
    try:
        resultados = pd.read_csv(csv_path, index_col=0)
    except Exception:
        resultados = pd.DataFrame(columns=["num_flush-like", "num_true-flush", "num_false_flush", "possiveis_flush"])
        
    num_flush_like = len(abs_descent_idx)
    num_true_flush = len(step)
    num_false_flush = num_flush_like - len(abs_ascent_idx)
    poss_flush = len(abs_ascent_idx) - num_true_flush
    
    data = {"num_flush-like": num_flush_like, "num_true-flush": num_true_flush, 
            "num_false_flush": num_false_flush, "possiveis_flush": poss_flush} 
    
    resultados.loc[f"caseID_{file_number}"] = data
    resultados.to_csv(csv_path)

    ### GERAÇÃO DE GRÁFICOS
    
    # Feitos os cálculos, a derivada é normalizada para ficar entre -1 e 1 para ser mais fácil a visualização no gráfico
    renormalized_der = norm_der / max(norm_der)
    
    # 1. Gráfico de Visão Geral (Overview)
    fig, sig_ax = plt.subplots(figsize=(20, 15))
    sig_ax.plot(time, sig_norm, label="SNUADC/ART (Normalized)", linewidth=1)
    sig_ax.plot(time, conv_norm, label="Convolution", linewidth=1)
    sig_ax.plot(time, renormalized_der, label="Derivative", color='r', linewidth=1)
    sig_ax.set_title(f"Raw signal and Convolution from caseID_{file_number}")
    sig_ax.vlines(time[descent_indices], 0, 1, label="Descent Region", color='tab:green', alpha=0.1)
    sig_ax.vlines(time[ascent_indices], 0, 1, label="Ascent Region", color='tab:olive', alpha=0.5)
    
    # Proteção caso abs_descent_idx esteja vazio
    if len(abs_descent_idx) > 0:
        sig_ax.vlines(time[abs_descent_idx], 0, 1, color='tab:cyan', alpha=1, label="Descent Peaks")
        
    sig_ax.legend(loc='upper right')
    sig_ax.grid()
    # Opcional: Descomente para salvar o overview
    # overview_path = os.path.join(folder_name, f'SNUADC_ART_{file_number}_overview.png')
    # fig.savefig(overview_path, bbox_inches='tight', pad_inches=0.1)
    # fig.show()
    #plt.close(fig) # Fecha a figura para não acumular na memória

    # 2. Gráficos Individuais dos Flushes Confirmados
    if len(step) == 0:
        print(f"Nenhum flush detectado para o CaseID {file_number}")
        return file_number, abs_ascent_idx, abs_descent_idx, ascent_indices, descent_indices, conv_norm, resultados
    
    file_name = f"SNUADC_ART_{file_number}"
    n_per_fig = 3   
    
    for i in range(0, len(step), n_per_fig):
        n_subplots = min(n_per_fig, len(step) - i)
        fig, axs = plt.subplots(n_subplots, 1, layout='constrained', figsize=(20, 15), squeeze=False)
        
        for j, ax in enumerate(axs.flat):
            idx = i + j
            if idx < len(step):
                start_pos = max(0, step[idx][0] - 20 * SAMPLE_RATE)
                end_pos = min(len(sig_filtered), step[idx][1] + 20 * SAMPLE_RATE)
                
                t_seg = time[start_pos:end_pos]
                
                ax.plot(t_seg, sig_norm[start_pos:end_pos], label=f'Instant {time[step[idx][0]]:.2f} s', linewidth=1)
                ax.plot(t_seg, conv_norm[start_pos:end_pos], linewidth=1, label="Convolution")
                ax.plot(t_seg, renormalized_der[start_pos:end_pos], linewidth=1, label="Derivative")
                
                mean_conv = np.mean(conv_norm[start_pos:end_pos])
                ax.hlines(2 * mean_conv, t_seg[0], t_seg[-1], label="2x Conv Mean", colors='purple', linestyles='dashed')
                
                ax.set_title(f'Flush {idx} at time {time[step[idx][0]]:.2f} s | Max derivative: {np.max(derivative[start_pos:end_pos]):.2f}')
                ax.set_ylabel('Amplitude Normalized')
                ax.legend(loc='upper right')
                ax.grid()
                
                # Salva o CSV do flush específico (salvando o sinal original não normalizado)
                folder_flush_csv = "flush_csv"
                save_flush = os.path.join(folder_name, folder_flush_csv, f'{file_name}_flushtest_{idx}.csv')
                # np.column_stack é mais seguro para juntar arrays 1D como colunas
                data_export = np.column_stack((t_seg, sig_filtered[start_pos:end_pos])) 
                with open(save_flush, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data_export)                  
                    
        figure_number = i // n_per_fig
        save_figures = os.path.join(folder_name, f'{file_name}_flushes_{figure_number}.png')
        fig.savefig(save_figures, bbox_inches='tight', pad_inches=0.1)
        #plt.close(fig)

    print(f"CaseID {file_number} processado com sucesso. {len(step)} flushes encontrados.")
    return file_number, abs_ascent_idx, abs_descent_idx, ascent_indices, descent_indices, conv_norm, resultados

# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    snuadc_art_cases = df_trks[df_trks['tname'] == 'SNUADC/ART']['caseid'].unique()
    
    data_ids = []
    
    # Exemplo: Testando no 6º caso
    file_number = snuadc_art_cases[5]
    print("Executando flush detector da sessão", file_number)
    
    result = flush_detector_pipeline(file_number, "resultados_detector")
    if result:
        idx, sub, des, asc, desc, conv, resultados = result
        data_ids.append(idx)