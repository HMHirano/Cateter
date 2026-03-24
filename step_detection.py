import vitaldb
import matplotlib.pyplot as plt
import numpy as np # Often used with vitaldb
import pandas as pd # Also commonly used
import os
import csv
import scipy.signal as signal


FOLDER_NAME = "resumo_testes_Alicia"
SAMPLE_RATE = 250                                   # Taxa de amostragem em Hz
STEP_WIDTH = "definido na função"                   # Largura do degrau em número de amostras
STEP_AMPLITUDE_FACTOR = 0.8                         # Fator para definir a amplitude do degrau em relação ao valor máximo do sinal
STEP_DETECTION_THRESHOLD = "definido na função"     # Limiar para detectar o degrau na convolução
SIGNAL_THRESHOLD = 200                              # Limiar para o sinal suavizado

'''
# Subida
DERIVATIVE_THRESHOLD_LOW = 5        # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_HIGH = 60      # Limiar superior para a derivada
'''

# Descida
DERIVATIVE_THRESHOLD_LOW = -15                      # Limiar inferior para a derivada   
DERIVATIVE_THRESHOLD_HIGH = 0                       # Limiar superior para a derivada

track_names = ['SNUADC/ART']
#vf = vitaldb.VitalFile(3, track_names)
#samples = vf.to_numpy(track_names, dt)


def convolution(csv_file, sample_rate, step_amplitude_factor, derivative_threshold_low, derivative_threshold_high, signal_threshold):
    dt = 1.0 / sample_rate
    samples = pd.read_csv(csv_file).to_numpy()
    print(f'Samples shape: {samples.shape}')

    time = samples[:, 0]
    signal_raw = samples[:, 1]

    # --- 1. Smooth Signal and Convolve with Step Function ---
    # Vou definir a largura do degrau de modo a ter aproximadamente 3 segundos de duração.
    step_amplitude = np.nanmax(signal_raw) * step_amplitude_factor # Arbitrário
    print(f"Max signal value set to: {step_amplitude}")
    # Utilizando 3 segundos para a largura do degrau, em flushes curtos como os da Alicia,
    # o valor da convolução não era igual a (Ampl**2)*width, mas sim  um pouco menor, pois
    # parte do degrau ficava fora do flush. Portanto, para capturar o máximo possível do flush,
    # e utilizar isso como critério, se
    step_width = int(3 * sample_rate) # 3 segundos
    step_function = np.ones(step_width) * step_amplitude

    signal_smooth = np.convolve(signal_raw, [1/3, 1/3, 1/3], mode='same')
    step_detection = np.convolve(signal_smooth, step_function, mode='same')
    print("Signal smoothed using np.convolve.")
    print("Smoothed signal convolved with step function.")

    triangle_window = np.bartlett(100)*100
    triangle_detection = np.convolve(signal_smooth, triangle_window/np.sum(triangle_window), mode='same') / 100
    print("Applied triangular window smoothing.")

    # --- 2. Calculate Derivative and Find Peaks ---
    derivative_raw = np.diff(signal_raw, prepend=signal_raw[0])
    derivative_smooth = np.diff(signal_smooth, prepend=signal_smooth[0])

    ## Through derivative
    # Condition 1: Derivative is high
    condition1 = (derivative_smooth > derivative_threshold_low) & (derivative_raw < derivative_threshold_high) # > 5 and < 60
    # Condition 2: The derivative is high in points where the signal is also high
    condition2 = (signal_smooth > signal_threshold) # > 200

    # Condition 3: The threshold of the convolution value must be high to stand out from other waveforms
    # but it also needs to capture the descending slope of the step, so we use as criteria a value slightly lower than
    # the complete overlap of a step function with amplitude 200, so that it will capture the slope used in the derivative
    # criteria. If the signal threshold changes, the step detection threshold should also be adjusted accordingly.
    step_detection_threshold = 0.9 * (signal_threshold**2) * step_width
    condition3 = (step_detection > step_detection_threshold) # Valor encontrado pelos gráficos > 30000

    # Find flush_idx where BOTH conditions are true
    peak_indices = np.where(condition1 & condition2)[0]

    step_indices = np.where(condition3)[0]

    square_waves = np.array([]) # quantidade de ondas quadradas detectadas
    flush_idx = np.array([])

    ''' Combine both criteria to find relevant points '''
    '''
    for i in range(len(peak_indices)-1):
        for j in range(len(step_indices)-1):
            difference_indices = step_indices[j] - peak_indices[i]
            if (difference_indices < 1000) and (difference_indices > 0): # Pega flush_idx de pico e degrau próximos e que não estejam na lista
                print(f"Comparing peak at {peak_indices[i]} with step at {step_indices[j]}")
                if flush_idx.size > 0:
                    print(f"Last added index: {flush_idx[-1]}")
                if (flush_idx.size == 0) or (peak_indices[i] - flush_idx[-1]) > 100: # Descarta flush_idx aproximadamente repetidos e verifica se a lista está vazia
                    flush_idx = np.append(flush_idx, peak_indices[i])
                break
    '''
    for step_idx in step_indices:
        # Salva todas as ondas quadradas detectadas para depois calcular flushes falsos. Supondo uma taxa de amostragem de 100 Hz,
            # 300 amostras equivalem a 3 segundos.
        if square_waves.size == 0:
            square_waves = np.append(square_waves, step_idx)
            print(f"Square wave detected at index: {step_idx}")
        elif step_idx - square_waves[-1] > 1000:
            square_waves = np.append(square_waves, step_idx)
            print(f"Square wave detected at index: {step_idx}")
        if flush_idx.size > 0:
            # Verifica se o pico não está muito próximo do último pico adicionado.
            if step_idx - flush_idx[-1] > 300:
                flush_idx = np.append(flush_idx, peak_indices[peak_indices == step_idx])
        else:    
            flush_idx = peak_indices[peak_indices == step_idx]


    print(f"Step indices:")
    print(step_indices)

    print(f"Final detected step indices:")
    print(flush_idx)

    print(f"Found {len(flush_idx)} points meeting both criteria.")

    return time, step_detection, signal_smooth, triangle_detection, derivative_smooth, step_indices, peak_indices, square_waves, flush_idx

# --- 3. Plotting ---
def plot_results(file_name, time, step_detection, signal_smooth, triangle_detection, derivative_smooth, step_indices, peak_indices, flush_idx):
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, step_detection)
    plt.title('Convolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(time, signal_smooth)
    plt.title('Smoothed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.vlines(x=time[step_indices], ymin=0, ymax=350, color='r', alpha=0.05, linestyle='-', label='Detected Steps')
    plt.vlines(x=time[peak_indices], ymin=0, ymax=350, color='g', alpha=0.1, linestyle='-', label='Detected Peaks')

    plt.subplot(2,2,3)
    plt.plot(time, triangle_detection)
    plt.title('Triangle Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(time, derivative_smooth)
    plt.title('Derivative Signal Smooth')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME, exist_ok=True)

    save_path = FOLDER_NAME + '/' + f'{file_name}_overview.png'

    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1) 

    if len(flush_idx) == 0:
        print("No flush_idx detected for detailed plotting.")
        plt.show(block= False)
        return
    
    fig, axes = plt.subplots(len(flush_idx), 1, figsize=(10, 3*len(flush_idx)), layout='constrained')

    if len(flush_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, flush_idx):
        idx = int(idx)
        start_pos = max(0, idx - 1000)
        end_pos = min(len(signal_smooth), idx + 4000)
        ax.plot(time[start_pos:end_pos], signal_smooth[start_pos:end_pos], label=f'Instant of flush {time[idx]:.2f} s')
        ax.set_title(f'Segment around detected step at time {time[idx]:.2f} s')
        ax.set_ylabel('Amplitude')
        ax.grid()

    save_path = FOLDER_NAME + '/' f'{file_name}_detailed.png'

    plt.savefig(save_path)

    plt.show(block=False)
    plt.pause(0.001)
    plt.close('all')

def csv_convolution_tests(file_name, fileToSaveCsv, time, square_waves, flush_idx):
    with open(f'{fileToSaveCsv}.csv', 'a', newline='') as csvfile: #abre o mesmo arquivo
        writer = csv.writer(csvfile)
        time_txt = ''
        for idx in flush_idx:
            time_round = round(time[int(idx)], 1)
            if time_txt == '':
                time_txt = f'{time_round}'
            else:
                time_txt = time_txt + ' - ' + f'{time_round}'
        print(f"Quantidade de ondas quadradas detectadas: {len(square_waves)}")
        writer.writerow([file_name, len(flush_idx), len(square_waves) - len(flush_idx), time_txt])

#def param_optimization():
#    pass

def detection_results(fileToSaveCsv):
    # Cria diretóriio se não existir
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME, exist_ok=True)
    csv_path = FOLDER_NAME + '/' + fileToSaveCsv    
    # Cria o arquivo vazio
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['arquivo', 'num__true_flush_tests', 'num_false_flush_tests', 't_descida'])
    #for i in range(1,101):
    for i in range(1, 101):
        if len(str(i)) == 1:
            file_suffix = f'00{i}'
        elif len(str(i)) == 2:
            file_suffix = f'0{i}'
        else:
            file_suffix = f'{i}'
        print(f'Processing file: pressao_sim_{file_suffix}.csv')
        file_path = '../vitaldb_project/SIM/' + f'pressao_sim_{file_suffix}.csv'
        time, step_detection, signal_smooth, triangle_detection, derivative_smooth, step_indices, peak_indices, square_waves, flush_idx = convolution(file_path, SAMPLE_RATE, STEP_AMPLITUDE_FACTOR, DERIVATIVE_THRESHOLD_LOW, DERIVATIVE_THRESHOLD_HIGH, SIGNAL_THRESHOLD)
        csv_convolution_tests(f'pressao_sim_{file_suffix}.csv', csv_path, time, square_waves, flush_idx)
        plot_results(f'pressao_sim_{file_suffix}', time, step_detection, signal_smooth, triangle_detection, derivative_smooth, step_indices, peak_indices, flush_idx)

detection_results('step_detection_results')

plt.show()




