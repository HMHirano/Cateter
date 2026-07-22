import os
import glob
import pandas as pd
import tsfel
import tkinter as tk
from tkinter import filedialog

def selecionar_diretorio():
    """Abre uma janela para o usuário escolher a pasta."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Selecione a pasta onde estão os arquivos CSV")

def main():
    print("Aguardando a seleção da pasta...")
    diretorio_entrada = selecionar_diretorio()
    
    if not diretorio_entrada:
        print("Nenhuma pasta foi selecionada. Encerrando o programa.")
        return

    arquivos_csv = glob.glob(os.path.join(diretorio_entrada, "*.csv"))
    
    if not arquivos_csv:
        print(f"Nenhum arquivo CSV foi encontrado na pasta: {diretorio_entrada}")
        return

    print(f"Encontrados {len(arquivos_csv)} arquivo(s) CSV. Iniciando processamento...")

    diretorio_saida = os.path.join(diretorio_entrada, "resultados_tsfel")
    os.makedirs(diretorio_saida, exist_ok=True)
    
    lista_de_resultados = []

    for caminho_arquivo in arquivos_csv:
        nome_arquivo = os.path.basename(caminho_arquivo)
        print(f"Processando: {nome_arquivo}...")
        
        try:
            # 1. Lê o arquivo informando que NÃO possui cabeçalho (header=None)
            # e já batiza as duas colunas para facilitar nossa vida
            df = pd.read_csv(caminho_arquivo, header=None, names=['Tempo', 'Amplitude'])
            
            # 2. Isola APENAS a coluna de Amplitude.
            # Os colchetes duplos [['Amplitude']] garantem que continue sendo um DataFrame (exigência do TSFEL)
            df_sinal = df[['Amplitude']]
            
            # 3. Recria a configuração do TSFEL a cada loop para evitar cache
            cfg = tsfel.get_features_by_domain()
            
            # 4. Extrai as features EXCLUSIVAMENTE da coluna de Amplitude
            features = tsfel.time_series_features_extractor(cfg, df_sinal, verbose=0)
            
            # Adiciona o nome do arquivo na primeira coluna para identificação
            features.insert(0, 'Arquivo_Origem', nome_arquivo)
            
            # Guarda o resultado na lista
            lista_de_resultados.append(features)
            
        except Exception as e:
            print(f"  -> Erro ao processar o arquivo '{nome_arquivo}': {e}")
            
        except Exception as e:
            print(f"  -> Erro ao processar o arquivo '{nome_arquivo}': {e}")

    # Finalização
    if lista_de_resultados:
        print("\nConsolidando e ordenando os resultados...")
        df_final = pd.concat(lista_de_resultados, ignore_index=True)
        df_final = df_final.sort_values(by='Arquivo_Origem')
        
        caminho_saida = os.path.join(diretorio_saida, "features_consolidadas_tsfel.csv")
        df_final.to_csv(caminho_saida, index=False)
        
        print(f"Sucesso! Os resultados ordenados foram salvos em: {caminho_saida}")
    else:
        print("\nFalha: Não foi possível gerar features de nenhum dos arquivos processados.")

if __name__ == "__main__":
    main()