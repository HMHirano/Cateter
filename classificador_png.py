#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:26:54 2026

@author: heitor
"""

import os
import glob
import pandas as pd
import tkinter as tk
import re
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # Necessário: pip install Pillow

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PNG Signal Classifier")
        self.root.geometry("1920x1080")
        
        # Configuração de DPI para Windows
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        # Estilos Visuais
        self.style = ttk.Style(self.root)
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
            
        self.style.configure("Flush.TButton", font=("Segoe UI", 12, "bold"), background="#d4edda", padding=10)
        self.style.configure("NotFlush.TButton", font=("Segoe UI", 12, "bold"), background="#f8d7da", padding=10)
        self.style.configure("Standard.TButton", font=("Segoe UI", 11), background="#e2e3e5", padding=10)
        
        # --- CONFIGURAÇÕES DE CAMINHO ---
        self.directory = "resultados_detector_flushlike" # Pasta com as imagens
        self.output_file = "classifications_png.csv"
        
        self.image_files = self.get_image_files(self.directory)
        self.results_dict = {}
        
        # Carregar progresso anterior
        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_csv(self.output_file)
                self.results_dict = {row['filename']: row['classification'] for _, row in existing_df.iterrows()}
            except Exception as e:
                print(f"Erro ao carregar classificações: {e}")

        # Lógica para continuar de onde parou
        self.current_index = 0
        self.zoom_factor = 1.0
        for i, file_path in enumerate(self.image_files):
            if os.path.basename(file_path) not in self.results_dict:
                self.current_index = i
                break
        else:
            if self.image_files:
                self.current_index = len(self.image_files)

        self.setup_ui()
        
        if not self.image_files:
            messagebox.showinfo("Sem arquivos", f"Nenhum arquivo PNG encontrado em: {self.directory}")
            self.root.destroy()
        else:
            self.load_current_image()

    def get_image_files(self, directory):
        # Busca apenas arquivos PNG
        files = glob.glob(os.path.join(directory, "*.png"))
        
        # Função interna para separar texto e números para a ordenação natural
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
        # Ordena usando a chave de ordenação natural
        return sorted(files, key=natural_sort_key)

    def setup_ui(self):
        # Frame para a Imagem (Substitui o Canvas do Matplotlib)
        self.image_container = ttk.Frame(self.root)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.image_label = ttk.Label(self.image_container)
        self.image_label.pack(expand=True)

        # Frame de Botões
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20, padx=20)
        
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)
        
        self.btn_flush = ttk.Button(btn_frame, text="Flush\n(Left Ctrl)", command=self.mark_flush, style="Flush.TButton")
        self.btn_flush.grid(row=0, column=0, sticky="ew", padx=10)
        
        self.btn_back = ttk.Button(btn_frame, text="Back\n(b)", command=self.go_back, style="Standard.TButton")
        self.btn_back.grid(row=0, column=1, sticky="ew", padx=10)
        
        self.btn_quit = ttk.Button(btn_frame, text="Quit\n(q)", command=self.finish_and_save, style="Standard.TButton")
        self.btn_quit.grid(row=0, column=2, sticky="ew", padx=10)
        
        self.btn_not_flush = ttk.Button(btn_frame, text="Not Flush\n(Right Ctrl)", command=self.mark_not_flush, style="NotFlush.TButton")
        self.btn_not_flush.grid(row=0, column=3, sticky="ew", padx=10)

        # Atalhos de teclado
        self.root.bind('<Control_L>', lambda event: self.mark_flush())
        self.root.bind('<Control_R>', lambda event: self.mark_not_flush())
        self.root.bind('b', lambda event: self.go_back())
        self.root.bind('q', lambda event: self.finish_and_save())
        
        # Novos atalhos para Zoom
        self.root.bind('<plus>', lambda event: self.zoom_in())
        self.root.bind('<minus>', lambda event: self.zoom_out())
        # Para teclados numéricos (numpad):
        self.root.bind('<KP_Add>', lambda event: self.zoom_in())
        self.root.bind('<KP_Subtract>', lambda event: self.zoom_out())
        
        self.root.focus_force()

    def load_current_image(self):
        if self.current_index >= len(self.image_files):
            self.finish_and_save()
            return
            
        img_path = self.image_files[self.current_index]
        basename = os.path.basename(img_path)
        self.root.title(f"Classifier - {basename} ({self.current_index + 1}/{len(self.image_files)})")
        
        try:
            # Salva a imagem original na memória
            self.original_img = Image.open(img_path)
            self.zoom_factor = 1.0 # Reseta o zoom ao trocar de imagem
            self.display_image() # Chama a função que desenha na tela
            
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}")
            self.current_index += 1
            self.load_current_image()

    def display_image(self):
        # Aumentamos a base inicial para aproveitar melhor sua tela 1920x1080
        base_width, base_height = 1600, 900
        
        # Aplica o fator de zoom
        new_size = (int(base_width * self.zoom_factor), int(base_height * self.zoom_factor))
        
        # Cria uma cópia para não alterar a original permanentemente
        img_copy = self.original_img.copy()
        img_copy.thumbnail(new_size, Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image=img_copy, master=self.root)
        self.image_label.config(image=self.photo)

    def zoom_in(self):
        self.zoom_factor += 0.2  # Aumenta 20% a cada aperto
        self.display_image()

    def zoom_out(self):
        if self.zoom_factor > 0.2: # Impede que o zoom fique negativo ou a imagem suma
            self.zoom_factor -= 0.2
            self.display_image()

    def mark_flush(self):
        self.save_classification('flush')
        
    def mark_not_flush(self):
        self.save_classification('not_flush')

    def go_back(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def save_classification(self, classification):
        if self.current_index < len(self.image_files):
            basename = os.path.basename(self.image_files[self.current_index])
            self.results_dict[basename] = classification
            self.save_to_csv(intermediate=True)
            self.current_index += 1
            self.load_current_image()

    def save_to_csv(self, intermediate=False):
        df = pd.DataFrame(list(self.results_dict.items()), columns=['filename', 'classification'])
        if not intermediate:
            df = df.sort_values(by=['filename'])
        df.to_csv(self.output_file, index=False)

    def finish_and_save(self, event=None):
        # Desvincula os atalhos para evitar comandos acidentais durante o salvamento
        self.root.unbind('<Control_L>')
        self.root.unbind('<Control_R>')
        self.root.unbind('b')
        self.root.unbind('q')
        
        # Salva o progresso e mostra a mensagem adequada
        if self.results_dict:
            self.save_to_csv(intermediate=False)
            messagebox.showinfo("Complete", f"All files classified!\n\nOrganized alphabetically and saved to:\n{self.output_file}")
        else:
            messagebox.showinfo("Complete", "No classifications were made.")
            
        # Encerra o loop e destrói a interface gráfica (fecha a janela)
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    # Certifique-se de que a pasta existe para não dar erro ao iniciar
    os.makedirs("resultados_detector/flush_png", exist_ok=True)
    
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.finish_and_save) 
    root.mainloop()