#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:26:54 2026

@author: heitor
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PNG Signal Classifier")
        self.root.geometry("1920x1080")
        
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        self.style = ttk.Style(self.root)
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
            
        self.style.configure("Flush.TButton", font=("Segoe UI", 12, "bold"), background="#d4edda", padding=10)
        self.style.configure("NotFlush.TButton", font=("Segoe UI", 12, "bold"), background="#f8d7da", padding=10)
        self.style.configure("Standard.TButton", font=("Segoe UI", 11), background="#e2e3e5", padding=10)
        
        # --- CONFIGURAÇÕES DE CAMINHO ROBUSTAS ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.directory = os.path.join(script_dir, "resultados_detector_flushlike") 
        self.output_file = os.path.join(self.directory, "todas_posicoes_flushes.csv")
        
        if os.path.exists(self.output_file):
            try:
                self.df = pd.read_csv(self.output_file)
                if 'Image_Name' not in self.df.columns:
                    raise ValueError("A coluna 'Image_Name' não foi encontrada no arquivo CSV.")
                
                if 'classification' not in self.df.columns:
                    self.df['classification'] = None
            except Exception as e:
                messagebox.showerror("Erro no CSV", f"Erro ao carregar o arquivo CSV: {e}")
                self.root.destroy()
                return
        else:
            messagebox.showerror("Erro", f"O arquivo CSV não foi encontrado em:\n{self.output_file}")
            self.root.destroy()
            return

        self.zoom_factor = 1.0
        
        # Inicia buscando o primeiro arquivo não classificado
        self.current_index = self.encontrar_proximo_vazio()

        self.setup_ui()
        
        if self.df.empty:
            messagebox.showinfo("Sem arquivos", "O arquivo CSV está vazio.")
            self.root.destroy()
        else:
            self.load_current_image()

    def encontrar_proximo_vazio(self):
        """
        Percorre o DataFrame do início ao fim procurando a primeira linha 
        que não possua classificação. Retorna o índice ou len(df) se acabar.
        """
        for i in range(len(self.df)):
            val = self.df.loc[i, 'classification']
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                return i
        return len(self.df)

    def setup_ui(self):
        self.image_container = ttk.Frame(self.root)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.image_label = ttk.Label(self.image_container)
        self.image_label.pack(expand=True)

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

        self.root.bind('<Control_L>', lambda event: self.mark_flush())
        self.root.bind('<Control_R>', lambda event: self.mark_not_flush())
        self.root.bind('b', lambda event: self.go_back())
        self.root.bind('q', lambda event: self.finish_and_save())
        
        self.root.bind('<plus>', lambda event: self.zoom_in())
        self.root.bind('<minus>', lambda event: self.zoom_out())
        self.root.bind('<KP_Add>', lambda event: self.zoom_in())
        self.root.bind('<KP_Subtract>', lambda event: self.zoom_out())
        
        self.root.focus_force()

    def load_current_image(self):
        while self.current_index < len(self.df):
            filename = self.df.loc[self.current_index, 'Image_Name']
            filename = str(filename).strip()
            
            if not filename.lower().endswith('.png'):
                filename += '.png'
            
            pure_filename = os.path.basename(filename)
            img_path = os.path.join(self.directory, pure_filename)
            
            try:
                self.original_img = Image.open(img_path)
                self.zoom_factor = 1.0 
                
                basename = os.path.basename(img_path)
                self.root.title(f"Classifier - {basename} ({self.current_index + 1}/{len(self.df)})")
                
                self.display_image() 
                return 
                
            except Exception as e:
                print(f"[AVISO] Arquivo não encontrado: {img_path}")
                # Se não achar a imagem, força ela como "Error" ou pula para a próxima vazia
                self.current_index += 1
                
        self.finish_and_save()

    def display_image(self):
        base_width, base_height = 1600, 900
        new_size = (int(base_width * self.zoom_factor), int(base_height * self.zoom_factor))
        
        img_copy = self.original_img.copy()
        img_copy.thumbnail(new_size, Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image=img_copy, master=self.root)
        self.image_label.config(image=self.photo)

    def zoom_in(self):
        self.zoom_factor += 0.2  
        self.display_image()

    def zoom_out(self):
        if self.zoom_factor > 0.2: 
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
        if self.current_index < len(self.df):
            # Salva o valor atual
            self.df.at[self.current_index, 'classification'] = classification
            self.save_to_csv()
            
            # PULA DIRETAMENTE PARA O PRÓXIMO NÃO CLASSIFICADO (em vez de += 1)
            self.current_index = self.encontrar_proximo_vazio()
            self.load_current_image()

    def save_to_csv(self):
        self.df.to_csv(self.output_file, index=False)

    def finish_and_save(self, event=None):
        self.root.unbind('<Control_L>')
        self.root.unbind('<Control_R>')
        self.root.unbind('b')
        self.root.unbind('q')
        
        self.save_to_csv()
        messagebox.showinfo("Complete", f"Progresso salvo com sucesso na coluna 'classification' de:\n{self.output_file}")
            
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.protocol("WM_DELETE_WINDOW", app.finish_and_save) 
    root.mainloop()