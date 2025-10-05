# Em: test_gemini.py (versão de diagnóstico final)

import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def final_diagnostic_test(file_path):
    # --- Carrega as colunas do seu arquivo CSV de forma robusta ---
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline()
        cleaned_header = header_line.strip().strip('"')
        df_columns = cleaned_header.split(',')
        print(f"--- Colunas encontradas no arquivo (corrigido) ---\n{df_columns}\n")
    except Exception as e:
        print(f"Não foi possível ler o arquivo CSV: {e}")
        return

    # --- Configura a API do Gemini ---
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # PASSO DE DIAGNÓSTICO: Listar todos os modelos disponíveis
        print("--- Verificando modelos disponíveis para sua API Key ---")
        model_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(model_list)
        print("-----------------------------------------------------\n")

        # Usando o nome de modelo mais básico e universal
        model_name = 'gemini-pro'
        if f'models/{model_name}' not in model_list:
            print(f"AVISO: O modelo '{model_name}' não foi encontrado na lista. Tentando mesmo assim.")

        model = genai.GenerativeModel(model_name)
        print("--- Configuração da API do Gemini OK ---\n")
    except Exception as e:
        print(f"AVISO: Não foi possível configurar a API do Gemini ou listar modelos. Verifique sua API Key. Erro: {e}")
        return

    # --- Cria o Prompt ---
    standard_schema_str = "['orbital_period', 'transit_duration', 'transit_depth', 'planet_radius', 'stellar_radius']"
    prompt = f"Contexto: ...\nEsquema Padrão: {standard_schema_str}\nColunas do Novo Dataset: {df_columns}\nTarefa: ..." # Prompt abreviado para clareza
    
    # --- Envia o Prompt e Imprime a Resposta Bruta ---
    try:
        print("--- Enviando prompt para o modelo 'gemini-pro' ---")
        response = model.generate_content(prompt)
        print("\n--- RESPOSTA BRUTA DA IA ---")
        print(response.text)
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(cleaned_response)
        print("\n--- JSON INTERPRETADO ---")
        print(parsed_json)
    except Exception as e:
        print(f"\n--- ERRO AO CHAMAR A API OU PROCESSAR A RESPOSTA ---")
        print(e)


# --- EXECUÇÃO DO TESTE ---
if __name__ == "__main__":
    csv_file_to_test = r"C:\Users\lucas\Downloads\cumulative_2025.10.04_17.45.56.csv"
    final_diagnostic_test(csv_file_to_test)