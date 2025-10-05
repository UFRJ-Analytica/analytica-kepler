# Em: src/agent.py

import pandas as pd
import joblib
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from src.parsers import parse_koi_data, parse_toi_data

# Carrega as variáveis de ambiente (sua API key) do arquivo .env
load_dotenv()

class ExoHunterAgent:
    def __init__(self, model_path="models/harmonized_rf_model.pkl"):
        self.model = self._load_model(model_path)
        self.parsers = {
            'koi': parse_koi_data,
            'toi': parse_toi_data
        }
        # Configura a API do Gemini
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.genai_model = genai.GenerativeModel('gemini-pro-latest')
        except Exception as e:
            print(f"AVISO: Não foi possível configurar a API do Gemini. A análise por IA estará desabilitada. Erro: {e}")
            self.genai_model = None

    def _load_model(self, path):
        # ... (código igual ao anterior) ...
        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"ERRO: Modelo não encontrado em '{path}'")
            return None

    def _identify_format_robust(self, df: pd.DataFrame) -> str:
        # ... (código da função de identificação robusta que já fizemos) ...
        columns = set(df.columns.str.lower())
        koi_fingerprint = {'kepid', 'koi_period', 'koi_depth'}
        if koi_fingerprint.issubset(columns): return 'koi'
        toi_fingerprint = {'tid', 'pl_orbper', 'pl_trandep'}
        if toi_fingerprint.issubset(columns): return 'toi'
        if 'koi_disposition' in columns: return 'koi'
        if 'tfopwg_disp' in columns: return 'toi'
        return 'unknown'

    def get_ai_suggested_mapping(self, df_columns: list) -> dict:
        """Usa a API do Gemini para sugerir um mapeamento de colunas."""
        if not self.genai_model:
            return {}

        standard_schema_str = "['orbital_period', 'transit_duration', 'transit_depth', 'planet_radius', 'stellar_radius']"
        
        prompt = f"""
        Contexto: Você é um subsistema de IA especialista em dados astronômicos para o agente ExoHunter.
        Sua tarefa é mapear as colunas de um novo dataset de exoplanetas para um esquema padrão.
        
        Esquema Padrão (nomes que eu entendo): 
        {standard_schema_str}

        Colunas do Novo Dataset (nomes que eu não conheço): 
        {df_columns}
        
        Tarefa: Analise as 'Colunas do Novo Dataset' e retorne um objeto JSON que mapeia os nomes desconhecidos para os nomes do 'Esquema Padrão'. 
        O JSON deve ter a chave 'column_mappings' e o valor deve ser um dicionário.
        Exemplo de saída: {{"column_mappings": {{"koi_period": "orbital_period", "koi_depth": "transit_depth"}}}}
        Inclua apenas os mapeamentos dos quais você tem alta confiança.
        """
        
        try:
            response = self.genai_model.generate_content(prompt)
            # Limpeza básica da resposta para extrair o JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response).get("column_mappings", {})
        except Exception as e:
            print(f"Erro ao chamar a API do Gemini ou ao processar a resposta: {e}")
            return {}

    def apply_custom_mapping(self, df: pd.DataFrame, custom_mapping: dict) -> pd.DataFrame:
        """Aplica um mapeamento fornecido pelo usuário para harmonizar o DataFrame."""
        harmonized_df = df.rename(columns=custom_mapping)

        # Lógica de conversão de unidades pode ser adicionada aqui se necessário
        # Ex: if 'transit_depth' in harmonized_df.columns: ...

        # Garante a presença e ordem correta das colunas para o modelo
        if self.model:
            model_features = self.model.feature_names_in_
            if 'mission_TESS' not in harmonized_df.columns:
                 # Por padrão, assumimos que dados desconhecidos se assemelham mais ao TESS
                 harmonized_df['mission_TESS'] = 1 
            harmonized_df = harmonized_df.reindex(columns=model_features)

        return harmonized_df


    def analyze_candidate(self, harmonized_df: pd.DataFrame) -> dict:
        # ... (código igual ao anterior, sem alterações) ...
        if self.model is None:
            raise ConnectionError("Modelo de Machine Learning não foi carregado.")
        prediction = self.model.predict(harmonized_df)[0]
        probabilities = self.model.predict_proba(harmonized_df)[0]
        class_map = self.model.classes_.tolist()
        prob_confirmed = probabilities[class_map.index('CONFIRMED')]
        insight = ""
        if prob_confirmed > 0.90: insight = "Candidato de Alta Prioridade..."
        elif prob_confirmed > 0.70: insight = "Candidato Promissor..."
        elif 0.40 <= prob_confirmed <= 0.60: insight = "⚠️ Alerta de Ambiguidade..."
        else: insight = "Provável Falso Positivo..."
        return { "prediction": prediction, "confidence": prob_confirmed if prediction == 'CONFIRMED' else (1 - prob_confirmed), "insight": insight }