# Em: src/agent.py (Versão Final Sincronizada)

import pandas as pd
import joblib
import os
import sys
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# --- Bloco de Importação Robusto ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from processing import preprocess_and_engineer_features

load_dotenv()

class ExoHunterAgent:
    def __init__(self, model_path="models/robust_generalist_model.pkl"):
        print("--- AGENT (Final): Iniciando __init__. ---")
        self.target_schema = {
            'orbital_period': 'O período orbital do planeta em dias.',
            'transit_duration': 'A duração do trânsito em horas.',
            'transit_depth': 'A profundidade do trânsito (a fração da luz estelar bloqueada).',
            'planet_radius': 'O raio do planeta em unidades de raios terrestres.',
            'stellar_radius': 'O raio da estrela em unidades de raios solares.',
            'impact_parameter': 'O parâmetro de impacto do trânsito (quão central é a passagem).',
            'stellar_gravity': 'A gravidade na superfície da estrela (log(g)), usada para calcular a densidade.'
        }
        
        artifact = self._load_artifact(model_path)
        if artifact:
            self.model = artifact['model']
            self.imputer = artifact['imputer']
            self.model_features = artifact['features']
            self.model_accuracy = artifact.get('accuracy', 0.0)
            print(f"--- AGENT (Final): Artefacto OK, {len(self.model_features)} features. ---")
        else:
            self.model, self.imputer, self.model_features, self.model_accuracy = None, None, list(self.target_schema.keys()), 0.0

        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self.genai_model = genai.GenerativeModel('gemini-1.0-pro')
            print("--- AGENT (Final): Gemini OK. ---")
        except Exception as e:
            self.genai_model = None
            print(f"--- AVISO AGENT (Final): Gemini falhou: {e} ---")

    def _load_artifact(self, path):
        try:
            return joblib.load(path)
        except FileNotFoundError:
            print(f"--- ERRO AGENT (Final): Modelo '{path}' não encontrado. ---")
            return None

    def suggest_disposition_column(self, available_columns: list) -> str | None:
        """Encontra a coluna de resultado ('disposition') de forma determinística e rápida."""
        print("--- AGENT (Final): Procurando pela coluna de resultado... ---")
        for col in available_columns:
            col_lower = col.lower()
            if 'disposition' in col_lower or 'disp' in col_lower:
                print(f"--- AGENT (Final): Coluna de resultado encontrada: {col} ---")
                return col
        return None

    def suggest_feature_mapping(self, available_columns: list, column_descriptions: dict, df_sample: pd.DataFrame = None, use_gemini: bool = True, find_disposition_col: bool = False) -> dict:
        """
        Motor de mapeamento unificado e otimizado. Usa um estágio determinístico rápido e,
        opcionalmente, um estágio de IA para preencher as lacunas.
        """
        print("\n--- AGENT (Final): Iniciando Motor de Mapeamento Unificado. ---")
        
        # --- Estágio 1: Determinístico (Rápido) ---
        print("--- AGENT (Final): Estágio 1 - Determinístico. ---")
        suggested_map = {}
        common_name_map = {
            'pl_orbper': 'orbital_period', 'koi_period': 'orbital_period',
            'pl_trandurh': 'transit_duration', 'koi_duration': 'transit_duration', 'pl_trandur': 'transit_duration',
            'pl_trandep': 'transit_depth', 'koi_depth': 'transit_depth',
            'pl_rade': 'planet_radius', 'koi_prad': 'planet_radius',
            'st_rad': 'stellar_radius', 'koi_srad': 'stellar_radius',
            'pl_imppar': 'impact_parameter', 'koi_impact': 'impact_parameter',
            'st_logg': 'stellar_gravity', 'koi_slogg': 'stellar_gravity'
        }
        for orig_col in available_columns:
            col_lower = orig_col.lower()
            if col_lower in common_name_map:
                target = common_name_map[col_lower]
                if target not in suggested_map.values():
                    suggested_map[orig_col] = target
        
        print(f"--- AGENT (Final): Estágio 1: {len(suggested_map)}/{len(self.target_schema)} mapeadas.")

        # --- Estágio 2: Gemini Otimizado (Opcional) ---
        features_to_find = {k:v for k, v in self.target_schema.items() if k not in suggested_map.values()}
        
        if use_gemini and column_descriptions and features_to_find and self.genai_model:
            print("--- AGENT (Final): Estágio 2 - Gemini Otimizado. ---")
            available_cols_with_desc = {k: v for k, v in column_descriptions.items() if k not in suggested_map.keys()}
            
            prompt = f"""
            Você é um especialista em astrofísica. Faça um "match" semântico entre as features que eu preciso (NEEDED) e as colunas disponíveis, usando APENAS as suas descrições oficiais.
            Features NEEDED: {json.dumps(features_to_find, indent=2)}
            Colunas disponíveis (com descrições): {json.dumps(available_cols_with_desc, indent=2)}
            Output EXATO: JSON com a chave "column_mappings". Exemplo: {{"pl_trandur": "transit_duration"}}
            """
            try:
                response = self.genai_model.generate_content(prompt)
                cleaned = response.text.strip()
                json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if json_match:
                    ai_map = json.loads(json_match.group()).get("column_mappings", {})
                    for col, feature in ai_map.items():
                        original_cased_col = next((c for c in available_columns if c.lower() == col.lower()), None)
                        if original_cased_col and feature in features_to_find:
                            suggested_map[original_cased_col] = feature
                    print(f"--- AGENT (Final): Estágio 2: +{len(ai_map)} mapeamentos. Total: {len(suggested_map)}.")
            except Exception as e:
                print(f"--- AVISO AGENT (Final): Gemini parse falhou: {e}. ---")

        # --- Lógica Adicional para Treino ---
        if find_disposition_col:
            disposition_col = self.suggest_disposition_column(available_columns)
            if disposition_col:
                suggested_map[disposition_col] = 'resultado'

        print(f"--- AGENT (Final): Mapeamento final: {suggested_map} ---")
        return suggested_map

    def harmonize_and_prepare_data(self, df: pd.DataFrame, user_validated_mapping: dict) -> pd.DataFrame:
        """Harmoniza e prepara os dados para o modelo."""
        print("--- AGENT (Final): Harmonizando dados. ---")
        df_renamed = df.rename(columns=user_validated_mapping)
        df_processed = preprocess_and_engineer_features(df_renamed)
        
        # Reindexar para garantir que todas as features do modelo estão presentes
        X_final_schema = df_processed.reindex(columns=self.model_features)
        
        # Imputar os dados em falta
        X_imputed = self.imputer.transform(X_final_schema)
        
        # Retornar um DataFrame com os dados preparados e nomes de colunas corretos
        prepared_df = pd.DataFrame(X_imputed, columns=self.model_features)
        return prepared_df

    def classify_candidates(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """Executa a classificação nos dados preparados."""
        print(f"--- AGENT (Final): Classificando {len(prepared_df)} candidatos. ---")
        if not self.model:
            raise ValueError("Modelo não carregado!")
        
        predictions = self.model.predict(prepared_df)
        probabilities = self.model.predict_proba(prepared_df)
        class_map = self.model.classes_.tolist()
        
        try:
            prob_idx = class_map.index('CONFIRMED')
            prob_confirmed = probabilities[:, prob_idx]
        except ValueError:
            prob_confirmed = [0.0] * len(prepared_df)
        
        results_df = pd.DataFrame(index=prepared_df.index)
        results_df['prediction'] = predictions
        results_df['probability_confirmed'] = prob_confirmed
        
        print("--- AGENT (Final): Classificação OK. ---")
        return results_df