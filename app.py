# Em: app.py (Versão 10.0 - Final, Corrigida e Verificada)

import streamlit as st
import pandas as pd
import os
import sys
import traceback
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# --- Bloco de Importação Robusto ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from processing import preprocess_and_engineer_features
from agent import ExoHunterAgent
from train import main as run_training_pipeline
from file_handler import read_exoplanet_data

# --- Configuração da Página e Carregamento do Agente ---
st.set_page_config(page_title="ExoHunter AI", page_icon="🪐", layout="wide")

@st.cache_resource
def load_agent(model_path="models/robust_generalist_model.pkl"):
    st.spinner(f"Inicializando a Consciência da IA com modelo {os.path.basename(model_path)}...")
    return ExoHunterAgent(model_path=model_path)

if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = "models/robust_generalist_model.pkl"

agent = load_agent(st.session_state.current_model_path)

st.title("🪐 ExoHunter AI: Your Generalist Discovery Partner")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🧠 AI Brain Details")
    if agent.model:
        model_name = os.path.basename(st.session_state.current_model_path)
        st.markdown(f"**Current Model:** `{model_name}`")
        acc_value = agent.model_accuracy * 100
        st.metric("Accuracy (on last training)", f"{acc_value:.2f}%")
        with st.expander("View the 7 essential model features"):
            st.json(agent.target_schema)
        if st.button("🔄 Revert to Initial Model"):
            st.session_state.current_model_path = "models/robust_generalist_model.pkl"
            st.cache_resource.clear()
            st.rerun()
    else:
        st.error("Model not loaded. Please run the training in the 'Manage AI Model' tab.")

# --- Estrutura de Abas ---
tab1, tab2 = st.tabs(["🔭 Classify Data from Any Mission", "⚙️ Manage AI Model"])

# ==============================================================================
# --- ABA 1: CLASSIFICAR DADOS ---
# ==============================================================================
with tab1:
    st.header("Candidate Analysis with the Engine")

    # (A sua lógica completa da Aba 1, que já estava a funcionar bem, deve estar aqui)
    # Para garantir que o ficheiro funcione, colei a sua última versão funcional da Aba 1.
    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
    if 'step' not in st.session_state: st.session_state.step = "upload"
    if 'original_df' not in st.session_state: st.session_state.original_df = None
    if 'column_descriptions' not in st.session_state: st.session_state.column_descriptions = None
    if 'suggested_map' not in st.session_state: st.session_state.suggested_map = None
    if 'final_results_df' not in st.session_state: st.session_state.final_results_df = None

    if st.button("Analyze New File"):
        keys_to_clear = ['step', 'original_df', 'column_descriptions', 'suggested_map', 'final_results_df']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.uploader_key += 1
        st.rerun()

    if st.session_state.step == "upload":
        uploaded_file = st.file_uploader(
            "Upload a file from any mission (KOI, TOI, K2, etc.)",
            type=['csv'],
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if uploaded_file:
            with st.spinner("Analyzing file structure and extracting metadata..."):
                df, descriptions = read_exoplanet_data(uploaded_file)
                if not df.empty:
                    st.session_state.original_df = df.reset_index(drop=True)
                    st.session_state.column_descriptions = descriptions
                    st.session_state.step = "mapping"
                    st.rerun()
                else:
                    st.error("Could not read the data from this file. Please check if it is a valid CSV.")

    if st.session_state.step == "mapping":
        st.subheader("Step 2: Feature Mapping Validation")
        
        if st.session_state.suggested_map is None:
            with st.spinner("The AI Agent is analyzing columns and descriptions to suggest a mapping..."):
                st.session_state.suggested_map = agent.suggest_feature_mapping(
                    st.session_state.original_df.columns.tolist(),
                    st.session_state.column_descriptions
                )
        
        suggested_map = st.session_state.suggested_map
        reversed_suggested_map = {v: k for k, v in suggested_map.items()}

        st.info("Below are the 7 features our model requires. Please validate and complete the mapping if necessary.")

        with st.form("mapping_form"):
            final_mapping = {}
            available_columns = ["(Não mapear)"] + st.session_state.original_df.columns.tolist()
            
            for feature_key, feature_desc in agent.target_schema.items():
                suggested_col = reversed_suggested_map.get(feature_key)
                try:
                    default_index = available_columns.index(suggested_col) if suggested_col else 0
                except ValueError:
                    default_index = 0

                selected_col = st.selectbox(
                    label=f"**{feature_key.replace('_', ' ').title()}**",
                    options=available_columns, index=default_index, help=feature_desc
                )
                
                if selected_col != "(Não mapear)":
                    final_mapping[selected_col] = feature_key
            
            if st.form_submit_button("Confirm Mapping and Start Classification", type="primary"):
                st.session_state.step = "classify"
                st.session_state.final_mapping = final_mapping
                st.rerun()

    if st.session_state.step == "classify":
        try:
            with st.spinner('The Agent is harmonizing units, creating features, and classifying the candidates...'):
                prepared_df = agent.harmonize_and_prepare_data(st.session_state.original_df, st.session_state.final_mapping)
                st.write(f"DEBUG: Len prepared_df: {len(prepared_df)}")
                st.write(f"DEBUG: Sample prepared_df features: {prepared_df[agent.model_features].head(1)}")
                
                results_df = agent.classify_candidates(prepared_df)
                st.write(f"DEBUG: Len results_df: {len(results_df)}")
                st.write(f"DEBUG: Probs mean: {results_df['probability_confirmed'].mean():.2%}")
                
                # Lógica de Priorização (ajustada para mais sensível)
                def assign_priority(prob):
                    if prob > 0.70: return "P1: Urgent Validation"
                    if prob > 0.50: return "P2: Follow-up Recommended"
                    if prob > 0.30: return "P3: Further Analysis"  # NOVO: Mais baixo para K2
                    return "P4: Low Priority"
                results_df['priority'] = results_df['probability_confirmed'].apply(assign_priority)
                results_df['confidence'] = results_df.apply(lambda r: r['probability_confirmed'] if r['prediction'] == 'CONFIRMED' else 1 - r['probability_confirmed'], axis=1)

                # Concat seguro
                final_df = st.session_state.original_df.join(results_df.drop(columns=st.session_state.original_df.columns, errors='ignore'), how='left').reset_index(drop=True)
                st.write(f"DEBUG: Len final_df: {len(final_df)}")
                
                # NOVO: Destaque confirmados reais se 'disposition' existir
                if 'disposition' in final_df.columns:
                    real_confirmed = final_df[final_df['disposition'] == 'CONFIRMED']
                    st.metric("Candidatos Confirmados Reais (Ground Truth)", len(real_confirmed))
                    if not real_confirmed.empty:
                        st.dataframe(real_confirmed[['pl_name', 'probability_confirmed', 'prediction', 'priority']].head())
                
                st.session_state.final_results_df = final_df
                st.session_state.step = "results"
                st.rerun()
        except Exception as e:
            st.error(f"ERRO no Classify: {str(e)}")
            st.error("Traceback:")
            st.code(traceback.format_exc())
            st.stop()
            
    # ETAPA 4: DASHBOARD
    if st.session_state.step == "results":
        if st.session_state.final_results_df is None or st.session_state.final_results_df.empty:
            st.error("No results generated. Check logs above or retrain the model")
        else:
            st.header("Candidate Analysis Dashboard")
            final_df = st.session_state.final_results_df
            
            p1_count = len(final_df[final_df['priority'] == 'P1: Validação Urgente'])
            p2_count = len(final_df[final_df['priority'] == 'P2: Acompanhamento Recomendado'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Candidates Analyzed", len(final_df))
            col2.metric("🪐 P1: Urgent Validation", p1_count)
            col3.metric("✨ P2: Follow-up Recommended", p2_count)

            st.subheader("Candidate Distribution by Priority")
            priority_counts = final_df['priority'].value_counts().sort_index()
            st.bar_chart(priority_counts)
            
            st.subheader("Detailed Results")
            all_priorities = priority_counts.index.tolist()
            selected_priorities = st.multiselect("Filter by priority", options=all_priorities, default=all_priorities)
            
            filtered_df = final_df[final_df['priority'].isin(selected_priorities)]
            st.dataframe(filtered_df.style.format({'probability_confirmed': '{:.2%}', 'confidence': '{:.2%}'}), use_container_width=True)




# ==============================================================================
# --- ABA 2: GERENCIAR MODELO DE IA ---
# ==============================================================================
with tab2:
    st.header("Manage and Enhance the AI Brain")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Standard Training")
        st.info("Run the standard pipeline with the latest data from KOI and TOI.")
        if st.button("Run Standard Training", type="primary"):
            with st.spinner("Iniciando o pipeline de treinamento padrão..."):
                run_training_pipeline()
            st.success("Treinamento padrão concluído!")
            st.session_state.current_model_path = "models/robust_generalist_model.pkl"
            st.cache_resource.clear()
            st.rerun()
    
    with col_right:
        st.subheader("Load Model")
        st.info("Revert to using the standard trained model.")
        if st.button("Load Standard Model"):
            st.session_state.current_model_path = "models/robust_generalist_model.pkl"
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.subheader("Custom Training with a New Dataset")

    # --- MÁQUINA DE ESTADOS ROBUSTA PARA O TREINO CUSTOM ---
    if 'custom_step' not in st.session_state:
        st.session_state.custom_step = "start"
    if 'custom_uploader_key' not in st.session_state:
        st.session_state.custom_uploader_key = 0

    if st.session_state.custom_step == "start":
        st.info("Carregue um CSV custom, mapeie as features e a coluna de resultado, e treine um novo modelo.")
        if st.button("▶️ Start New Custom Training"):
            keys_to_clear = ['custom_df', 'custom_descriptions', 'custom_suggested_map', 'custom_final_mapping']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.custom_uploader_key += 1
            st.session_state.custom_step = "upload_custom"
            st.rerun()

    if st.session_state.custom_step == "upload_custom":
        custom_uploaded_file = st.file_uploader(
            "Upload your CSV file for training",
            type=['csv'],
            key=f"custom_uploader_{st.session_state.custom_uploader_key}"
        )
        if custom_uploaded_file:
            with st.spinner("Analisando CSV custom..."):
                custom_df, custom_descriptions = read_exoplanet_data(custom_uploaded_file)
                if not custom_df.empty:
                    st.session_state.custom_df = custom_df.reset_index(drop=True)
                    st.session_state.custom_descriptions = custom_descriptions
                    st.session_state.custom_step = "custom_mapping"
                    st.rerun()
                else:
                    st.error("CSV custom inválido.")
        if st.button("Cancel Training"):
            st.session_state.custom_step = "start"
            st.rerun()

    if st.session_state.custom_step == "custom_mapping":
        st.subheader("Mapping for Custom Training")
        if st.session_state.custom_df is not None:
            use_gemini = st.checkbox("Use AI for advanced mapping", value=False)
            
            if 'custom_suggested_map' not in st.session_state:
                with st.spinner("Agente a fazer mapeamento..."):
                    st.session_state.custom_suggested_map = agent.suggest_feature_mapping(
                        st.session_state.custom_df.columns.tolist(),
                        st.session_state.custom_descriptions,
                        use_gemini=use_gemini,
                        find_disposition_col=True
                    )
            
            custom_suggested_map = st.session_state.custom_suggested_map
            reversed_custom_map = {v: k for k, v in custom_suggested_map.items()}
            
            with st.form("custom_mapping_form"):
                custom_final_mapping = {}
                available_cols = ["(Não mapear)"] + st.session_state.custom_df.columns.tolist()
                
                st.write("**Input Feature Mapping:**")
                for feature_key, feature_desc in agent.target_schema.items():
                    suggested_col = reversed_custom_map.get(feature_key)
                    default_index = available_cols.index(suggested_col) if suggested_col in available_cols else 0
                    selected_col = st.selectbox(
                        label=f"{feature_key.replace('_', ' ').title()}",
                        options=available_cols, index=default_index, help=feature_desc
                    )
                    if selected_col != "(Não mapear)":
                        custom_final_mapping[selected_col] = feature_key
                
                st.write("**Output Column Mapping (Target):**")
                result_suggested = reversed_custom_map.get('resultado', "(Não mapear)")
                result_col = st.selectbox(
                    label="Result Column (with 'CONFIRMED'/'FALSE POSITIVE' labels)",
                    options=available_cols,
                    index=available_cols.index(result_suggested) if result_suggested in available_cols else 0,
                    help="Esta é a coluna que o modelo tentará prever."
                )
                if result_col != "(Não mapear)":
                    custom_final_mapping[result_col] = 'disposition'

                if st.form_submit_button("Train New Model with This Dataset", type="primary"):
                    if len([v for v in custom_final_mapping.values() if v != 'disposition']) < 4 or 'disposition' not in custom_final_mapping.values():
                        st.warning("Please map at least 4 input features + the result column.")
                    else:
                        st.session_state.custom_final_mapping = custom_final_mapping
                        st.session_state.custom_step = "custom_train"
                        st.rerun()

    # No app.py, na Aba 2, substitua o bloco da Etapa de Treino Custom

# ETAPA CUSTOM: Treino (Agora centralizada)
if st.session_state.custom_step == "custom_train":
    try:
        with st.spinner("Preparing custom data and initiating combined training..."):
            # Importa a nova função de treino
            from train import run_combined_training

            custom_df = st.session_state.custom_df
            custom_mapping = st.session_state.custom_final_mapping
            
            # 1. Prepara o DataFrame customizado
            rename_dict = {k: v for k, v in custom_mapping.items() if v != 'resultado'}
            target_col_original = next((k for k,v in custom_mapping.items() if v == 'resultado'), None)
            
            custom_renamed = custom_df.rename(columns=rename_dict)
            if target_col_original:
                custom_renamed['disposition'] = custom_df[target_col_original]

            custom_processed = preprocess_and_engineer_features(custom_renamed)

            # 2. Define um nome único para o novo modelo
            output_filename = f"combined_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pkl"

            # 3. Chama o motor de treino central, passando os dados customizados
            new_model_path, new_accuracy = run_combined_training(
                custom_processed_df=custom_processed,
                output_filename=output_filename
            )
            
            st.success(f"Custom training complete! New model accuracy: {new_accuracy:.2%}")
            
            # 4. Muda para o novo modelo
            st.session_state.current_model_path = new_model_path
            st.session_state.custom_step = "start" # Volta ao início do fluxo
            st.cache_resource.clear()
            st.balloons()
            st.rerun()
            
    except Exception as e:
        st.error(f"ERRO no Treino Combinado: {str(e)}")
        st.code(traceback.format_exc())
        st.session_state.custom_step = "custom_mapping" # Permite ao utilizador corrigir o mapeamento