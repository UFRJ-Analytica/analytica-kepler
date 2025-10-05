# Em: app.py (versão final sem o loop de recarregamento)

import streamlit as st
import pandas as pd
from src.agent import ExoHunterAgent

# --- CONFIGURAÇÃO DA PÁGINA E INICIALIZAÇÃO DO AGENTE ---
st.set_page_config(page_title="ExoHunter AI", page_icon="🪐", layout="wide")

@st.cache_resource
def load_agent():
    return ExoHunterAgent()

agent = load_agent()

# Inicializa o "state" do Streamlit
if 'analysis_ready' not in st.session_state:
    st.session_state.analysis_ready = False
if 'harmonized_df' not in st.session_state:
    st.session_state.harmonized_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0


# --- INTERFACE PRINCIPAL ---
st.title("🪐 ExoHunter AI: Plataforma de Análise de Exoplanetas")
st.write("""
Faça o upload de seu arquivo CSV. O Agente de IA irá analisar cada candidato, harmonizar os dados para nosso modelo padrão e fornecer um relatório completo de predições.
""")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Carregue seu arquivo CSV", 
    type=['csv'], 
    key=st.session_state.upload_key # Usar uma chave para poder resetar
)

if uploaded_file:
    try:
        user_df = pd.read_csv(uploaded_file, comment='#')
        if len(user_df.columns) == 1:
            uploaded_file.seek(0)
            header_line = uploaded_file.readline().decode('utf-8')
            cleaned_header = header_line.strip().strip('"').split(',')
            uploaded_file.seek(0)
            user_df = pd.read_csv(uploaded_file, header=None, names=cleaned_header, skiprows=1, comment='#')

        st.session_state.original_df = user_df
        st.info(f"Arquivo '{uploaded_file.name}' carregado com {len(user_df)} candidatos.")
        
        known_format = agent._identify_format_robust(user_df)

        if known_format != 'unknown':
            parser_func = agent.parsers[known_format]
            st.session_state.harmonized_df = parser_func(user_df)
            st.session_state.analysis_ready = True
            # A linha st.rerun() foi REMOVIDA daqui

        else: # Lógica da IA
            st.warning("Formato de dados desconhecido. Ativando módulo de IA para sugerir mapeamento...")
            with st.spinner("A IA (Gemini) está analisando as colunas..."):
                suggested_map = agent.get_ai_suggested_mapping(user_df.columns.tolist())
            
            if not suggested_map:
                st.error("A IA não conseguiu encontrar um mapeamento confiável.")
            else:
                with st.form("mapping_form"):
                    st.subheader("Validação do Mapeamento Sugerido pela IA")
                    final_mapping = {}
                    model_features = agent.model.feature_names_in_.tolist()
                    for original_col, suggested_col in suggested_map.items():
                        col1, col2 = st.columns(2)
                        col1.text(f"Coluna do seu arquivo: `{original_col}`")
                        try: sugg_index = model_features.index(suggested_col)
                        except ValueError: sugg_index = 0
                        user_choice = col2.selectbox(f"Mapear para:", model_features, index=sugg_index, key=original_col)
                        final_mapping[original_col] = user_choice
                    
                    submitted = st.form_submit_button("Confirmar Mapeamento e Harmonizar")
                    if submitted:
                        st.session_state.harmonized_df = agent.apply_custom_mapping(user_df, final_mapping)
                        st.session_state.analysis_ready = True
                        # A linha st.rerun() foi REMOVIDA daqui também

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")


if st.session_state.analysis_ready:
    st.success("Dados harmonizados e prontos para análise!")
    
    if st.button("Executar Análise com o Modelo de ML"):
        with st.spinner('O Agente ExoHunter está analisando os dados...'):
            # ... (código de análise otimizado) ...
            harmonized_df = st.session_state.harmonized_df.copy()
            predictions = agent.model.predict(harmonized_df)
            probabilities = agent.model.predict_proba(harmonized_df)
            class_map = agent.model.classes_.tolist()
            prob_confirmed_col_index = class_map.index('CONFIRMED')
            results_df = pd.DataFrame({'prediction': predictions, 'prob_confirmed': probabilities[:, prob_confirmed_col_index]})
            def generate_insight(pc):
                if pc > 0.9: return "Alta Prioridade..."
                if pc > 0.7: return "Candidato Promissor..."
                if 0.4 <= pc <= 0.6: return "⚠️ Alerta de Ambiguidade..."
                return "Provável Falso Positivo..."
            results_df['insight'] = results_df['prob_confirmed'].apply(generate_insight)
            results_df['confidence'] = results_df.apply(lambda row: row['prob_confirmed'] if row['prediction'] == 'CONFIRMED' else (1 - row['prob_confirmed']), axis=1)
            final_df = pd.concat([st.session_state.original_df.reset_index(drop=True), results_df], axis=1)

        st.header("Dashboard de Resultados da Análise")
        # ... (código do dashboard) ...
        prediction_counts = final_df['prediction'].value_counts()
        confirmed_count = prediction_counts.get('CONFIRMED', 0)
        fp_count = prediction_counts.get('FALSE POSITIVE', 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(final_df)); col2.metric("🪐 Planetas", confirmed_count); col3.metric("☄️ Falsos Positivos", fp_count)
        st.bar_chart(prediction_counts)
        st.header("Resultados Detalhados")
        st.dataframe(final_df)
        
        # Botão para resetar e analisar um novo arquivo
        if st.button("Analisar Novo Arquivo"):
            st.session_state.clear()
            st.rerun()