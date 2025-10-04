import pandas as pd
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# --- 1. FUNÇÕES DE CARREGAMENTO DE DADOS ---

def fetch_toi_data():
    """Busca dados do TESS (TOI) via API da NASA Exoplanet Archive."""
    print("Buscando dados do TESS (TOI) via API...")
    TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    # Query otimizada para pegar apenas colunas necessárias e dados limpos
    query = """
    select
      tfopwg_disp, pl_orbper, pl_trandep, pl_trandurh, pl_rade, st_rad
    from toi
    where tfopwg_disp is not null and pl_orbper is not null and pl_trandep is not null
    and pl_trandurh is not null and pl_rade is not null and st_rad is not null
    """
    url = f"{TAP_BASE}?query={quote_plus(' '.join(query.split()))}&format=csv"
    try:
        df = pd.read_csv(url)
        print(f"Sucesso! {len(df)} registros do TOI carregados.")
        return df
    except Exception as e:
        print(f"Falha ao buscar dados do TOI: {e}")
        return pd.DataFrame() # Retorna dataframe vazio em caso de erro

def load_koi_data(path="datasets/cumulative_koi.csv"):
    """
    Carrega dados do Kepler (KOI) de um arquivo local.
    É necessário baixar o arquivo CSV do link abaixo e salvar na pasta 'datasets'.
    Link: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
    """
    print(f"Carregando dados do Kepler (KOI) de '{path}'...")
    if not os.path.exists(path):
        print(f"ERRO: Arquivo de dados do KOI não encontrado em '{path}'.")
        print("Por favor, baixe o arquivo do NASA Exoplanet Archive e salve-o no local correto.")
        return pd.DataFrame()
    
    df = pd.read_csv(path, comment='#')
    print(f"Sucesso! {len(df)} registros do KOI carregados.")
    return df

# --- 2. HARMONIZAÇÃO E PREPARAÇÃO ---

def harmonize_and_combine(koi_df, toi_df):
    """
    Unifica os dataframes do Kepler e TESS, harmonizando colunas e unidades.
    """
    if koi_df.empty or toi_df.empty:
        print("Um dos dataframes está vazio. Abortando a combinação.")
        return pd.DataFrame()

    print("Harmonizando datasets...")

    # --- Esquema Padrão de colunas ---
    standard_columns = [
        'disposition', 'orbital_period', 'transit_duration', 'transit_depth',
        'planet_radius', 'stellar_radius', 'mission'
    ]

    # --- Processar Kepler (KOI) ---
    koi_rename_map = {
        'koi_disposition': 'disposition', 'koi_period': 'orbital_period',
        'koi_duration': 'transit_duration', 'koi_depth': 'transit_depth',
        'koi_prad': 'planet_radius', 'koi_srad': 'stellar_radius'
    }
    koi_processed = koi_df.rename(columns=koi_rename_map)
    # Conversão de Unidade: ppm para valor decimal (1% = 10000 ppm)
    koi_processed['transit_depth'] = koi_processed['transit_depth'] / 10000.0
    koi_processed['mission'] = 'Kepler'
    koi_final = koi_processed.reindex(columns=standard_columns)

    # --- Processar TESS (TOI) ---
    toi_label_map = {
        'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'PC': 'CANDIDATE',
        'FP': 'FALSE POSITIVE', 'FA': 'FALSE POSITIVE'
    }
    toi_rename_map = {
        'tfopwg_disp': 'disposition', 'pl_orbper': 'orbital_period',
        'pl_trandurh': 'transit_duration', 'pl_trandep': 'transit_depth',
        'pl_rade': 'planet_radius', 'st_rad': 'stellar_radius'
    }
    toi_processed = toi_df.rename(columns=toi_rename_map)
    toi_processed['disposition'] = toi_processed['disposition'].map(toi_label_map)
    toi_processed['mission'] = 'TESS'
    # O TOI já vem com transit_depth em um formato decimal, então não precisa de conversão.
    toi_final = toi_processed.reindex(columns=standard_columns)

    # --- Combinar ---
    combined_df = pd.concat([koi_final, toi_final], ignore_index=True)
    combined_df.dropna(inplace=True)
    
    print(f"Datasets combinados. Total de amostras prontas para treino: {len(combined_df)}")
    print("Colunas finais:", combined_df.columns.tolist())
    
    return combined_df

# --- 3. TREINAMENTO DO MODELO ---

def train_model(df):
    """Treina o modelo de RandomForest e o salva."""
    print("\nIniciando treinamento do modelo...")
    
    # Filtrar classes para um problema de classificação mais claro (CONFIRMED vs FALSE POSITIVE)
    # A classe 'CANDIDATE' não é um rótulo final, então é melhor excluí-la do treino inicial.
    df_filtered = df[df['disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    print(f"Treinando com {len(df_filtered)} amostras (CONFIRMED e FALSE POSITIVE).")

    # Definir features (X) e alvo (y)
    target = 'disposition'
    X = df_filtered.drop(columns=[target])
    y = df_filtered[target]

    # Converter a feature categórica 'mission' para numérica (One-Hot Encoding)
    X = pd.get_dummies(X, columns=['mission'], drop_first=True)

    # Dividir dados para treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Tamanho do set de treino: {len(X_train)}, Tamanho do set de teste: {len(X_test)}")

    # Treinar o classificador
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred))

    # Salvar o modelo
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = "models/harmonized_rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Modelo treinado e salvo com sucesso em: {model_path}")

    return model

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    koi_dataframe = load_koi_data()
    toi_dataframe = fetch_toi_data()
    
    final_dataframe = harmonize_and_combine(koi_dataframe, toi_dataframe)
    
    if not final_dataframe.empty:
        train_model(final_dataframe)