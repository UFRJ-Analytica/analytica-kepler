# Em: src/train.py (Versão 10.1 - Corrigido o UnboundLocalError)

import pandas as pd
import joblib
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Importa a nossa função de processamento centralizada
from processing import preprocess_and_engineer_features

def fetch_nasa_archive_data(table_name: str) -> pd.DataFrame:
    """Busca dados de uma tabela do NASA Exoplanet Archive."""
    print(f"--- TRAIN v10.1: Buscando dados da tabela '{table_name}'... ---")
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    query = f"select+*+from+{table_name}"
    full_query = f"{base_url}{query}&format=csv"
    try:
        response = requests.get(full_query)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data, comment='#')
        print(f"--- TRAIN v10.1: Sucesso! {len(df)} registos baixados de '{table_name}'. ---")
        return df
    except Exception as e:
        print(f"--- ERRO CRÍTICO TRAIN v10.1: Falha ao buscar dados da API. Erro: {e} ---")
        return None

def run_combined_training(custom_processed_df: pd.DataFrame = None, output_filename: str = "robust_generalist_model.pkl"):
    """
    O novo motor de treino. Busca os dados padrão (KOI, TOI), opcionalmente os combina
    com um dataset customizado já processado, e treina um novo modelo.
    """
    print("\n--- TRAIN v10.1: Iniciando Pipeline de Treinamento Combinado ---")
    
    koi_df_raw = fetch_nasa_archive_data(table_name='cumulative')
    toi_df_raw = fetch_nasa_archive_data(table_name='toi')

    if koi_df_raw is None or toi_df_raw is None:
        raise RuntimeError("Falha na obtenção dos dados padrão (KOI/TOI). Abortando.")

    koi_rename_map = {
        'koi_disposition': 'disposition', 'koi_period': 'orbital_period', 'koi_duration': 'transit_duration', 
        'koi_depth': 'transit_depth', 'koi_prad': 'planet_radius', 'koi_srad': 'stellar_radius', 
        'koi_impact': 'impact_parameter', 'koi_slogg': 'stellar_gravity'
    }
    toi_rename_map = {
        'tfopwg_disp': 'disposition', 'pl_orbper': 'orbital_period', 'pl_trandurh': 'transit_duration', 
        'pl_trandep': 'transit_depth', 'pl_rade': 'planet_radius', 'st_rad': 'stellar_radius', 
        'pl_imppar': 'impact_parameter', 'st_logg': 'stellar_gravity'
    }
    
    koi_df_processed = preprocess_and_engineer_features(koi_df_raw.rename(columns=koi_rename_map))
    toi_df_processed = preprocess_and_engineer_features(toi_df_raw.rename(columns=toi_rename_map))
    
    datasets_to_combine = [koi_df_processed, toi_df_processed]
    if custom_processed_df is not None and not custom_processed_df.empty:
        print(f"--- TRAIN v10.1: Adicionando {len(custom_processed_df)} amostras do dataset customizado. ---")
        datasets_to_combine.append(custom_processed_df)

    full_dataset = pd.concat(datasets_to_combine, ignore_index=True, sort=False)
    
    final_features = [
        'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius', 
        'stellar_radius', 'impact_parameter', 'stellar_density'
    ]
    
    # --- A CORREÇÃO CRÍTICA ESTÁ AQUI ---
    # Usamos 'full_dataset' para a filtragem inicial.
    final_dataset = full_dataset[full_dataset['disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    final_dataset.dropna(subset=['disposition'], inplace=True)
    
    print(f"--- TRAIN v10.1: Dataset combinado final pronto com {len(final_dataset)} amostras válidas. ---")

    X = final_dataset[final_features]
    y = final_dataset['disposition']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=final_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=2)
    model.fit(X_train, y_train.values.ravel())
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- TRAIN v10.1: Acurácia do Modelo Combinado: {accuracy:.2%} ---")
    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred))

    model_artifact = {
        'model': model, 'imputer': imputer, 
        'features': final_features, 'accuracy': accuracy
    }
    
    if not os.path.exists('models'): os.makedirs('models')
    model_path = os.path.join("models", output_filename)
    joblib.dump(model_artifact, model_path)
    print(f"\n✅ TRAIN v10.1: Artefacto do modelo combinado salvo com sucesso em: {model_path}")
    
    return model_path, accuracy

def main():
    """Função de conveniência para o treino padrão (sem dados custom)."""
    run_combined_training(output_filename="robust_generalist_model.pkl")

if __name__ == '__main__':
    main()