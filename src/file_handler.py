# Em: src/file_handler.py (Versão 1.1 - Headers Insensitive)

import pandas as pd
import io

def read_exoplanet_data(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """
    Lê CSV de exoplanetas e extrai descrições de # COLUMN, com lowercase para robustez.
    """
    print("--- FILE_HANDLER v1.1: Iniciando leitura. ---")
    
    try:
        string_data = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        string_data = uploaded_file.getvalue().decode("latin-1")  # Fallback
        print("--- AVISO FILE_HANDLER: Usado latin-1 para decode. ---")
    
    # --- Etapa 1: Extrair Descrições ---
    column_descriptions = {}
    lines = string_data.split('\n')
    for line in lines:
        if line.strip().startswith("# COLUMN"):
            try:
                parts = line.replace("# COLUMN ", "").split(":", 1)
                col_name = parts[0].strip().lower()  # Lower para match insensitive
                col_desc = parts[1].strip()
                column_descriptions[col_name] = col_desc
            except IndexError:
                continue
    
    print(f"--- FILE_HANDLER v1.1: {len(column_descriptions)} descrições extraídas. ---")

    # --- Etapa 2: Ler DataFrame ---
    data = io.StringIO(string_data)
    try:
        df = pd.read_csv(data, comment='#')
        # Lower columns para consistência no map
        df.columns = df.columns.str.lower()
        print(f"--- FILE_HANDLER v1.1: DF lido: {len(df)} linhas, {len(df.columns)} cols. ---")
    except Exception as e:
        print(f"--- ERRO FILE_HANDLER: {e} ---")
        return pd.DataFrame(), {}

    return df, column_descriptions