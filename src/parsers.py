# Em: src/parsers.py (versão final com seleção de colunas)

import pandas as pd

def parse_koi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduz um DataFrame no formato KOI, seleciona as colunas corretas 
    e adiciona a feature de missão.
    """
    print("Usando o parser do KOI...")
    
    # Colunas que nosso modelo foi treinado para usar
    model_features = [
        'orbital_period', 'transit_duration', 'transit_depth', 
        'planet_radius', 'stellar_radius', 'mission_TESS'
    ]

    # Mapeamento de nomes de colunas
    koi_rename_map = {
        'koi_period': 'orbital_period',
        'koi_duration': 'transit_duration',
        'koi_depth': 'transit_depth',
        'koi_prad': 'planet_radius',
        'koi_srad': 'stellar_radius'
    }
    parsed_df = df.rename(columns=koi_rename_map)
    
    # Conversão de Unidade
    if 'transit_depth' in parsed_df.columns:
        parsed_df['transit_depth'] = parsed_df['transit_depth'] / 10000.0
    
    # Adiciona a feature de missão (One-Hot Encoded)
    # Para dados do Kepler, mission_TESS é sempre 0
    parsed_df['mission_TESS'] = 0
    
    # PASSO CRUCIAL: Retorna um novo DataFrame contendo APENAS as colunas 
    # que o modelo espera, na ordem correta.
    # Usamos .reindex() para garantir que todas as colunas estejam presentes, 
    # preenchendo com Nulo se alguma estiver faltando (embora não deva acontecer aqui).
    return parsed_df.reindex(columns=model_features)


def parse_toi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduz um DataFrame no formato TOI, seleciona as colunas corretas 
    e adiciona a feature de missão.
    """
    print("Usando o parser do TOI...")
    
    model_features = [
        'orbital_period', 'transit_duration', 'transit_depth', 
        'planet_radius', 'stellar_radius', 'mission_TESS'
    ]
    
    toi_rename_map = {
        'pl_orbper': 'orbital_period',
        'pl_trandurh': 'transit_duration',
        'pl_trandep': 'transit_depth',
        'pl_rade': 'planet_radius',
        'st_rad': 'stellar_radius'
    }
    parsed_df = df.rename(columns=toi_rename_map)
    
    # Adiciona a feature de missão (One-Hot Encoded)
    # Para dados do TESS, mission_TESS é sempre 1
    parsed_df['mission_TESS'] = 1
    
    return parsed_df.reindex(columns=model_features)

# A função parse_user_input não é mais necessária, pois a lógica de UI
# já cria o DataFrame no formato correto. Podemos removê-la para simplificar.