# Em: src/parsers.py (versão 3.0 - Com Engenharia de Features)

import pandas as pd
import numpy as np # Usaremos para cálculos matemáticos

def parse_koi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduz features, mantém a coluna de resultado e CRIA NOVAS FEATURES de um DataFrame KOI.
    """
    print("Usando o parser do KOI e criando novas features...")
    
    koi_rename_map = {
        'koi_period': 'orbital_period',
        'koi_duration': 'transit_duration',
        'koi_depth': 'transit_depth',
        'koi_prad': 'planet_radius',
        'koi_srad': 'stellar_radius',
        'koi_impact': 'impact_parameter' # <-- Nova feature mapeada
    }
    parsed_df = df.rename(columns=koi_rename_map)
    
    # --- ENGENHARIA DE FEATURES ---
    # 1. Densidade Estelar (logarítmica, aproximada)
    # Usamos np.log10 para normalizar a escala. Adicionamos 1 para evitar log de zero.
    if 'koi_slogg' in parsed_df.columns and 'stellar_radius' in parsed_df.columns:
        # Usamos .loc para evitar SettingWithCopyWarning
        parsed_df.loc[:, 'stellar_density'] = parsed_df['koi_slogg'] - np.log10(parsed_df['stellar_radius'] + 1)

    # Conversão de Unidade
    if 'transit_depth' in parsed_df.columns:
        parsed_df.loc[:, 'transit_depth'] = parsed_df['transit_depth'] / 10000.0
    
    parsed_df['mission_TESS'] = 0
    return parsed_df


def parse_toi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduz features, mantém a coluna de resultado e CRIA NOVAS FEATURES de um DataFrame TOI.
    """
    print("Usando o parser do TOI e criando novas features...")
    
    toi_rename_map = {
        'pl_orbper': 'orbital_period',
        'pl_trandurh': 'transit_duration',
        'pl_trandep': 'transit_depth',
        'pl_rade': 'planet_radius',
        'st_rad': 'stellar_radius',
        'pl_imppar': 'impact_parameter' # <-- Nova feature mapeada
    }
    parsed_df = df.rename(columns=toi_rename_map)

    # --- ENGENHARIA DE FEATURES ---
    # 1. Densidade Estelar (logarítmica, aproximada)
    if 'st_logg' in parsed_df.columns and 'stellar_radius' in parsed_df.columns:
        parsed_df.loc[:, 'stellar_density'] = parsed_df['st_logg'] - np.log10(parsed_df['stellar_radius'] + 1)
        
    parsed_df['mission_TESS'] = 1
    return parsed_df