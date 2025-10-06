# Em: src/processing.py (versão 9.4 - Units Padronizadas)

import pandas as pd
import numpy as np

def preprocess_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    O processador universal, agora com conversões pós-rename e fórmula corrigida para density.
    """
    print("--- PROCESSING v9.4: Iniciando pré-processamento resiliente. ---")
    
    processed_df = df.copy()

    # --- Etapa 1: Tratamento de Unidades (AGORA PÓS-RENAME: Usa nomes padronizados) ---
    # Transit Depth: Sempre /100 se >0.1 (assume % -> fraction)
    if 'transit_depth' in processed_df.columns:
        numeric_depth = pd.to_numeric(processed_df['transit_depth'], errors='coerce')
        mask_percent = numeric_depth > 0.1  # Threshold: >10% raw é improvável como fraction
        processed_df.loc[mask_percent, 'transit_depth'] = numeric_depth[mask_percent] / 100.0
        print("--- PROCESSING v9.4: 'transit_depth' convertido de % para fração onde aplicável. ---")
    
    # Disposition mapping (para TOI-like, mas genérico)
    if 'disposition' in processed_df.columns:
        target_map = {'CP': 'CONFIRMED', 'KP': 'CONFIRMED', 'FP': 'FALSE POSITIVE', 'CANDIDATE': 'CANDIDATE'}
        processed_df['disposition'] = processed_df['disposition'].map(target_map).fillna(processed_df['disposition'])

    # --- Etapa 2: Normalização de Transit Duration (Pós-rename, detecção melhorada) ---
    if 'transit_duration' in processed_df.columns:
        numeric_dur = pd.to_numeric(processed_df['transit_duration'], errors='coerce')
        # Detecção: Mediana <1 sugere days; >1 e <50 sugere hours. Use mediana para robustez.
        median_dur = numeric_dur.median()
        if 0.01 < median_dur < 1:  # Típico days (0.1-0.5)
            print("--- PROCESSING v9.4: 'transit_duration' em DIAS. Convertendo para HORAS. ---")
            processed_df['transit_duration'] = numeric_dur * 24.0
        elif median_dur > 50:  # Improvável hours (>2 days equiv)
            print("--- PROCESSING v9.4: 'transit_duration' suspeito (>50h). Convertendo de hours para days? Ajuste manual. ---")
            processed_df['transit_duration'] = numeric_dur / 24.0  # Fallback
        else:
            print("--- PROCESSING v9.4: 'transit_duration' em HORAS. OK. ---")

    # Filtro para transit candidates se flag disponível
    if 'tran_flag' in processed_df.columns:
        processed_df = processed_df[processed_df['tran_flag'] == 1]
        print(f"--- PROCESSING v9.4: Filtrado para {len(processed_df)} candidatos com trânsitos (tran_flag=1). ---")

    # --- Etapa 3: Engenharia de Features Robusta (Corrigida + st_dens Fallback) ---
    if 'stellar_gravity' in processed_df.columns and 'stellar_radius' in processed_df.columns:
        print("--- PROCESSING v9.4: Criando 'stellar_density'. ---")
        
        processed_df['stellar_density'] = np.nan
        
        # Fallback: Use st_dens se disponível (K2 tem!)
        if 'st_dens' in processed_df.columns:
            processed_df['stellar_density'] = pd.to_numeric(processed_df['st_dens'], errors='coerce')
            print("--- PROCESSING v9.4: Usando 'st_dens' direto para density. ---")
        else:
            # Compute approx: log10(ρ) ≈ logg - 3*log10(R)  (ρ ∝ g/R^3 para massa const, mas approx boa para main-seq)
            sg_num = pd.to_numeric(processed_df['stellar_gravity'], errors='coerce')
            sr_num = pd.to_numeric(processed_df['stellar_radius'], errors='coerce')
            valid_mask = sg_num.notna() & sr_num.notna() & (sr_num > 0)
            if valid_mask.any():
                processed_df.loc[valid_mask, 'stellar_density'] = \
                    sg_num[valid_mask] - 3 * np.log10(sr_num[valid_mask])
                print(f"--- PROCESSING v9.4: Density computada para {valid_mask.sum()} linhas. ---")
            else:
                print("--- AVISO PROCESSING v9.4: Sem dados válidos para compute density. ---")

    print("--- PROCESSING v9.4: Concluído. ---")
    return processed_df