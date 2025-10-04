# Analytica Kepler — Caçando Exoplanetas com IA

**Time:** Analytica Kepler (UFRJ Analytica)  
**Desafio:** NASA Space Apps 2025 — *Um mundo distante: caçando exoplanetas com IA*  
**Stack:** Python • Jupyter • (futuro) Streamlit

> Nosso objetivo é treinar modelos de IA/ML com dados abertos (Kepler/K2/TESS) e analisar **novos dados** (incluindo NEOSSat) para **priorizar/identificar** exoplanetas, com uma interface simples para pesquisadores e curiosos.

---

## 🚀 O que este repositório entrega

- **Notebook starter comentado (PT‑BR)** com pipeline de ponta a ponta:
  - Download de **FITS do NEOSSat** e inspeção rápida
  - Curva de luz *toy* (aperture sum) para validar o fluxo
  - Ingestão de **KOI/TOI** via **NASA Exoplanet Archive (TAP)**
  - **Baseline ML** com `RandomForest` para classificar `koi_disposition`
  - **Feature `mission`** (Kepler/TESS; pronto para NEOSSat como domínio novo)
- Estrutura de pastas preparada (com `.gitkeep`) para dados, modelos e saídas
- Checklist de próximos passos (fotometria, features de trânsito, UI Streamlit, deploy)

> **Classificação que fazemos aqui**: dado um conjunto de **atributos/tabulares** (ex.: período, duração, profundidade do trânsito, raio planetário/estelar…), o modelo aprende a prever a **disposição** do objeto — `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`. Isso **triagem**/prioriza os melhores alvos e reduz esforço manual, alinhado ao que o desafio pede.

---

## 📁 Estrutura

```
.
├─ data/               # dados locais (não versionados)  ← .gitignore
├─ datasets/           # exemplos pequenos para teste     ← .gitignore
├─ models/             # artefatos .pkl                   ← .gitignore
├─ outputs/            # relatórios/figuras               ← .gitignore
├─ figures/            # imagens geradas                  ← .gitignore
├─ logs/               # logs                             ← .gitignore
├─ notebooks/
│  └─ AnalyticaKepler_Exoplanets_Starter_UPDATED.ipynb
└─ README.md
```

> Observação: grandes arquivos (FITS, HDF, CSVs comprimidos, modelos) estão no `.gitignore`. Mantenha somente amostras pequenas em `datasets/`.

---

## ⚙️ Instalação rápida

```bash
# Recomendado: criar venv
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# Dependências essenciais
pip install numpy pandas matplotlib requests beautifulsoup4 astropy scikit-learn jupyter

# (Opcional para fotometria/curvas): photutils, lightkurve, pyvo
# pip install photutils lightkurve pyvo
```

---

## 🧪 Notebook principal

1. Abra o Jupyter:
   ```bash
   jupyter notebook
   ```
2. Execute o notebook:
   - `notebooks/AnalyticaKepler_Exoplanets_Starter_UPDATED.ipynb`

### Conteúdo do notebook

1) **NEOSSat (CSA)**  
   - Lista e baixa alguns **FITS** (amostra) do repositório público.  
   - Inspeção do cabeçalho e renderização da imagem para verificação.

2) **Curva de luz *toy***  
   - *Aperture sum* ao redor do pixel mais brilhante (com subtração de fundo).  
   - Ilustra o fluxo **FITS → métrica de fluxo** (não é fotometria científica).

3) **KOI/TOI via TAP** (NASA Exoplanet Archive)  
   - Carrega tabelas **KOI** (Kepler) e **TOI** (TESS) por HTTP (CSV).  
   - **Dica TOI:** use `AS` para nomes “amigáveis” e evitar erros 400 (colunas válidas):
     ```sql
     select top 500
       toi,
       tid as tic_id,
       tfopwg_disp,
       pl_orbper  as orbital_period,
       pl_trandep as transit_depth,
       pl_trandurh as transit_duration,
       pl_rade    as planet_radius,
       st_rad     as stellar_radius,
       st_teff    as stellar_teff
     from toi
     ```

4) **Baseline ML**  
   - Treina `RandomForest` para **`koi_disposition`** com features:  
     `koi_period, koi_duration, koi_depth, koi_prad, koi_impact, koi_srad, koi_smass`  
   - **Inclui `mission`** (`Kepler`, `TESS`) via *one‑hot*.  
   - Mostra **acurácia**, **classification_report** e **importância de features**.

5) **Exporta modelo**  
   - Salva em `models/koi_rf_with_mission.pkl`.

---

## 🧩 Como isso atende ao desafio

- **Treino supervisionado** usando rótulos oficiais (KOI/TOI);
- **Generalização entre missões** (feature `mission` e, futuramente, *domain adaptation*);
- **Análise de novos dados**: pipeline para gerar *features* de **NEOSSat** (FITS → LC/estatísticas → predição);
- **Interface web** (próximo passo): Streamlit para upload de dados, ajuste de hiperparâmetros e visualização de métricas.

> **Rotulagem TOI (harmonização)**: mapeie `tfopwg_disp` para as classes do Kepler, p.ex. `CP/KP → CONFIRMED`, `PC → CANDIDATE`, `FP/FA → FALSE POSITIVE`, `APC → AMBIG` (tratar à parte).

---

## 🧱 Próximos passos (backlog)

- **Fotometria**: `photutils` (aperture/PSF), alinhamento entre frames, *flags* de qualidade.
- **Features de trânsito**: profundidade/duração refinadas, odd/even, SNR, ajuste trapezoidal.
- **Validação**: PR‑AUC por classe, curvas de calibração, *cross‑mission split* (treina Kepler → testa TESS).
- **Domain adaptation**: `mission` + metadados de cabeçalho (NEOSSat) como *features* de robustez.
- **Streamlit (Railway)**: upload de CSV/LC, batch scoring, gráficos (LC dobrada, importância, PR‑curves).

---

## 🔗 Fontes de dados e documentação

- **NASA Exoplanet Archive (TAP)**  
  - Documentação TAP: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html  
  - Colunas **KOI (cumulative)**: https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html  
  - Colunas **TOI**: https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html  

- **NEOSSat — Agência Espacial Canadense (CSA)**  
  - Dataset aberto (FITS, árvore por ano): https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30  
  - Página NEOSSat: https://www.asc-csa.gc.ca/eng/satellites/neossat/  
  - CADC (NEOSSat): https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/neossat/  
  - Guia FITS NEOSSat (PDF): (ver pasta de documentos no portal Open Data)

- **JWST (contexto e follow‑up científico)**  
  - Informações: https://www.asc-csa.gc.ca/eng/satellites/jwst/about.asp

> **Observação de licença**: dados do NEOSSat seguem **Open Government Licence – Canada (OGL‑Canada)**. Cite as fontes (CSA/CADC, NASA Exoplanet Archive) ao publicar resultados/derivados.

---

## ▶️ Exemplo mínimo (TOI com alias)

```python
from urllib.parse import quote_plus
import pandas as pd

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
q = "
select top 500
  toi,
  tid as tic_id,
  tfopwg_disp,
  pl_orbper  as orbital_period,
  pl_trandep as transit_depth,
  pl_trandurh as transit_duration,
  pl_rade    as planet_radius,
  st_rad     as stellar_radius,
  st_teff    as stellar_teff
from toi
"
url = f"{TAP_BASE}?query={quote_plus(' '.join(q.split()))}&format=csv"
toi_df = pd.read_csv(url)
toi_df.head()
```

---

## 🤝 Contribuição

- Issues e PRs são bem-vindos.  
- Use notebooks claros, células curtas e comentários em PT‑BR/EN.  
- Evite subir arquivos grandes (FITS, modelos) — utilize *releases* ou *data links*.

---

## 📜 Agradecimentos

- **NASA Space Apps**, **NASA Exoplanet Archive**  
- **Agência Espacial Canadense (CSA)** pelo **NEOSSat** e dados abertos  
- Comunidade open‑source (Astropy, scikit‑learn, etc.)

---

## 📣 Contato

- **Equipe:** Analytica Kepler — UFRJ Analytica  
- **Objetivo:** acelerar a descoberta e priorização de exoplanetas com ferramentas abertas e acessíveis.
