# Base SAIC: Modelagem em Grafos de Coocorrência e Detecção de Comunidades (BraSNAM 2027)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brasnam2027-saic/base-saic/blob/main/notebooks/saic_brasnam.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Este repositório contém os dados curados, o código-fonte modular, os *logs* de execução e as tabelas/figuras para reprodução integral dos experimentos apresentados no artigo:

> **"Modelagem em Grafos de Coocorrência e Detecção de Comunidades na Base Histórica da Assembleia Nacional Constituinte de 1988: Da Curadoria dos Dados à Descoberta de Eixos Temáticos"**  
> Submissão ao **BraSNAM 2027** (*Brazilian Workshop on Social Network Analysis and Mining*), evento integrante do Congresso da Sociedade Brasileira de Computação (CSBC).  
> *Modo de revisão duplamente cega (Double-Blind Review).*

---

## 1. Estrutura do Repositório

```text
base-saic/
├── README.md                          # Documentação completa e instruções de reprodução
├── requirements.txt                   # Dependências do projeto com versões fixas
├── SAIC.csv                           # Base de dados curada e desambiguada (72.719 linhas x 21 colunas)
├── sbc-template.sty                   # Folha de estilo SBC para compilação LaTeX
├── artigo_brasnam_saic.tex            # Fonte LaTeX do artigo no padrão SBC
├── brasnam_saic_overleaf.zip          # Pacote pronto para compilação no Overleaf
├── data/
│   └── README.md                      # Instruções de dados, metadados e reprodução
├── src/
│   ├── curate.py                      # Auditoria da base, deduplicação e lematização spaCy
│   ├── build_network.py               # Construção da rede PPMI, baseline ótimo e sensibilidade
│   ├── louvain_stability.py           # Análise de estabilidade em 50 seeds (Q, K, NMI, ARI)
│   ├── contingency.py                 # Matriz de contingência (Regra A vs. Regra B e robustez)
│   └── figures.py                     # Visualização do grafo e tabela de centralidades
├── notebooks/
│   └── saic_brasnam.ipynb             # Notebook Colab-ready reproduzindo o pipeline de ponta a ponta
├── results/
│   ├── tables/
│   │   ├── tabela_sensibilidade.csv/.tex       # Grade completa de min_df vs. percentil PPMI
│   │   ├── tabela_estabilidade_louvain.csv/.tex# Estatísticas de estabilidade em 50 seeds
│   │   ├── tabela_comunidades.csv/.tex         # Síntese das 8 comunidades e líderes de centralidade
│   │   ├── tabela_contingencia_regraA.csv      # Matriz de contingência (Regra A)
│   │   ├── tabela_contingencia_regraB.csv      # Matriz de contingência (Regra B)
│   │   └── tabela_contingencia_comparacao.csv  # Métricas de correlação e erro MAE
│   └── figures/
│       ├── grafo_comunidades_brasnam.png/.pdf  # Rede semântica em 300 dpi e PDF vetorial
│       ├── heatmap_contingencia_brasnam.png/.pdf# Heatmap daltonismo-safe (viridis)
│       └── fig_estabilidade_louvain.png/.pdf   # Boxplots de Q, NMI e distribuição de K
└── logs/
    ├── baseline_verification.log      # Log de reprodução do baseline ótimo (|V|=322, |E|=1823, Q=0.6363)
    ├── louvain_stability.log          # Log de execução das 50 seeds de Louvain
    ├── contingency_comparison.log     # Log de contraste Regra A vs. Regra B
    └── curate_audit.log               # Log de auditoria dos 72.719 registros e 21 colunas
```

---

## 2. Como Reproduzir os Resultados

### Opção A: Execução Local via Terminal

> **Privacidade dos Dados e Cache de Lematização:** Em conformidade com as diretrizes de privacidade individual e para manter o repositório leve e ágil (< 10 MB sem os dados brutos), o arquivo binário `data/lemmatized_docs.pkl` **não é versionado no repositório** (ignorado via `.gitignore`). Na primeira execução, o script `python src/curate.py` processa as 70.150 cartas únicas a partir de `SAIC.csv` em cerca de 10 a 12 minutos em CPU convencional e persiste o cache localmente em `data/` para reaproveitamento instantâneo em todas as etapas subsequentes.

1. **Clone o repositório:**
   ```bash
   git clone [URL_DO_REPOSITORIO]
   cd base-saic
   ```

2. **Crie e ative um ambiente virtual (Python 3.10+):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # No Linux/macOS
   .\venv\Scripts\Activate.ps1    # No Windows PowerShell
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute os módulos sequencialmente:**
   ```bash
   # 1. Auditoria e verificação dos dados brutos:
   python src/curate.py

   # 2. Construção da rede PPMI e verificação do baseline:
   python src/build_network.py

   # 3. Análise de estabilidade do Louvain em 50 seeds:
   python src/louvain_stability.py

   # 4. Matriz de contingência (Regra A vs. Regra B) e heatmap:
   python src/contingency.py

   # 5. Geração da visualização da rede e tabela de comunidades:
   python src/figures.py
   ```
   *Tempo total de execução local:* ~2 minutos.

### Opção B: Execução via Google Colab

Basta abrir o arquivo [`notebooks/saic_brasnam.ipynb`](notebooks/saic_brasnam.ipynb) e clicar no botão **"Open in Colab"** no topo da página. Caso o cache local de lematização não esteja presente no ambiente, a célula de inicialização executa automaticamente a rotina de curadoria e lematização a partir de `SAIC.csv`.

---

## 3. Resumo dos Experimentos e Achados Principais

| Experimento | Parâmetros Principais | Métricas / Resultados Obtidos |
|---|---|---|
| **P1: Baseline Ótimo** | $\text{min\_df} = 2{,}0\%$, Percentil 95 ($\theta_{95} = 1{,}25$) | $\|V\| = 322$, $\|E\| = 1.823$, densidade $3{,}53\%$, 1 comp. conexa, $Q = 0{,}6363$, $K = 8$. |
| **P2: Estabilidade Louvain** | 50 *seeds* independentes ($0$ a $49$) | $Q = 0{,}6361 \pm 0{,}0034$ (CV = $0{,}54\%$), $K = 8$ (62\%) e $K = 7$ (38\%); NMI (todas as 50) = $0{,}8047 \pm 0{,}0415$; NMI (excl. seed 42, $N=49$) = $0{,}8007 \pm 0{,}0311$; ARI (excl. seed 42, $N=49$) = $0{,}7079 \pm 0{,}0484$; fusões em $K=7$: C1+C2 em 78,9% (15/19), C2+C6 em 10,5% (2/19), C3+C8 em 5,3% (1/19), C1+C5 em 5,3% (1/19). |
| **P3: Matriz de Contingência** | Regra A (normalizada por $\|C_k\|$) vs. Regra B (*winner-takes-all*) | Concordância total nos polos dominantes (*top-1*); MAE = $8{,}55$ p.p.; evidencia omissão de Segurança Pública em 1986. |
| **P4: Centralidades** | Subgrafos induzidos não ponderados | Identificação de nós centrais: C1 (*agricultura*), C2 (*união*), C3 (*mandato*), C4 (*ensino*), C5 (*prisão/aborto/violência*), C6 (*previdência*), C7 (*detetive*), C8 (*Deus*). |

---

## 4. Dicionário de Dados (`SAIC.csv`)

A base curada contém 72.719 linhas e 21 colunas:

| # | Coluna | Tipo | Descrição |
|---|---|---|---|
| 1 | `ID PESSOA` | Inteiro | Identificador único desambiguado do remetente (64.869 cidadãos únicos). |
| 2 | `NOME` | Texto | Nome do cidadão remetente (padronizado e tratado por *Fuzzy Matching*). |
| 3 | `DESTINATÁRIO` | Texto | Parlamentar destinatário (2.190 parlamentares desambiguados). |
| 4 | `CATÁLOGO` | Texto | Classificação temática oficial Prodasen (24 macrotemas). |
| 5 | `INDEXAÇÃO` | Texto | Subtemas específicos oficiais (1.308 subtemas únicos). |
| 6 | `TEXTO SUGESTÃO` | Texto | Transcrição integral da carta do cidadão (70.149 textos únicos). |
| 7 | `ORIGEM` | Texto | Código de lote/origem do formulário no Prodasen. |
| 8 | `DATA` | Data | Data de registro inicial do lote. |
| 9 | `FORMUL` | Inteiro | Número sequencial do formulário. |
| 10 | `DV` | Inteiro | Dígito verificador. |
| 11 | `DATA2` | Data | Data secundária de processamento no sistema SAIC. |
| 12 | `MUNICÍPIO` | Texto | Município de residência (3.542 municípios padronizados). |
| 13 | `UF` | Texto | Unidade da Federação (26 UFs históricas vigentes em 1986–1987). |
| 14 | `CEP` | Inteiro | Código postal de 5 dígitos da época. |
| 15 | `SEXO` | Texto | Sexo autodeclarado (`Masculino`, `Feminino` ou `N/A`). |
| 16 | `MORADOR` | Texto | Situação de moradia (`Zona urbana`, `Zona rural` ou `N/A`). |
| 17 | `INSTRUÇÃO` | Texto | Grau de escolaridade autodeclarado (8 categorias). |
| 18 | `ESTADO CIVIL` | Texto | Estado civil autodeclarado (5 categorias). |
| 19 | `FAIXA ETÁRIA` | Texto | Faixa de idade autodeclarada (8 faixas). |
| 20 | `FAIXA RENDA` | Texto | Faixa de renda em salários mínimos da época (8 faixas). |
| 21 | `ATIVIDADE` | Texto | Setor econômico ou profissão (10 categorias). |

---

## 5. Diretrizes Éticas e Privacidade dos Dados

- **Privacidade individual:** Conforme exigido para acervos históricos públicos e pela Lei Geral de Proteção de Dados (LGPD), não são divulgados nem explorados dados cadastrais sensíveis individualizados. Os nomes dos remetentes foram utilizados exclusivamente na etapa algorítmica interna de desambiguação e deduplicação de remetentes.
- **Isenção de segmentação discriminatória:** O escopo do artigo e deste repositório restringe-se estritamente à análise da rede semântica de termos e ao contraste com a taxonomia do Prodasen, sem qualquer segmentação ou perfilamento sociodemográfico discriminatório.

---

## 6. Licença e Citação

- **Dados:** Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Código-Fonte:** MIT License.

```bibtex
@inproceedings{brasnam2027_saic,
  author    = {Autores omitidos para revisão duplamente cega},
  title     = {Modelagem em Grafos de Coocorrência e Detecção de Comunidades na Base Histórica da Assembleia Nacional Constituinte de 1988: Da Curadoria dos Dados à Descoberta de Eixos Temáticos},
  booktitle = {Anais do XVI Brazilian Workshop on Social Network Analysis and Mining (BraSNAM 2027)},
  year      = {2027},
  pages     = {1--12},
  publisher = {SBC}
}
```
