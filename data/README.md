# Diretório de Dados (`data/`)

Este diretório armazena os metadados e os caches intermediários gerados localmente:

1. **`lemmatized_docs.pkl` (gerado localmente via `src/curate.py`, não versionado no Git):**
   - Cache serializado contendo os 70.150 textos únicos da base SAIC lematizados pelo modelo `pt_core_news_md` da biblioteca `spaCy` (sem *stopwords*, caracteres não-alfabéticos ou pontuações).
   - Por razões de privacidade individual dos cidadãos remetentes e para manter o repositório leve (< 10 MB), esse arquivo binário é ignorado pelo Git (`.gitignore`).
   - É gerado automaticamente na primeira execução de:
     ```bash
     python src/curate.py
     ```
     ou pela célula correspondente do notebook no Colab (~10 a 12 minutos em CPU convencional). Uma vez gerado, todas as execuções posteriores executam em menos de 2 minutos.

2. **`SAIC.csv` (localizado na raiz do projeto):**
   - Base tabular histórica higienizada (72.719 registros x 21 atributos, desambiguada e normalizada).
