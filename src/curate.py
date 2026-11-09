"""
src/curate.py
Módulo de auditoria, higienização e pré-processamento de linguagem natural da base SAIC.
Executa:
1. Auditoria da integridade dos campos da base SAIC tratada (72.719 registros, 21 colunas).
2. Deduplicação exata de textos idênticos da coluna 'TEXTO SUGESTÃO' (70.151 cartas únicas válidas).
3. Pipeline NLP via spaCy (pt_core_news_md): tokenização, remoção de stopwords/pontuação/números e lematização.
4. Exportação/carregamento do cache de documentos lematizados em data/lemmatized_docs.pkl.
"""

import os
import sys
import time
import pickle
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "SAIC.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_PATH = os.path.join(DATA_DIR, "lemmatized_docs.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "curate_audit.log")

def audit_dataset(df):
    """Gera resumo de integridade das 21 colunas da base SAIC."""
    summary_rows = []
    for col in df.columns:
        summary_rows.append({
            'Coluna': col,
            'Nulos (N/A)': int(df[col].isna().sum()),
            'Valores Únicos': int(df[col].nunique(dropna=True))
        })
    return pd.DataFrame(summary_rows)

def preprocess_corpus(df_dedup, use_spacy=False):
    """
    Processa os textos das cartas. Se o cache existir, carrega diretamente.
    Caso use_spacy=True e o cache não exista, executa o modelo spaCy.
    """
    if os.path.exists(CACHE_PATH):
        print(f"Carregando lematizações diretamente do cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        return cache["docs"], cache["catalogo"], cache["indices"]

    print("Cache não encontrado. Iniciando processamento com spaCy...")
    import spacy
    nlp = spacy.load("pt_core_news_md", disable=["ner", "parser"])
    
    texts = df_dedup['TEXTO SUGESTÃO'].astype(str).tolist()
    catalogo = df_dedup['CATÁLOGO'].tolist()
    indices = df_dedup.index.tolist()
    
    processed_docs = []
    t_start = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=1000)):
        lemmas = [
            token.lemma_.lower().strip()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha
        ]
        processed_docs.append(" ".join(lemmas))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(texts)} textos processados ({time.time() - t_start:.1f}s)...")
            
    cache_data = {
        "docs": processed_docs,
        "catalogo": catalogo,
        "indices": indices
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache salvo com sucesso em {CACHE_PATH}")
    return processed_docs, catalogo, indices

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Lendo base SAIC de {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"AVISO: Arquivo {CSV_PATH} não encontrado. Verifique instruções em data/README.md.")
        return

    df = pd.read_csv(CSV_PATH, sep=';', low_memory=False)
    n_rows, n_cols = df.shape
    print(f"Base carregada: {n_rows} linhas x {n_cols} colunas")

    df_audit = audit_dataset(df)
    
    # Deduplicação
    df_dedup = df.dropna(subset=['TEXTO SUGESTÃO']).drop_duplicates(subset=['TEXTO SUGESTÃO']).copy()
    n_dedup = len(df_dedup)
    n_unique_persons = df['ID PESSOA'].nunique() if 'ID PESSOA' in df.columns else None

    # Distribuição de macrotemas
    cat_counter = Counter()
    for val in df['CATÁLOGO'].dropna():
        items = [x.strip() for x in str(val).split(',') if x.strip()]
        cat_counter.update(items)

    log_content = f"""================================================================================
AUDITORIA E CURADORIA DA BASE DE DADOS SAIC
Data da auditoria: {time.strftime('%Y-%m-%d %H:%M:%S')}
Ambiente: Python {sys.version}
Tempo de execução: {time.time() - t0:.2f} segundos
================================================================================

ESTATÍSTICAS GERAIS:
- Total de registros brutos: {n_rows}
- Total de colunas: {n_cols}
- Remetentes únicos desambiguados (ID PESSOA): {n_unique_persons}
- Cartas preenchidas com sugestão: {int(df['TEXTO SUGESTÃO'].notna().sum())}
- Textos únicos válidos após deduplicação: {n_dedup}
- Macrotemas oficiais no Catálogo: {len(cat_counter)}

INTEGRIDADE DAS COLUNAS:
{df_audit.to_string(index=False)}

DISTRIBUIÇÃO DOS 24 MACROTEMAS DO CATÁLOGO:
{pd.DataFrame(cat_counter.most_common(), columns=['Macrotema', 'Frequência']).to_string(index=False)}
================================================================================
"""
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(log_content)
    print("\n" + log_content)

    # Validar processamento/cache
    docs, catalogo, indices = preprocess_corpus(df_dedup, use_spacy=False)
    print(f"Verificação concluída: {len(docs)} documentos lematizados disponíveis.")

if __name__ == "__main__":
    main()
