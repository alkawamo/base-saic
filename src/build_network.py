"""
src/build_network.py
Construção da rede de coocorrência de termos, cálculo do PPMI e análise de sensibilidade.
Reproduz e valida os parâmetros do baseline:
min_df = 2.0%, percentil 95 (theta_95 = 1.25) -> |V| = 322, |E| = 1823, densidade 3.53%, 1 comp conexa, Q = 0.6363, K = 8.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "lemmatized_docs.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "baseline_verification.log")
TABLE_CSV = os.path.join(BASE_DIR, "results", "tables", "tabela_sensibilidade.csv")
TABLE_TEX = os.path.join(BASE_DIR, "results", "tables", "tabela_sensibilidade.tex")

def compute_ppmi_matrix(docs, min_df):
    vec = CountVectorizer(binary=True, min_df=min_df)
    A = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    n_docs, n_terms = A.shape
    
    C = (A.T @ A).toarray().astype(float)
    diag = np.diag(C).copy()
    np.fill_diagonal(C, 0.0)
    
    p_ij = C / n_docs
    p_i = diag / n_docs
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(p_ij / np.outer(p_i, p_i))
    ppmi = np.maximum(0.0, np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0))
    return A, vocab, ppmi

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TABLE_CSV), exist_ok=True)

    print(f"Carregando documentos lematizados de {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        cache = pickle.load(f)
    docs = cache["docs"]
    n_docs = len(docs)
    print(f"Total de documentos carregados: {n_docs}")

    # Sensibilidade completa (min_df vs percentil)
    exp_rows = []
    print("\nExecutando grade de sensibilidade (min_df vs percentis de PPMI)...")
    for min_df in [0.01, 0.015, 0.02, 0.025, 0.03]:
        t_start = time.time()
        _, vocab, ppmi = compute_ppmi_matrix(docs, min_df)
        n_terms = len(vocab)
        iu = np.triu_indices(n_terms, k=1)
        pos_vals = ppmi[iu][ppmi[iu] > 0]
        
        for perc in [50, 75, 90, 95]:
            corte = np.percentile(pos_vals, perc)
            G_tmp = nx.Graph()
            mask = ppmi[iu] >= corte
            for u_idx, v_idx, w in zip(iu[0][mask], iu[1][mask], ppmi[iu][mask]):
                G_tmp.add_edge(vocab[u_idx], vocab[v_idx], weight=float(w))
            
            comms_tmp = nx.community.louvain_communities(G_tmp, weight='weight', seed=42)
            Q = nx.community.modularity(G_tmp, comms_tmp, weight='weight')
            dens = nx.density(G_tmp) * 100
            n_comp = nx.number_connected_components(G_tmp)
            
            exp_rows.append({
                'min_df': min_df,
                'min_df_str': f"{min_df*100:.1f}%",
                'percentil': perc,
                'corte_ppmi': round(float(corte), 2),
                'arestas': G_tmp.number_of_edges(),
                'nos': G_tmp.number_of_nodes(),
                'densidade': round(float(dens), 2),
                'comp_conexas': n_comp,
                'modularity_Q': round(float(Q), 4),
                'n_comunidades': len(comms_tmp)
            })
            print(f"min_df={min_df*100:.1f}%, p={perc:02d} -> |V|={G_tmp.number_of_nodes()}, |E|={G_tmp.number_of_edges()}, Q={Q:.4f}, K={len(comms_tmp)}, Comp={n_comp}")

    df_exp = pd.DataFrame(exp_rows)
    df_exp.to_csv(TABLE_CSV, index=False, encoding='utf-8')
    print(f"\nTabela salva em {TABLE_CSV}")

    # Gerar LaTeX (booktabs)
    tex_lines = [
        "\\begin{tabular}{ccccccccc}",
        "\\toprule",
        "\\textbf{\\texttt{min\\_df}} & \\textbf{Percentil} & \\textbf{Corte ($\\theta_p$)} & \\textbf{Arestas ($|E|$)} & \\textbf{Nós ($|V|$)} & \\textbf{Densidade} & \\textbf{Comp.\\ Conexas} & \\textbf{Mod.\\ ($Q$)} & \\textbf{Comunidades} \\\\",
        "\\midrule"
    ]
    for min_df_val in [0.01, 0.015, 0.02, 0.025, 0.03]:
        sub = df_exp[df_exp['min_df'] == min_df_val]
        first = True
        min_df_label = f"{min_df_val*100:.1f}\\%"
        if min_df_val == 0.02:
            min_df_label = f"\\textbf{{{min_df_label}}}"
        
        for _, r in sub.iterrows():
            p_val = int(r['percentil'])
            corte_str = f"{r['corte_ppmi']:.2f}".replace('.', ',')
            e_str = f"{int(r['arestas']):,}".replace(',', '.')
            v_str = f"{int(r['nos'])}"
            dens_str = f"{r['densidade']:.2f}\\%".replace('.', ',')
            comp_str = f"{int(r['comp_conexas'])}"
            q_str = f"{r['modularity_Q']:.4f}".replace('.', ',')
            k_str = f"{int(r['n_comunidades'])}"

            if min_df_val == 0.02 and p_val == 95:
                row_str = f" & \\textbf{{{p_val}}} & \\textbf{{{corte_str}}} & \\textbf{{{e_str}}} & \\textbf{{{v_str}}} & \\textbf{{{dens_str}}} & \\textbf{{{comp_str}}} & \\textbf{{{q_str}}} & \\textbf{{{k_str}}} \\\\"
            else:
                row_str = f" & {p_val} & {corte_str} & {e_str} & {v_str} & {dens_str} & {comp_str} & {q_str} & {k_str} \\\\"
            
            if first:
                row_str = f"\\multirow{{4}}{{*}}{{{min_df_label}}}" + row_str
                first = False
            tex_lines.append(row_str)
        if min_df_val != 0.03:
            tex_lines.append("\\midrule")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    with open(TABLE_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Tabela LaTeX salva em {TABLE_TEX}")

    # Verificar baseline ótimo
    opt_row = df_exp[(df_exp['min_df'] == 0.02) & (df_exp['percentil'] == 95)].iloc[0]
    baseline_match = (
        opt_row['nos'] == 322 and
        opt_row['arestas'] == 1823 and
        opt_row['comp_conexas'] == 1 and
        abs(opt_row['modularity_Q'] - 0.6363) < 0.0002 and
        opt_row['n_comunidades'] == 8
    )

    log_content = f"""================================================================================
VERIFICAÇÃO DE REPRODUTIBILIDADE DO BASELINE (P1)
Data da execução: {time.strftime('%Y-%m-%d %H:%M:%S')}
Ambiente: Python {sys.version}
Documentos analisados: {n_docs}
Tempo de execução: {time.time() - t0:.2f} segundos
================================================================================

PARÂMETROS DO GRAFO ÓTIMO:
- min_df: 2.0%
- Percentil PPMI: 95
- Limiar de corte (theta_95): {opt_row['corte_ppmi']}

MÉTRICAS OBTIDAS:
- Vértices (|V|): {opt_row['nos']} (Esperado: 322)
- Arestas (|E|): {opt_row['arestas']} (Esperado: 1823)
- Densidade: {opt_row['densidade']}% (Esperado: 3.53%)
- Componentes conexas: {opt_row['comp_conexas']} (Esperado: 1)
- Modularidade (Q): {opt_row['modularity_Q']:.4f} (Esperado: 0.6363)
- Comunidades (K): {opt_row['n_comunidades']} (Esperado: 8)

STATUS DO BASELINE: {'[OK] 100% REPRODUZIDO COM SUCESSO' if baseline_match else '[DIVERGÊNCIA IDENTIFICADA]'}
================================================================================
"""
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(log_content)
    print("\n" + log_content)

if __name__ == "__main__":
    main()
