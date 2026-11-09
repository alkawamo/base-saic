"""
src/figures.py
Geração da visualização da rede de coocorrência (322 nós, 1.823 arestas, 8 comunidades)
e tabela de centralidades das 8 comunidades.
Exporta:
- results/figures/grafo_comunidades_brasnam.png (300 dpi)
- results/figures/grafo_comunidades_brasnam.pdf (vetorial)
- grafo_comunidades_brasnam.png (na raiz para compilação LaTeX direta)
- results/tables/tabela_comunidades.csv
- results/tables/tabela_comunidades.tex
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "lemmatized_docs.pkl")
FIG_PNG = os.path.join(BASE_DIR, "results", "figures", "grafo_comunidades_brasnam.png")
FIG_PDF = os.path.join(BASE_DIR, "results", "figures", "grafo_comunidades_brasnam.pdf")
ROOT_PNG = os.path.join(BASE_DIR, "grafo_comunidades_brasnam.png")
TABLE_CSV = os.path.join(BASE_DIR, "results", "tables", "tabela_comunidades.csv")
TABLE_TEX = os.path.join(BASE_DIR, "results", "tables", "tabela_comunidades.tex")

COMMUNITY_METADATA = {
    'C1': {'name': 'C1: Desenv. Econômico e Setores Produtivos', 'color': '#2ca02c', 'scope': 'agrícola, terra, reforma, rural, produzir, preço, inflação, banco, indústria. Reforma agrária e macroeconomia.'},
    'C2': {'name': 'C2: Adm. Pública e Org. do Estado', 'color': '#1f77b4', 'scope': 'estado, município, órgão, ministério, cargo, concurso, funcionário, artigo, assegurar. Estrutura burocrática e funcionalismo.'},
    'C3': {'name': 'C3: Estrutura Política do Estado', 'color': '#ff7f0e', 'scope': 'deputado, senador, prefeito, vereador, eleição, voto, partido, democracia. Representação parlamentar e eleições.'},
    'C4': {'name': 'C4: Política Educacional', 'color': '#9467bd', 'scope': 'educação, escola, universidade, faculdade, curso, aluno, gratuito, religioso. Acesso ao ensino público e religioso.'},
    'C5': {'name': 'C5: Sistema Penal e Segurança', 'color': '#d62728', 'scope': 'crime, criminoso, polícia, policial, militar, pena, morte, punição, família, criança. Segurança pública, pena de morte e família.'},
    'C6': {'name': 'C6: Seguridade Social', 'color': '#8c564b', 'scope': 'inps, aposentadoria, aposentado, médico, hospital, assistência, salário, mínimo, alimentação. Previdência, saúde e renda.'},
    'C7': {'name': 'C7: Autonomia e Reg. Profissional', 'color': '#e377c2', 'scope': 'autonomia, autônomo, profissional, liberal, livre, exercer, detetive, particular. Profissões liberais e detetives particulares.'},
    'C8': {'name': 'C8: Apelo Popular e Manifestações', 'color': '#7f7f7f', 'scope': 'fome, desemprego, pobre, dinheiro, comprar, sofrer, pedir, esperar, Sarney. Carestia, desabafos e críticas ao governo.'}
}

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(FIG_PNG), exist_ok=True)
    os.makedirs(os.path.dirname(TABLE_CSV), exist_ok=True)

    print("Carregando base pré-processada...")
    with open(DATA_PATH, "rb") as f:
        cache = pickle.load(f)
    docs = cache["docs"]

    print("Construindo rede PPMI ótima (min_df=2.0%, percentil 95)...")
    vec = CountVectorizer(binary=True, min_df=0.02)
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

    iu = np.triu_indices(n_terms, k=1)
    pos_vals = ppmi[iu][ppmi[iu] > 0]
    corte_95 = np.percentile(pos_vals, 95)

    G = nx.Graph()
    mask = ppmi[iu] >= corte_95
    for u_idx, v_idx, w in zip(iu[0][mask], iu[1][mask], ppmi[iu][mask]):
        G.add_edge(vocab[u_idx], vocab[v_idx], weight=float(w))

    comms = nx.community.louvain_communities(G, weight='weight', seed=42)

    comm_map = {}
    for c in comms:
        c_lower = {w.lower() for w in c}
        if 'agricultura' in c_lower:
            comm_map['C1'] = c
        elif 'união' in c_lower:
            comm_map['C2'] = c
        elif 'mandato' in c_lower:
            comm_map['C3'] = c
        elif 'ensino' in c_lower:
            comm_map['C4'] = c
        elif 'prisão' in c_lower:
            comm_map['C5'] = c
        elif 'previdência' in c_lower:
            comm_map['C6'] = c
        elif 'investigação' in c_lower or 'detetive' in c_lower:
            comm_map['C7'] = c
        elif 'deus' in c_lower:
            comm_map['C8'] = c

    # 1. Gerar Tabela de Comunidades
    print("\nCalculando métricas de centralidade intra-comunidade...")
    table_rows = []
    tex_rows = []
    for k in range(1, 9):
        c_id = f"C{k}"
        nodes_c = comm_map[c_id]
        subG = G.subgraph(nodes_c)
        deg = nx.degree_centrality(subG)
        bet = nx.betweenness_centrality(subG)
        clo = nx.closeness_centrality(subG)

        top_deg = max(deg.items(), key=lambda x: x[1])[0]
        top_bet = max(bet.items(), key=lambda x: x[1])[0]
        top_clo = max(clo.items(), key=lambda x: x[1])[0]

        meta = COMMUNITY_METADATA[c_id]
        table_rows.append({
            'Comunidade': meta['name'],
            'Nos': len(nodes_c),
            'Maior_Grau': top_deg,
            'Maior_Intermediacao': top_bet,
            'Maior_Proximidade': top_clo,
            'Escopo_Tematico': meta['scope']
        })

        tex_rows.append(
            f"\\textbf{{{meta['name']}}} & {len(nodes_c)} & \\textit{{{top_deg}}} & \\textit{{{top_bet}}} & \\textit{{{top_clo}}} & \\textit{{{meta['scope']}}} \\\\"
        )

    df_comm = pd.DataFrame(table_rows)
    df_comm.to_csv(TABLE_CSV, index=False, encoding='utf-8')
    print(f"Tabela de comunidades salva em {TABLE_CSV}")

    # Gerar LaTeX (booktabs)
    body_rows = []
    for i, r in enumerate(tex_rows):
        body_rows.append(r)
        if i < 7:
            body_rows.append("\\addlinespace")
    body_str = "\n".join(body_rows)

    tex_code = f"""\\begin{{table}}[htb]
\\centering
\\caption{{Síntese das 8 comunidades detectadas e nós com maiores centralidades (Grau, Intermediação e Proximidade) em cada subgrafo.}}
\\label{{tab:comunidades}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{p{{3.1cm}}ccccp{{5.1cm}}}}
\\toprule
\\textbf{{Comunidade (\\textit{{Bottom-Up}})}} & \\textbf{{Nós}} & \\textbf{{Grau}} & \\textbf{{Interm.}} & \\textbf{{Proxim.}} & \\textbf{{Principais Termos e Escopo Temático}} \\\\
\\midrule
{body_str}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""
    with open(TABLE_TEX, "w", encoding="utf-8") as f:
        f.write(tex_code)
    print(f"Tabela LaTeX de comunidades salva em {TABLE_TEX}")

    # 2. Gerar Visualização do Grafo
    print("\nCalculando layout da rede para renderização de alta resolução...")
    plt.figure(figsize=(15, 12), dpi=300)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

    pos = nx.spring_layout(G, k=0.17, iterations=160, seed=42)

    # Cores dos nós
    node_color_map = {}
    node_comm_id = {}
    for c_id, nodes in comm_map.items():
        for n in nodes:
            node_color_map[n] = COMMUNITY_METADATA[c_id]['color']
            node_comm_id[n] = c_id

    # Desenhar arestas com transparência
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights)
    min_w = min(weights)
    edge_alphas = [0.15 + 0.35 * (w - min_w) / (max_w - min_w + 1e-6) for w in weights]
    edge_widths = [0.4 + 1.2 * (w - min_w) / (max_w - min_w + 1e-6) for w in weights]

    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.6, edge_color='#888888')

    # Desenhar nós
    degrees = dict(G.degree())
    node_sizes = [30 + 18 * degrees[n] for n in G.nodes()]
    node_colors = [node_color_map[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors='white', linewidths=0.8, alpha=0.92)

    # Rotular os termos mais centrais de cada comunidade
    labels_to_draw = {}
    for c_id, nodes in comm_map.items():
        subG = G.subgraph(nodes)
        deg_sub = dict(subG.degree())
        top_nodes = sorted(deg_sub.items(), key=lambda x: x[1], reverse=True)[:5]
        for n, _ in top_nodes:
            labels_to_draw[n] = n

    for n, text in labels_to_draw.items():
        x, y = pos[n]
        c_id = node_comm_id[n]
        color = COMMUNITY_METADATA[c_id]['color']
        plt.text(x, y, text, fontsize=8.5, fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=color, alpha=0.85, linewidth=0.8),
                 zorder=10)

    # Legenda com as 8 comunidades
    legend_patches = []
    for k in range(1, 9):
        c_id = f"C{k}"
        meta = COMMUNITY_METADATA[c_id]
        n_count = len(comm_map[c_id])
        legend_patches.append(
            mpatches.Patch(color=meta['color'], label=f"{meta['name']} (N={n_count})")
        )
    plt.legend(handles=legend_patches, loc='lower left', frameon=True, fontsize=10,
               title="Comunidades Temáticas (Louvain, Q = 0,6363)", title_fontsize=11,
               fancybox=True, framealpha=0.92)

    plt.title("Rede de Coocorrência de Termos da Assembleia Nacional Constituinte de 1988\n"
              r"Grafo Ótimo PPMI ($|V| = 322$ termos, $|E| = 1.823$ arestas, $\text{min\_df} = 2,0\%$, Percentil 95)",
              fontsize=13, fontweight='bold', pad=15)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(FIG_PDF, bbox_inches='tight')
    plt.savefig(ROOT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualizações da rede exportadas em:\n - {FIG_PNG}\n - {FIG_PDF}\n - {ROOT_PNG}")

if __name__ == "__main__":
    main()
