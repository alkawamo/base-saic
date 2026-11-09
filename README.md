# Base de Dados SAIC (Assembleia Nacional Constituinte de 1988) — Versão Tratada e Modelagem em Grafos

Este repositório disponibiliza a versão higienizada, padronizada e enriquecida da base de dados histórica do **Sistema de Apoio Informático à Constituinte (SAIC)**, originalmente digitada pelo Centro de Informática e Processamento de Dados do Senado Federal (Prodasen) entre 1986 e 1987, bem como o *notebook* (Google Colab / Jupyter) para reprodução da modelagem em grafos de coocorrência de palavras e detecção de comunidades temáticas.

---

## 1. Sobre o Acervo SAIC

Durante os trabalhos da Assembleia Nacional Constituinte que resultaram na Constituição Brasileira de 1988, cidadãos de todo o país enviaram sugestões por meio de formulários distribuídos em agências dos Correios, prefeituras e assembleias legislativas. O acervo reúne **72.719 registros (cartas/formulários)**.

Na versão tratada disponibilizada neste repositório (`SAIC.csv`), foram realizados os seguintes procedimentos de curadoria de dados:
- **Padronização de *encoding* (UTF-8) e valores ausentes (`N/A`)**, incluindo a limpeza de artefatos de planilhas eletrônicas (como `#NOME?`, asteriscos e símbolos residuais);
- **Correção de deslocamentos estruturais entre colunas** (realocação de macrotemas que constavam em `DESTINATÁRIO` para `CATÁLOGO`, transferência de subtemas que estavam em `CATÁLOGO` para `INDEXAÇÃO`, e extração de trechos de cartas que estavam em `INDEXAÇÃO` para `TEXTO SUGESTÃO`);
- **Padronização geográfica (`MUNICÍPIO` e `UF`)** via cruzamento com a base oficial de municípios brasileiros, preservando o recorte histórico de 1986–1987 (municípios do atual estado do Tocantins mantidos vinculados a Goiás, conforme a divisão territorial vigente à época);
- **Desambiguação de nomes próprios (`NOME` e `DESTINATÁRIO`)** via *Fuzzy Matching* (distância de Levenshtein com limiar de 90%) e criação do identificador único de remetente (**`ID PESSOA`**), consolidando as 72.719 cartas em **64.869 cidadãos distintos**.

---

## 2. Estrutura do Repositório

- **`SAIC.csv`**: Base de dados completa higienizada e padronizada (`72.719 linhas x 21 colunas`, codificação `UTF-8`, separador ponto e vírgula `;`).
- **`analise_grafo_saic.ipynb`**: Notebook Python compatível com Google Colaboratory / Jupyter contendo todo o fluxo de processamento de linguagem natural (`spaCy` — `pt_core_news_md`), construção da matriz documento-termo binária, cálculo da matriz de coocorrência ponderada por *Positive Pointwise Mutual Information* (PPMI), poda por percentis, detecção de comunidades via algoritmo de Louvain, cálculo de métricas de centralidade e geração das visualizações.

---

## 3. Dicionário de Dados (`SAIC.csv`)

| # | Coluna | Tipo | Descrição |
|---|---|---|---|
| 1 | `ID PESSOA` | Inteiro | Identificador único atribuído a cada cidadão remetente (baseado na combinação padronizada de `NOME`, `MUNICÍPIO` e `UF`). |
| 2 | `NOME` | Texto | Nome do cidadão remetente da carta (normalizado e desambiguado). |
| 3 | `DESTINATÁRIO` | Texto | Parlamentar (deputado constituinte ou senador) a quem a carta foi opcionalmente direcionada. |
| 4 | `CATÁLOGO` | Texto | Classificação temática principal atribuída pelo Prodasen (composta por um ou mais dos 24 macrotemas oficiais, separados por vírgula). |
| 5 | `INDEXAÇÃO` | Texto | Subtemas específicos atribuídos pelo Prodasen (combinações dos 1.308 subtemas possíveis, separados por vírgula). |
| 6 | `TEXTO SUGESTÃO` | Texto | Transcrição integral da sugestão, crítica ou manifestação enviada pelo cidadão. |
| 7 | `ORIGEM` | Texto | Código interno de lote/origem do formulário no Prodasen. |
| 8 | `DATA` | Data | Data de registro inicial do lote. |
| 9 | `FORMUL` | Inteiro | Número sequencial do formulário dentro do lote. |
| 10 | `DV` | Inteiro | Dígito verificador do formulário. |
| 11 | `DATA2` | Data | Data secundária de processamento no sistema SAIC. |
| 12 | `MUNICÍPIO` | Texto | Município de residência do remetente (grafia e acentuação padronizadas). |
| 13 | `UF` | Texto | Sigla da Unidade da Federação do remetente (26 UFs vigentes em 1986–1987). |
| 14 | `CEP` | Inteiro | Código de Endereçamento Postal (formato de 5 dígitos vigente na década de 1980). |
| 15 | `SEXO` | Texto | Sexo informado pelo remetente (`Masculino`, `Feminino` ou `N/A`). |
| 16 | `MORADOR` | Texto | Situação de domicílio (`Zona urbana`, `Zona rural` ou `N/A`). |
| 17 | `INSTRUÇÃO` | Texto | Grau de escolaridade informado pelo remetente (8 faixas possíveis ou `N/A`). |
| 18 | `ESTADO CIVIL` | Texto | Estado civil informado pelo remetente (5 categorias ou `N/A`). |
| 19 | `FAIXA ETÁRIA` | Texto | Faixa de idade do remetente (8 faixas possíveis ou `N/A`). |
| 20 | `FAIXA RENDA` | Texto | Faixa de renda em salários mínimos da época (8 faixas possíveis ou `N/A`). |
| 21 | `ATIVIDADE` | Texto | Setor de atividade econômica ou profissão do remetente (10 categorias ou `N/A`). |

---

## 4. Como Reproduzir no Google Colab ou Localmente

1. Abra o arquivo `analise_grafo_saic.ipynb` no Google Colab (ou em um ambiente Jupyter local com Python 3.10+).
2. Certifique-se de que o arquivo `SAIC.csv` esteja no mesmo diretório (ou carregado na sessão do Colab).
3. Execute as células sequencialmente (o notebook instala e carrega automaticamente o modelo `pt_core_news_md` do `spaCy` caso não esteja presente).
