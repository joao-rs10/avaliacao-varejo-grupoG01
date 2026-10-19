# Complexidade (Big O) — tempo e memória

## Variáveis

| Símbolo | Significado | Valor na instância G01 |
|---|---|---|
| n | nº de SKUs da Questão 1 (ou do subconjunto, no Brute Force) | 30 (subconjunto: 8) |
| m_i / m | nº de níveis de preço permitidos do SKU i / máximo (após o piso de margem) | m ≤ 7 |
| M = Π m_i | nº de combinações do Brute Force | 246.960 (n=8) |
| P, G | população e nº de gerações do GA | A: 60/150, B: 160/300 |
| t_o | tamanho do torneio | 3 (A), 5 (B) |
| B, Δ | orçamento promocional e passo de incentivo | 71.806,75 e 100 |
| N | nº de cenários Monte Carlo | 2.000 (construção), 5.000 (validação) |
| T, I, K | semanas, SKUs e níveis de preço da Questão 2 | 12, 24, 6 |
| S_i, ΣS | estoque inicial do SKU i e soma dos estoques | ΣS = 3.893 |
| Q | nós de quantis da demanda usados na DP | 20 |

## Questão 1

| Algoritmo | Tempo | Memória | Trecho dominante | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Brute Force ingênuo | O(M · n) | O(n) | laço `itertools.product` chamando `fitness_ingenuo` (`brute_force.py`) | M cresce exponencialmente (m^n): 7^30 ≈ 2·10^25 → GA |
| Brute Force vetorizado | O(M) operações elementares (+ O(n·m) de pré-cálculo) | O(n·m + Π m_cauda) ≈ 200 mil floats | soma por broadcast da "cauda" + laço sobre a "cabeça" | mesmo problema exponencial; só adia o limite (~10^7–10^8 combinações) |
| Algoritmo Genético | O(G · P · (n + t_o)) — fitness O(n) por indivíduo, seleção O(t_o) por filho, crossover e mutação O(n) | O(P · n + n · m + G) | `fitness_pop` (indexação numpy) e a geração de máscaras por geração | não garante ótimo; validado contra o Brute Force. Em fitness não aditivo cresceria o custo de avaliar |
| Greedy (incentivos) | O(n + (B/Δ) · log n) | O(n) | `heappop/heappush` em `alocar_guloso` | ótimo para ganhos côncavos; com ganhos não côncavos usar DP/mochila |

## Questão 2

| Algoritmo | Tempo | Memória | Trecho dominante | Quando deixa de ser viável → alternativa |
|---|---|---|---|---|
| Monte Carlo (estatísticas) | O(K · N · T · I) para gerar demanda + O(K · T · I · N log N) para os quantis | O(N · T · I) (um nível por vez) + O(I·T·K·Q) | `np.quantile` em `estatisticas_por_preco` | N·T·I muito grande → amostrar em lotes / reduzir variância (antitéticas) |
| Programação Dinâmica | O(T · K · Q · ΣS) — cada estado (t,s) testa K preços × Q quantis | O(T · ΣS) (política e V por SKU) | laço sobre `k` com `np.interp` em `resolver_dp` | estoques da ordem de 10^5+ por SKU → discretizar o estoque em baldes |
| Greedy semanal | O(T · (B_s/Δ) · log I) | O(T · I) (matriz de gastos) | heap em `alocar_guloso` + interpolação de V | — |
| Validação Monte Carlo | O(N · T · I) | O(N · T · I) | laço semanal vetorizado de `simular_plano` | N·T·I ≫ memória → simular em lotes |

### Onde há reutilização de subproblemas (DP)
`V(t+1, ·)` é calculado uma única vez por semana e consultado por todos os `(s, k, q)` da semana `t`:
são **47.004 subproblemas memoizados** (`T · Σ(S_i+1)`) atendendo **5.640.480 consultas** (≈120 por subproblema).
Sem memoização, a recursão ingênua teria (K·Q)^T = 120^12 ramificações por SKU.

## Medições reais
Ver `docs/desempenho.md` (tempo com `time.perf_counter`, memória com `tracemalloc`, entradas pequena/média/grande).
