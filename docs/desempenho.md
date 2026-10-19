# Desempenho medido

Medições com `time.perf_counter()` (tempo) e `tracemalloc` (pico de memória), via `common/medicao.py`
(a função é executada duas vezes: uma só para o tempo, outra para a memória, porque o tracemalloc
distorce o tempo). Máquina de execução: sandbox Linux, Python 3.13.16, numpy 2.5.3. Os tempos variam
por máquina; as proporções é que importam. Arquivos de origem: `questao_01/resultados/*.json|csv` e
`questao_02/resultados/*.json|csv`.

## Questão 1 (seed 1021)

| Etapa | Entrada | Tempo | Pico de memória |
|---|---|---|---|
| Brute Force (vetorizado) | n=8, 246.960 combinações | 0,020 s | 3,8 MiB |
| GA config A no subconjunto | n=8 | 0,004 s | 0,04 MiB |
| GA config A (completo) | n=30, P=60 | 0,009 s | 0,12 MiB |
| GA config B (completo) | n=30, P=160 | 0,018 s | 0,32 MiB |
| Greedy | n=30 | 0,002 s | 0,005 MiB |

### Otimização de código — antes/depois (`otimizacao_antes_depois.json`)

| Comparação | Antes | Depois | Ganho |
|---|---|---|---|
| Brute Force, n=6 (20.580 combinações) | 15,09 s (laço Python + `pandas.iat` a cada combinação) | 0,0006 s (tabelas pré-calculadas + soma por broadcast) | ≈ 23.700× |
| Fitness de 60 indivíduos × 20 repetições | 4,34 s | 0,0006 s | ≈ 6.900× |

Ressalva honesta: o "antes" é propositalmente ingênuo (acessos `iat` do pandas dentro do laço), por
isso o ganho é enorme. O ponto da otimização é mostrar o mecanismo: tirar o modelo econômico de dentro
do laço (pré-cálculo por SKU×nível) e vetorizar.

### Comparação de configurações do GA (`ga_comparacao_configs.csv`)

| Config | P | Mutação | Elite | Torneio | Gerações executadas | Geração do melhor | Fitness final | Tempo | Memória |
|---|---|---|---|---|---|---|---|---|---|
| A | 60 | 0,03 | 2 | 3 | 66 | 26 | 471.879,22 | 0,009 s | 0,12 MiB |
| B | 160 | 0,015 | 4 | 5 | 79 | 19 | 471.879,22 | 0,018 s | 0,32 MiB |

As duas chegam ao mesmo fitness; B converge em menos gerações (19 vs 26) porém custa ~2× em tempo e
~2,6× em memória. A é a escolhida (mesmo resultado, mais barata).

## Questão 2 (seed 1021, 2.000 cenários de construção, 5.000 de validação)

| Etapa | Tempo | Pico de memória |
|---|---|---|
| Monte Carlo (estatísticas) | 0,25 s | 31,1 MiB |
| Programação Dinâmica | 0,09 s | 0,78 MiB |
| Greedy semanal | 0,014 s | 0,09 MiB |
| Validação (5.000 cenários) | 0,44 s | 76,9 MiB |

### Escalabilidade (`escalabilidade.csv`)

| Eixo | SKUs | Cenários (constr./valid.) | MC tempo / mem | DP tempo / mem | Validação tempo / mem |
|---|---|---|---|---|---|
| SKUs | 6 | 2000 / 5000 | 0,056 s / 7,8 MiB | 0,031 s / 0,54 MiB | 0,059 s / 19,2 MiB |
| SKUs | 12 | 2000 / 5000 | 0,124 s / 15,6 MiB | 0,046 s / 0,64 MiB | 0,083 s / 38,5 MiB |
| SKUs | 24 | 2000 / 5000 | 0,223 s / 31,1 MiB | 0,084 s / 0,78 MiB | 0,320 s / 76,9 MiB |
| Cenários | 24 | 2000 / 5000 | 0,234 s / 31,1 MiB | 0,088 s / 0,78 MiB | 0,155 s / 76,9 MiB |
| Cenários | 24 | 3500 / 5000 | 0,433 s / 54,2 MiB | 0,098 s / 0,78 MiB | 0,209 s / 76,9 MiB |
| Cenários | 24 | 5000 / 5000 | 0,604 s / 77,3 MiB | 0,085 s / 0,78 MiB | 0,154 s / 76,9 MiB |

Leitura: o tempo e a memória do Monte Carlo crescem ~linearmente em I (SKUs) e em N (cenários), como
previsto em O(N·T·I); a memória da validação é N·T·I; a DP quase não depende de N (só dos quantis) e
cresce suavemente com ΣS. Em tamanhos tão pequenos os tempos têm ruído (ex.: a validação com I=24
aparece com 0,32 s numa linha e 0,15 s em outra — mesma configuração, diferença de aquecimento/cache).
