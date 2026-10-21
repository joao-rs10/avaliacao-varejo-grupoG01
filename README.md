# Checkpoint 6 — Recuperação financeira de varejistas por precificação online

**Grupo: G01** — integrantes: `Joâo Vitor Reis Silva      rm:563115` 
**Seed base: 21**
**Seed final: 1021**  (= 1000 × 1 + 21)
**ID da instância: 4f84efec9073**

> ⚠️ **Atenção — dados de exemplo.** Estes valores vêm do notebook com o código `G01` e as matrículas de
> **exemplo** (`RM123456`, `RM234567`, `RM345678`). Para a entrega oficial: edite só o Bloco 02 do notebook
> `data/Gerador_Inputs_Checkpoint_6.ipynb`, execute tudo, substitua os 5 arquivos de `data/`, rode
> `bash reproduzir.sh` (trocando `--seed 1021` pela seed_final nova) e atualize esta seção e as tabelas abaixo.
> O código não depende da instância (lê o grupo do `manifesto_dados.json` e confere os hashes).
> Nota: as saídas salvas dentro do notebook original mostram `seed_final: 21`, de uma versão anterior do código;
> o código atual calcula `1000 × grupo + seed_base`, que é o que o enunciado pede.

Python 3.13.16 · dependências em `requirements.txt` (numpy, pandas, matplotlib, pytest).

## Como reproduzir
```bash
pip install -r requirements.txt
python -m pytest -q                                   
python questao_01/main.py --seed 1021 --modo completo 
python questao_02/main.py --seed 1021 --cenarios 5000 
```
Todo o aleatório usa seed explícita; o mesmo comando gera os mesmos números.

## Cenário-surpresa (só por parâmetros)
```bash
python questao_01/main.py --meta-mult 1.10 --orcamento-mult 0.75 --custo-logistico-mult 1.15 \
                          --remover-sku Q1-005 --populacao 100 --mutacao 0.05
python questao_02/main.py --preco-concorrente-mult 0.92 --custo-logistico-mult 1.15 \
                          --orcamento-mult 0.75 --divida-mult 1.5 --remover-sku Q2-001
```
(`common/cenarios.py` aplica as mudanças aos dados/configuração; os algoritmos não são tocados.)

## Questão 1 — Operação Caixa 90
**Nesta questão utilizamos 3 algoritmos.** O **Brute Force** (`brute_force.py`) resolve exatamente um
subconjunto de 8 SKUs (246.960 combinações) e produz o ótimo de referência. O **Algoritmo Genético**
(`genetic_algorithm.py`) roda nesse mesmo subconjunto (gap % contra o ótimo) e depois no catálogo completo,
gerando a política de preços. O **Greedy** (`greedy.py`) recebe a política do GA e distribui o orçamento
promocional pelo maior benefício marginal por real. Fluxo: BF → valida GA → GA → Greedy (`main.py`).

| Resultado principal | Valor | Produzido por |
|---|---|---|
| Gap GA vs Brute Force (10 execuções) | 0,00% (média e máx.) | `main.py` → `resultados/tabela_bf_vs_ga.csv` |
| Fitness do GA por geração | gráfico | `resultados/ga_fitness_por_geracao.png` |
| Comparação de 2 configs do GA | A (P=60) × B (P=160): mesmo fitness; B mais cara | `resultados/ga_comparacao_configs.csv` |
| Antes/depois do Greedy por SKU | tabela | `resultados/tabela_antes_depois_greedy.csv` |
| Caixa: preço atual → GA → GA+Greedy | 407.616 → 511.103 → 529.036 | `resultados/tabela_comparativa_politicas.csv` |
| Meta de caixa (328.938) | **atingida**, folga R$ 200.098 | `resultados/resultado_final.json` |
| Cobertura da dívida (309.715) | 170,8% | idem |
| Margem final / estoque remanescente | 23,7% / ≈1.037 un. | idem |
| Orçamento promocional | R$ 26.100 de R$ 71.807 (36%); retorno 1,69 por R$ | idem |
| Otimização antes/depois | BF n=6: 15,1 s → 0,0006 s | `resultados/otimizacao_antes_depois.json` (`otimizacao.py`) |
| Tempo e memória | ver `docs/desempenho.md` | `resultados/desempenho.json` |

## Questão 2 — Plano de Sobrevivência em 12 Semanas
**Nesta questão utilizamos 3 algoritmos.** O **Monte Carlo** (`monte_carlo.py`) estima, por SKU/semana/preço, a
distribuição da demanda e a margem sob 3 fontes de incerteza (demanda, preço do concorrente, custo logístico) e,
no fim, valida o plano em 5.000 cenários independentes. A **Programação Dinâmica** (`dynamic_programming.py`)
escolhe a trajetória semanal de preços (estado = semana × estoque). O **Greedy** (`greedy.py`) aloca o orçamento
promocional semanal sobre o plano da DP. Fluxo: MC → DP → Greedy → MC (`pipeline.py`); diagrama em `docs/arquitetura.png`.

| Resultado principal (5.000 cenários independentes) | Valor | Produzido por |
|---|---|---|
| P(pagar semana 4 / 8 / 12) | 100% / 100% / 100% | `resultados/resultado_final.json`, `probabilidade_pagamento.png` |
| P(pagar as três) / cenários de falha | 100% / 0 | idem |
| Caixa final esperado / P5 (percentil de risco) | R$ 360.905 / R$ 331.612 | idem; `distribuicao_caixa_final.png` |
| Margem esperada / estoque final | 26,9% / 543 un. (13,9%) | idem |
| Trajetória de preço e estoque (3 SKUs) | gráfico + CSV completo | `trajetoria_preco_estoque.png`, `trajetoria_precos_dp.csv` |
| Plano promocional semanal | R$ 25.800 de R$ 56.900 | `plano_promocional_greedy.csv` |
| Comparativo (preço fixo × DP × DP+Greedy) | caixa 152.539 → 344.218 → 360.905 | `comparativo_politicas.csv` |
| Teste de estresse (dívidas ×1,5 / ×2,0) | plano 100% / 96% × preço fixo 14% / 0% | `teste_estresse_dividas.csv` |
| Escalabilidade (3 tamanhos) | ver `docs/desempenho.md` | `escalabilidade.csv`, `questao_02/escalabilidade.py` |
| Reuso de subproblemas na DP | 47.004 estados atendem 5,6 mi consultas | `resultado_final.json` → `dp` |

Nesta instância o risco de inadimplência já é baixo com preço fixo; o ganho do plano aparece no caixa esperado
e no teste de estresse (veja a seção "Limite honesto" em `docs/decisoes.md`).

## Estrutura
`common/` (dados + integridade, medição, incentivo, cenários) · `questao_01/` · `questao_02/` · `tests/` ·
`docs/` (`arquitetura.png`, `complexidade.md`, `desempenho.md`, `decisoes.md`) · `data/` (instância oficial,
**não editar** — os hashes SHA-256 são conferidos a cada execução) · `IA_USO.md`.
