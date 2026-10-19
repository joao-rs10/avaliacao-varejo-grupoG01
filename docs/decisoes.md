# Decisões de modelagem e premissas

O enunciado deixa várias escolhas em aberto. Estas são as adotadas; todas são parâmetros
(`ParametrosQ1`, `ParametrosQ2`) e podem ser alteradas sem mexer nos algoritmos.

## Questão 1
- **Demanda:** `demanda(p) = demanda_base_90d · nível^(-elasticidade)`; vendas = `min(estoque, round(demanda))`.
- **Caixa operacional:** contribuição `unidades · (p·(1−taxa_marketplace) − custo_unitario − custo_logistico)`.
  Foi a mesma definição que o gerador usou para calibrar meta e dívida.
- **Piso de margem:** níveis cuja margem % < `margem_minima_percentual` ficam fora do domínio do SKU
  (o GA e o Brute Force nem os enxergam). Na instância G01, 3 SKUs (Q1-007, Q1-008, Q1-009) não têm nenhum
  nível viável: foram **retirados do canal** (vendem 0), pois vender destruiria caixa.
- **Custo de carregamento do estoque remanescente** (`custo · (2% + 10%·idade/360)`): sem ele, descontos
  quase nunca valeriam a pena e o GA só escolheria preços altos. É uma premissa nossa, não do enunciado.
- **Teto de desconto:** concessões totais ≤ 0,5% da receita base (penalidade de R$ 5 por R$ excedido).
  Evita "queima" indiscriminada e acopla os SKUs (sem ele o fitness seria separável por SKU).
- **GA:** cromossomo = posição no conjunto de níveis permitidos de cada SKU (respeita a margem por construção);
  seleção por torneio (pressão ajustável); crossover uniforme (SKUs são genes independentes, sem blocos a preservar);
  mutação por gene sorteando outro nível permitido; elitismo (o melhor nunca regride); parada por nº máximo de
  gerações ou paciência (gerações sem melhora). Config A (P=60) e B (P=160) comparadas.
- **Greedy:** `uplift(x) = a·(1−e^(−x/k))` (a=35%, k=4% da receita do SKU): retorno marginal decrescente.
  Investe `Δ=R$100` por vez no SKU com maior (Δganho/Δ) enquanto o retorno marginal > 1. Só vale onde o preço do GA
  deixou estoque (por isso depende do GA). Usou 36% do orçamento: o restante não se pagaria.

### Limites honestos da Q1
Nesta instância o fitness é quase aditivo por SKU e o teto de desconto só aperta alguns SKUs, então o GA
atinge o ótimo do Brute Force em 100% das 10 execuções (gap 0%). Isso valida o GA, mas **não** prova que ele
seria suficiente em problemas fortemente acoplados.

## Questão 2
- **Incerteza:** demanda (lognormal, média 1), preço do concorrente (passeio aleatório; efeito na demanda via
  `(pc_t/pc_0)^θ`, θ=0,5) e custo logístico (lognormal, média 1), cada um com componente comum (ρ=0,5) + idiossincrático.
- **Dívida:** paga no fim das semanas 4, 8 e 12 se o saldo cobrir; senão **falha** e o valor vencido rola para o
  próximo vencimento. `P(pagar)` por vencimento é a fração de cenários que pagam no prazo.
- **DP:** estado (semana, estoque), decisão = nível de preço, transição estocástica via Q=20 quantis da demanda
  (vindos do Monte Carlo), V(t+1, ·) interpolado linearmente. Política dependente do estoque, avaliada em cenários
  novos na validação. Limitação: a DP trata as semanas como independentes (ignora a correlação serial do preço do
  concorrente); a validação por simulação completa captura essa dinâmica.
- **Pesos de urgência:** como a DP é por SKU, o acoplamento com as dívidas é feito por pesos `w_t ≥ 1` nas semanas
  anteriores a cada vencimento. Se `P(pagar)` de algum vencimento ficar abaixo de `1 − limite_risco_falha` nos
  cenários de construção, o peso sobe (+0,5) e o ciclo MC→DP→Greedy→MC se repete (até 5 vezes).
- **Greedy semanal:** mesmo modelo de incentivo da Q1, mas o ganho desconta o **custo de oportunidade do estoque**
  lido da função valor da DP. Na primeira versão (sem isso) o ganho previsto era ~R$ 98 mil contra ~R$ 36 mil realizado
  na simulação; com a correção o Greedy passou a ter retorno líquido positivo (+R$ 16,7 mil na amostra de teste).
- **Percentil de risco escolhido pelo grupo:** P5 do caixa final (parâmetro `--percentil-risco`).
- **Independência:** construção usa `SeedSequence([seed, 301])`; validação usa `SeedSequence([seed, 302])`.

### Limite honesto da Q2
Na instância G01 o plano paga as três dívidas em 100% dos 5.000 cenários e o preço fixo atual também. A dívida
total é ~72–88% do caixa base de 12 semanas e o plano ainda eleva o caixa. O valor da DP aparece no
**teste de estresse** (`teste_estresse_dividas.csv`): com dívidas ×1,5 o preço fixo paga as 3 em só 14% dos cenários
e o plano em 100%; com ×2,0, 0% contra 96%; a partir de ×2,5 nenhum dos dois consegue.
