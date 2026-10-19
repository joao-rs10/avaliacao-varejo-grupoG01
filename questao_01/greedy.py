"""Etapa C — Greedy: distribui o orçamento promocional sobre a política do GA.

Entrada: política de preços produzida pelo GA (integração GA -> Greedy).
Para cada SKU, o ganho bruto de caixa ao investir x reais é

    ganho_i(x) = margem_un_i * min(estoque_livre_i, demanda_i * uplift(x))

onde `estoque_livre` é o que sobrou após as vendas da política (se o preço do GA já
esgota o estoque, o incentivo não tem retorno — por isso o Greedy depende do GA).
A regra gulosa investe sempre no SKU de maior benefício marginal por real investido e
para quando o retorno marginal cai abaixo de `limiar_retorno` ou o orçamento acaba.
"""
from __future__ import annotations

import numpy as np

from common.incentivo import alocar_guloso, uplift

from .modelo import ModeloQ1


def greedy_incentivos(modelo: ModeloQ1, pos):
    p = modelo.p
    k = modelo.niveis_da_politica(pos)
    ar = np.arange(modelo.n)
    unid = modelo.unidades[ar, k]
    margem = modelo.margem_un[ar, k]
    demanda = modelo.demanda[ar, k]
    livre = np.where(modelo.retirado, 0.0, modelo.estoque - unid)
    receita = np.maximum(modelo.receita[ar, k], 1.0)
    escala = p.escala_incentivo * receita

    def ganho(i, x):
        extra = demanda[i] * float(uplift(x, p.uplift_max, escala[i]))
        return margem[i] * min(livre[i], extra)

    x, iteracoes = alocar_guloso(ganho, modelo.n, p.orcamento_promocional,
                                 p.passo_incentivo, p.limiar_retorno)
    ganho_bruto = np.array([ganho(i, x[i]) for i in range(modelo.n)])
    extra_un = np.array([min(livre[i], demanda[i] * float(uplift(x[i], p.uplift_max, escala[i])))
                         for i in range(modelo.n)])
    return {"investimento": x, "ganho_bruto": ganho_bruto, "unidades_extra": extra_un,
            "iteracoes": iteracoes, "livre": livre}
