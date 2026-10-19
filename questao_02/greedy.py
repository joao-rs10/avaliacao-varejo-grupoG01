"""Etapa C — Greedy semanal: destaques/mídia/frete promocional escassos.

Entrada: trajetória de preços da DP (nível, estoque esperado e vendas esperadas por semana).
Em cada semana t, o orçamento promocional semanal é distribuído entre os SKUs pelo maior
benefício marginal por real investido (mesmo modelo côncavo de common/incentivo.py):

    e_i(x)     = min(estoque_apos_vendas_i, demanda_i * uplift(x))        # unidades extra vendidas
    ganho_i(x) = margem_i * e_i(x) - [V(t+1, s) - V(t+1, s - e_i(x))] / w_t

O segundo termo é o CUSTO DE OPORTUNIDADE do estoque, lido da função valor da DP: vender
agora uma unidade que seria vendida em semana futura não gera caixa novo. Sem esse termo o
Greedy superestima o ganho (verificado: ganho previsto ~98 mil vs. realizado ~36 mil).

O orçamento NÃO se acumula entre semanas (é um recurso semanal escasso).
"""
from __future__ import annotations

import numpy as np

from common.incentivo import alocar_guloso, uplift

from .dynamic_programming import PlanoDP
from .modelo import ModeloQ2
from .monte_carlo import Estatisticas, PlanoPromo


def greedy_semanal(modelo: ModeloQ2, traj, stats: Estatisticas, plano_dp: PlanoDP,
                   pesos=None) -> PlanoPromo:
    """traj: DataFrame de reconstruir_trajetoria. Tempo O(T·(B/passo)·log I)."""
    p = modelo.p
    T, I = modelo.T, modelo.I
    pesos = np.ones(T) if pesos is None else np.asarray(pesos, dtype=float)
    gasto = np.zeros((T, I))
    fator = np.ones((T, I))
    por_semana = {t: g.sort_values("i") for t, g in traj.groupby("semana")}
    for t in range(T):
        g = por_semana[t + 1]
        k = g["k"].to_numpy()
        ar = np.arange(I)
        Dm = stats.demanda_media[ar, t, k]
        m = stats.margem_media[ar, t, k]
        livre = np.maximum(0.0, g["estoque_inicio"].to_numpy() - g["vendas_esperadas"].to_numpy())
        S_i = modelo.estoque0
        receita = np.maximum(g["vendas_esperadas"].to_numpy() * g["preco"].to_numpy(), 1.0)
        escala = p.escala_incentivo * receita

        def ganho(i, x, Dm=Dm, m=m, livre=livre, escala=escala, t=t):
            e = min(livre[i], Dm[i] * float(uplift(x, p.uplift_max, escala[i])))
            if e <= 0.0:
                return 0.0
            V = plano_dp.valor[i][t + 1]
            grade = np.arange(len(V), dtype=float)
            custo_oport = np.interp(livre[i], grade, V) - np.interp(livre[i] - e, grade, V)
            return m[i] * e - custo_oport / pesos[t]

        x, _ = alocar_guloso(ganho, I, p.orcamento_semanal, p.passo_incentivo, p.limiar_retorno)
        gasto[t] = x
        fator[t] = 1.0 + uplift(x, p.uplift_max, escala)
    return PlanoPromo(gasto, fator)
