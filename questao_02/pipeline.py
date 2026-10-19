"""Fluxo integrado Monte Carlo -> DP -> Greedy -> (calibração) -> validação Monte Carlo.

`construir_plano` fecha o ciclo de integração exigido:
  1. MC (construção, n_construcao cenários) estima quantis de demanda e margens por preço  -> DP
  2. DP devolve a política de preços (dependente do estoque) e a trajetória esperada       -> Greedy
  3. Greedy aloca o orçamento promocional semanal sobre essa trajetória
  4. O plano completo volta ao MC (mesmos cenários de CONSTRUÇÃO) para medir P(pagar) por
     vencimento; se algum vencimento ficar abaixo de 1 - limite_risco_falha, o peso de urgência
     das semanas anteriores a ele aumenta (nu_d += passo) e o ciclo 2-4 se repete.
  5. A VALIDAÇÃO final usa cenários independentes (outra TAG de seed) — ver `validar_plano`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamic_programming import PlanoDP, pesos_de_urgencia, reconstruir_trajetoria, resolver_dp
from .greedy import greedy_semanal
from .modelo import ModeloQ2
from .monte_carlo import (TAG_CONSTRUCAO, TAG_VALIDACAO, Estatisticas, PlanoPromo, estatisticas_por_preco,
                          gerar_choques, plano_sem_promo, resumir, rng_para, simular_plano)


@dataclass
class PlanoCompleto:
    stats: Estatisticas
    dp: PlanoDP
    trajetoria: object
    promo: PlanoPromo
    pesos: np.ndarray
    nu: dict
    historico: list = field(default_factory=list)


def construir_plano(modelo: ModeloQ2, seed: int, n_construcao: int = 2000, max_iter: int = 5,
                    passo_nu: float = 0.5, usar_promo: bool = True) -> PlanoCompleto:
    if n_construcao < 2000:
        raise ValueError("O enunciado exige >= 2000 cenários na construção/estimativa.")
    stats = estatisticas_por_preco(modelo, n_construcao, seed)                       # MC  (A)
    choques = gerar_choques(modelo, n_construcao, rng_para(seed, TAG_CONSTRUCAO))   # mesmos cenários
    venc = [s for s, _ in modelo.vencimentos]
    nu = {d: 0.0 for d in venc}
    melhor, historico = None, []
    for it in range(max_iter + 1):
        pesos = pesos_de_urgencia(modelo, nu)
        dp = resolver_dp(modelo, stats, pesos)                                       # DP  (B)
        traj = reconstruir_trajetoria(modelo, dp, stats)
        promo = greedy_semanal(modelo, traj, stats, dp, pesos) if usar_promo else plano_sem_promo(modelo)  # Greedy (C)
        res = simular_plano(modelo, dp.politica, promo, n_construcao, seed, choques=choques)  # volta ao MC
        s = resumir(res, modelo)
        probs = s["prob_pagamento_por_vencimento"]
        historico.append({"iteracao": it, "nu": dict(nu), "prob_pagar_todos": s["prob_pagar_todos_no_prazo"],
                          **{f"P_{k}": v for k, v in probs.items()},
                          "caixa_final_esperado": s["caixa_final_esperado"],
                          "gasto_promocional": float(promo.gasto.sum())})
        cand = PlanoCompleto(stats, dp, traj, promo, pesos, dict(nu))
        chave = (s["prob_pagar_todos_no_prazo"], s["caixa_final_esperado"])
        if melhor is None or chave > melhor[0]:
            melhor = (chave, cand)
        falhando = [d for d in venc if probs[f"semana_{d}"] < 1 - modelo.p.limite_risco_falha]
        if not falhando:
            break
        for d in falhando:
            nu[d] += passo_nu
    plano = melhor[1]
    plano.historico = historico
    return plano


def validar_plano(modelo: ModeloQ2, plano: PlanoCompleto, seed: int, n_validacao: int = 5000,
                  skus_registrar=None, usar_promo: bool = True) -> dict:
    """Validação em cenários INDEPENDENTES (TAG_VALIDACAO != TAG_CONSTRUCAO)."""
    if n_validacao < 5000:
        raise ValueError("O enunciado exige >= 5000 cenários independentes na validação final.")
    promo = plano.promo if usar_promo else plano_sem_promo(modelo)
    return simular_plano(modelo, plano.dp.politica, promo, n_validacao, seed, TAG_VALIDACAO, skus_registrar)
