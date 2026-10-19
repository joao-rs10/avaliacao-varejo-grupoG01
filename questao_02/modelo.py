"""Modelo econômico e de incerteza da Questão 2 (Plano de Sobrevivência em 12 Semanas).

Demanda do SKU i na semana t sob o nível de preço k (preço p = preco_atual * nivel_k):

    D = base_i * eps_d * nivel_k^(-elasticidade_i) * (pc_t / pc_0)^theta * (1 + uplift(promo))

  eps_d   ~ lognormal(media 1, sigma_demanda)             choque de demanda
  pc_t    = pc_0 * exp(passeio aleatório, sigma_preco_concorrente)  preço do concorrente
  theta   = sensibilidade ao preço do concorrente (concorrente mais caro -> mais demanda)
  custo logístico = base_i * lognormal(media 1, sigma_custo_logistico)

Cada fonte de variação tem um componente comum a todos os SKUs (correlação `rho`) e um
idiossincrático — assim o risco não "diversifica por completo" entre os 24 SKUs.

Caixa semanal (contribuição) = vendas * (p*(1-taxa) - custo_unitario - custo_logistico_t) - gasto promocional.
Dívidas pagas no fim das semanas 4, 8 e 12; se o saldo não cobre a parcela vencida,
o compromisso FALHA e o valor vencido é acumulado (rola) para o próximo vencimento.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParametrosQ2:
    niveis: tuple = (0.82, 0.88, 0.94, 1.00, 1.06, 1.12)
    n_semanas: int = 12
    caixa_inicial: float = 0.0
    dividas: tuple = ((4, 0.0), (8, 0.0), (12, 0.0))   # (semana, valor)
    sigma_demanda: float = 0.1
    sigma_concorrente: float = 0.05
    sigma_logistico: float = 0.1
    orcamento_semanal: float = 0.0
    limite_risco_falha: float = 0.10
    percentil_risco: float = 5.0        # percentil de risco do caixa final reportado
    rho: float = 0.5                    # correlação (componente comum) entre SKUs
    theta: float = 0.5                  # sensibilidade ao preço do concorrente
    valor_residual_frac: float = 0.0    # valor terminal do estoque (fração do custo) na DP
    # incentivo (mesmo modelo da Q1)
    uplift_max: float = 0.35
    escala_incentivo: float = 0.04
    passo_incentivo: float = 50.0
    limiar_retorno: float = 1.0
    n_quantis: int = 20                 # nós de quantis da demanda usados na DP

    @staticmethod
    def de_config(cfg: dict, **extra) -> "ParametrosQ2":
        v = cfg["variabilidade"]
        return ParametrosQ2(
            niveis=tuple(cfg["niveis_preco_relativos"]),
            n_semanas=cfg["numero_semanas"],
            caixa_inicial=cfg["caixa_inicial"],
            dividas=tuple((int(k.split("_")[1]), float(val)) for k, val in cfg["dividas"].items()),
            sigma_demanda=v["sigma_demanda"],
            sigma_concorrente=v["sigma_preco_concorrente"],
            sigma_logistico=v["sigma_custo_logistico"],
            orcamento_semanal=cfg["orcamento_promocional_semanal"],
            limite_risco_falha=cfg["limite_risco_falha"],
            **extra,
        )


class ModeloQ2:
    def __init__(self, catalogo: pd.DataFrame, params: ParametrosQ2):
        if len(catalogo) == 0:
            raise ValueError("Catálogo vazio.")
        if params.n_semanas < 1:
            raise ValueError("n_semanas >= 1")
        for s, _ in params.dividas:
            if not (1 <= s <= params.n_semanas):
                raise ValueError(f"Vencimento na semana {s} fora do horizonte.")
        self.p = params
        self.df = catalogo.reset_index(drop=True)
        df = self.df
        self.I = len(df)
        self.T = params.n_semanas
        self.niveis = np.array(params.niveis, dtype=float)
        self.K = len(self.niveis)
        self.preco_atual = df["preco_atual"].to_numpy(float)
        self.precos = self.preco_atual[:, None] * self.niveis[None, :]          # (I,K)
        self.taxa = df["taxa_marketplace"].to_numpy(float)
        self.custo = df["custo_unitario"].to_numpy(float)
        self.frete = df["custo_logistico_base"].to_numpy(float)
        self.estoque0 = df["estoque_inicial"].to_numpy(int)
        self.base = df["demanda_base_semanal"].to_numpy(float)
        self.elast = df["elasticidade"].to_numpy(float)
        # fator de preço determinístico da demanda: nivel^(-e)   (I,K)
        self.fator_preco = self.niveis[None, :] ** (-self.elast[:, None])
        self.margem_det = self.precos * (1 - self.taxa[:, None]) - self.custo[:, None]   # sem frete (I,K)
        self.vencimentos = sorted(params.dividas)

    def subconjunto(self, n: int, fator_escala: float | None = None) -> "ModeloQ2":
        """Primeiros n SKUs; dívidas/caixa/orçamento escalam com a capacidade de caixa do subconjunto."""
        n = min(n, self.I)
        sub = self.df.iloc[:n]
        if fator_escala is None:
            fator_escala = self._capacidade(self.df.iloc[:n]) / self._capacidade(self.df)
        p = self.p
        novo = ParametrosQ2(**{**p.__dict__,
                               "caixa_inicial": p.caixa_inicial * fator_escala,
                               "dividas": tuple((s, v * fator_escala) for s, v in p.dividas),
                               "orcamento_semanal": p.orcamento_semanal * fator_escala})
        return ModeloQ2(sub, novo)

    @staticmethod
    def _capacidade(df) -> float:
        m = df["preco_atual"] * (1 - df["taxa_marketplace"]) - df["custo_unitario"] - df["custo_logistico_base"]
        un = np.minimum(df["demanda_base_semanal"], np.maximum(1, df["estoque_inicial"] / 12))
        return float((un * m.clip(lower=0)).sum())
