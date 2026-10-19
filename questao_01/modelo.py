"""Modelo econômico da Questão 1 (Operação Caixa 90).

Tudo que os três algoritmos (Brute Force, GA, Greedy) compartilham está aqui:
tabelas pré-calculadas por (SKU, nível de preço), a função de fitness e a
avaliação completa de uma política.

Premissas (documentadas também em docs/decisoes.md):
  * demanda(p)  = demanda_base_90d * nivel^(-elasticidade)       (elasticidade constante)
  * unidades    = min(estoque, round(demanda(p)))
  * margem_un   = p*(1-taxa_marketplace) - custo_unitario - custo_logistico
  * caixa       = unidades * margem_un           (caixa operacional de contribuição)
  * nível viável: margem_un / p >= margem_minima_percentual (piso de margem)
  * SKU sem nenhum nível viável é RETIRADO do canal (vende 0 un.; vender destruiria caixa)
  * custo de carregamento do estoque remanescente (capital parado, cresce com a idade)
  * teto de desconto: soma das concessões (unid. * preco_atual * (1-nivel)) <= fração da
    receita base  -> evita "queima" indiscriminada e acopla os SKUs entre si
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

PENALIDADE_DESCONTO = 5.0  # R$ de fitness perdidos por R$ de desconto acima do teto


@dataclass(frozen=True)
class ParametrosQ1:
    niveis: tuple = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10)
    margem_minima: float = 0.05
    orcamento_promocional: float = 0.0
    meta_caixa: float = 0.0
    divida_total: float = 0.0
    horizonte_dias: int = 90
    # carregamento do estoque remanescente: custo * (base + idade_dias/360 * adicional)
    carreg_base: float = 0.02
    carreg_idade: float = 0.10
    teto_desconto_frac: float = 0.005  # fração da receita base que pode virar desconto
    # incentivo comercial
    uplift_max: float = 0.35
    escala_incentivo: float = 0.04  # k_i = escala * receita do SKU na política
    passo_incentivo: float = 100.0
    limiar_retorno: float = 1.0

    @staticmethod
    def de_config(cfg: dict, **extra) -> "ParametrosQ1":
        return ParametrosQ1(
            niveis=tuple(cfg["niveis_preco_relativos"]),
            margem_minima=cfg["margem_minima_percentual"],
            orcamento_promocional=cfg["orcamento_promocional"],
            meta_caixa=cfg["meta_caixa_90d"],
            divida_total=cfg["divida_total_90d"],
            horizonte_dias=cfg.get("horizonte_dias", 90),
            **extra,
        )


class ModeloQ1:
    """Tabelas (n SKUs x K níveis) e operações vetorizadas sobre políticas.

    Uma política é um vetor de inteiros `pos` de tamanho n: `pos[i]` é a POSIÇÃO
    dentro da lista de níveis permitidos do SKU i (`self.permitidos[i]`).
    """

    def __init__(self, catalogo: pd.DataFrame, params: ParametrosQ1):
        if len(catalogo) == 0:
            raise ValueError("Catálogo vazio.")
        self.p = params
        self.df = catalogo.reset_index(drop=True)
        df = self.df
        self.n = len(df)
        niv = np.array(params.niveis, dtype=float)
        if (niv <= 0).any():
            raise ValueError("Níveis de preço devem ser positivos.")
        self.niveis = niv
        self.K = len(niv)

        preco_atual = df["preco_atual"].to_numpy(float)[:, None]
        self.precos = preco_atual * niv[None, :]                                  # (n,K)
        taxa = df["taxa_marketplace"].to_numpy(float)[:, None]
        custo = df["custo_unitario"].to_numpy(float)[:, None]
        frete = df["custo_logistico"].to_numpy(float)[:, None]
        self.estoque = df["estoque"].to_numpy(int)
        elast = df["elasticidade"].to_numpy(float)[:, None]
        base = df["demanda_base_90d"].to_numpy(float)[:, None]
        idade = df["idade_estoque_dias"].to_numpy(float)

        self.demanda = base * niv[None, :] ** (-elast)                           # (n,K) float
        self.margem_un = self.precos * (1 - taxa) - custo - frete                # (n,K)
        self.viavel = (self.margem_un / self.precos) >= params.margem_minima     # (n,K) bool
        self.retirado = ~self.viavel.any(axis=1)                                 # (n,)

        unid = np.minimum(self.estoque[:, None], np.rint(self.demanda)).astype(float)
        unid[self.retirado, :] = 0.0
        self.unidades = unid
        self.caixa = unid * self.margem_un
        restante = self.estoque[:, None] - unid
        taxa_carr = params.carreg_base + params.carreg_idade * idade / 360.0
        self.carregamento = restante * custo * taxa_carr[:, None]
        self.valor = self.caixa - self.carregamento                              # contribuição ao fitness
        self.receita = unid * self.precos
        self.concessao = unid * preco_atual * np.maximum(0.0, 1.0 - niv[None, :])
        self.concessao[self.retirado, :] = 0.0

        # teto de desconto (referência: receita a preço atual com unid. de nível 1.0)
        idx_base = int(np.argmin(np.abs(niv - 1.0)))
        self.idx_base = idx_base
        unid_ref = np.minimum(self.estoque, np.rint(base[:, 0]))
        unid_ref[self.retirado] = 0.0
        self.teto_desconto = params.teto_desconto_frac * float(
            np.sum(unid_ref * df["preco_atual"].to_numpy(float))
        )

        # níveis permitidos por SKU (piso de margem); retirado -> só o nível base
        self.permitidos = []
        for i in range(self.n):
            if self.retirado[i]:
                self.permitidos.append(np.array([idx_base]))
            else:
                self.permitidos.append(np.flatnonzero(self.viavel[i]))
        self.n_opcoes = np.array([len(a) for a in self.permitidos])
        m = int(self.n_opcoes.max())
        self.m_max = m
        # tabelas "compactadas" pelo índice de posição permitida (padding com -inf / 0)
        self.val_pos = np.full((self.n, m), -np.inf)
        self.conc_pos = np.zeros((self.n, m))
        for i, perm in enumerate(self.permitidos):
            self.val_pos[i, : len(perm)] = self.valor[i, perm]
            self.conc_pos[i, : len(perm)] = self.concessao[i, perm]

    # ---------------------------------------------------------------- fitness
    def fitness_pop(self, pop: np.ndarray) -> np.ndarray:
        """Fitness vetorizado de uma população (P, n) de posições. O(P·n)."""
        linhas = np.arange(self.n)[None, :]
        valor = self.val_pos[linhas, pop].sum(axis=1)
        conc = self.conc_pos[linhas, pop].sum(axis=1)
        excesso = np.maximum(0.0, conc - self.teto_desconto)
        return valor - PENALIDADE_DESCONTO * excesso

    def fitness(self, pos) -> float:
        return float(self.fitness_pop(np.asarray(pos)[None, :])[0])

    def fitness_ingenuo(self, pos) -> float:
        """Versão ingênua (laço Python, recalcula o modelo): usada só no benchmark
        antes/depois da otimização."""
        valor, conc = 0.0, 0.0
        for i in range(self.n):
            k = int(self.permitidos[i][pos[i]])
            preco = self.df["preco_atual"].iat[i] * self.niveis[k]
            d = self.df["demanda_base_90d"].iat[i] * self.niveis[k] ** (-self.df["elasticidade"].iat[i])
            u = 0 if self.retirado[i] else min(int(self.estoque[i]), int(round(d)))
            m = (preco * (1 - self.df["taxa_marketplace"].iat[i])
                 - self.df["custo_unitario"].iat[i] - self.df["custo_logistico"].iat[i])
            idade = self.df["idade_estoque_dias"].iat[i]
            carr = (self.estoque[i] - u) * self.df["custo_unitario"].iat[i] * (
                self.p.carreg_base + self.p.carreg_idade * idade / 360.0)
            valor += u * m - carr
            if not self.retirado[i]:
                conc += u * self.df["preco_atual"].iat[i] * max(0.0, 1.0 - self.niveis[k])
        return valor - PENALIDADE_DESCONTO * max(0.0, conc - self.teto_desconto)

    # --------------------------------------------------------------- políticas
    def niveis_da_politica(self, pos) -> np.ndarray:
        """Converte posições (compactas) em índices de nível 0..K-1."""
        return np.array([self.permitidos[i][int(pos[i])] for i in range(self.n)])

    def politica_base(self) -> np.ndarray:
        """Posições da política 'manter preço atual' (nível 1.0 ou o único permitido)."""
        pos = np.zeros(self.n, dtype=int)
        for i, perm in enumerate(self.permitidos):
            alvo = np.flatnonzero(perm == self.idx_base)
            pos[i] = alvo[0] if len(alvo) else len(perm) - 1
        return pos

    def avaliar(self, pos) -> dict:
        k = self.niveis_da_politica(pos)
        ar = np.arange(self.n)
        caixa = float(self.caixa[ar, k].sum())
        receita = float(self.receita[ar, k].sum())
        unid = self.unidades[ar, k]
        resto = self.estoque - unid
        return {
            "caixa_operacional": caixa,
            "receita": receita,
            "margem_percentual": caixa / receita if receita > 0 else 0.0,
            "unidades_vendidas": float(unid.sum()),
            "estoque_remanescente": float(resto.sum()),
            "estoque_remanescente_valor_custo": float((resto * self.df["custo_unitario"].to_numpy()).sum()),
            "custo_carregamento": float(self.carregamento[ar, k].sum()),
            "concessao_desconto": float(self.concessao[ar, k].sum()),
            "teto_desconto": float(self.teto_desconto),
            "fitness": float(self.valor[ar, k].sum()
                             - PENALIDADE_DESCONTO * max(0.0, self.concessao[ar, k].sum() - self.teto_desconto)),
            "skus_retirados": int(self.retirado.sum()),
        }

    # ------------------------------------------------------------ sub-instância
    def subconjunto(self, indices) -> "ModeloQ1":
        return ModeloQ1(self.df.iloc[list(indices)], self.p)
