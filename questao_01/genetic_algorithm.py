"""Etapa B — Algoritmo Genético para a política de preços do catálogo.

Representação: cromossomo = vetor de n genes; gene i = posição na lista de níveis
permitidos do SKU i (o piso de margem já está embutido no domínio do gene, então todo
indivíduo respeita a margem mínima por construção).

Operadores (justificativa completa em docs/decisoes.md):
  * seleção    — torneio de tamanho `torneio` (pressão seletiva ajustável, O(P·t))
  * crossover  — uniforme com probabilidade `taxa_cruzamento` (mistura SKUs independentes
                 sem quebrar blocos, adequado porque o fitness é aditivo por SKU)
  * mutação    — por gene com prob. `taxa_mutacao`: sorteia outro nível permitido
  * elitismo   — os `elite` melhores passam intactos (fitness nunca regride)
  * parada     — `geracoes` máximas OU `paciencia` gerações sem melhora do melhor fitness
Toda aleatoriedade vem de um único np.random.Generator semeado (reprodutível).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .modelo import ModeloQ1


@dataclass(frozen=True)
class ConfigGA:
    nome: str = "A"
    populacao: int = 60
    geracoes: int = 150
    taxa_cruzamento: float = 0.9
    taxa_mutacao: float = 0.03
    elite: int = 2
    torneio: int = 3
    paciencia: int = 40

    def validar(self):
        if self.populacao < 2 or self.geracoes < 1:
            raise ValueError("populacao >= 2 e geracoes >= 1")
        if not (0 <= self.taxa_mutacao <= 1 and 0 <= self.taxa_cruzamento <= 1):
            raise ValueError("taxas devem estar em [0, 1]")
        if not (0 <= self.elite < self.populacao):
            raise ValueError("elite deve estar em [0, populacao)")
        if self.torneio < 1:
            raise ValueError("torneio >= 1")


CONFIGS_PADRAO = {
    "A": ConfigGA("A", populacao=60, geracoes=150, taxa_cruzamento=0.9, taxa_mutacao=0.03,
                  elite=2, torneio=3, paciencia=40),
    "B": ConfigGA("B", populacao=160, geracoes=300, taxa_cruzamento=0.85, taxa_mutacao=0.015,
                  elite=4, torneio=5, paciencia=60),
}


def algoritmo_genetico(modelo: ModeloQ1, cfg: ConfigGA, seed: int):
    """Retorna (melhor_pos, melhor_fitness, historico) com historico = dict de listas
    {'melhor': [...], 'media': [...]} por geração (geração 0 = população inicial)."""
    cfg.validar()
    rng = np.random.default_rng(seed)
    n, P = modelo.n, cfg.populacao
    nop = modelo.n_opcoes

    pop = np.floor(rng.random((P, n)) * nop[None, :]).astype(int)
    fit = modelo.fitness_pop(pop)

    melhor_fit = float(fit.max())
    melhor_pos = pop[int(np.argmax(fit))].copy()
    hist = {"melhor": [melhor_fit], "media": [float(fit.mean())]}
    sem_melhora = 0

    for _ in range(cfg.geracoes):
        ordem = np.argsort(-fit)
        elite = pop[ordem[: cfg.elite]].copy()

        # --- seleção por torneio (vetorizada): P - elite pais por filho
        n_filhos = P - cfg.elite
        cand = rng.integers(0, P, size=(2 * n_filhos, cfg.torneio))
        vencedores = cand[np.arange(2 * n_filhos), np.argmax(fit[cand], axis=1)]
        pais_a, pais_b = pop[vencedores[:n_filhos]], pop[vencedores[n_filhos:]]

        # --- crossover uniforme (só em parte dos pares)
        faz_cruz = rng.random(n_filhos) < cfg.taxa_cruzamento
        mascara = rng.random((n_filhos, n)) < 0.5
        filhos = np.where(faz_cruz[:, None] & mascara, pais_b, pais_a)

        # --- mutação: novo nível permitido sorteado por gene
        muta = rng.random((n_filhos, n)) < cfg.taxa_mutacao
        novos = np.floor(rng.random((n_filhos, n)) * nop[None, :]).astype(int)
        filhos = np.where(muta, novos, filhos)

        pop = np.vstack([elite, filhos]) if cfg.elite else filhos
        fit = modelo.fitness_pop(pop)

        j = int(np.argmax(fit))
        if fit[j] > melhor_fit + 1e-9:
            melhor_fit, melhor_pos, sem_melhora = float(fit[j]), pop[j].copy(), 0
        else:
            sem_melhora += 1
        hist["melhor"].append(melhor_fit)
        hist["media"].append(float(fit.mean()))
        if sem_melhora >= cfg.paciencia:
            break

    return melhor_pos, melhor_fit, hist
