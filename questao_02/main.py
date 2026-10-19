"""Questão 2 — Plano de Sobrevivência em 12 Semanas: Monte Carlo -> DP -> Greedy -> Monte Carlo.

Uso:
  python questao_02/main.py --seed 1021 --cenarios 5000
  python questao_02/main.py --cenarios 5000 --preco-concorrente-mult 0.92 --custo-logistico-mult 1.15   # surpresa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.cenarios import aplicar_cenario_q2
from common.dados import carregar_q2
from common.medicao import medir
from questao_02.escalabilidade import estudo_escalabilidade
from questao_02.modelo import ModeloQ2, ParametrosQ2
from questao_02.monte_carlo import (TAG_VALIDACAO, PlanoPromo, estatisticas_por_preco, plano_sem_promo,
                                    politica_constante, resumir, simular_plano)
from questao_02.dynamic_programming import resolver_dp
from questao_02.greedy import greedy_semanal
from questao_02.pipeline import construir_plano, validar_plano


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Questão 2 — Plano de Sobrevivência em 12 Semanas")
    ap.add_argument("--dados", default=str(RAIZ / "data"))
    ap.add_argument("--seed", type=int, default=None, help="padrão: seed_final do config")
    ap.add_argument("--cenarios", type=int, default=5000, help="cenários da VALIDAÇÃO (>= 5000)")
    ap.add_argument("--cenarios-construcao", type=int, default=2000, help="cenários da construção (>= 2000)")
    ap.add_argument("--percentil-risco", type=float, default=5.0)
    ap.add_argument("--saida", default=str(RAIZ / "questao_02" / "resultados"))
    ap.add_argument("--sem-escalabilidade", action="store_true")
    ap.add_argument("--sem-estresse", action="store_true")
    # cenário-surpresa
    ap.add_argument("--preco-concorrente-mult", type=float, default=1.0)
    ap.add_argument("--custo-logistico-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--divida-mult", type=float, default=1.0)
    ap.add_argument("--remover-sku", action="append", default=[])
    ap.add_argument("--sem-integridade", action="store_true")
    return ap.parse_args(argv)


def escolher_representativos(modelo: ModeloQ2):
    capital = modelo.estoque0 * modelo.preco_atual
    cobertura = modelo.estoque0 / (modelo.base * modelo.T)
    return sorted({int(np.argmax(capital)), int(np.argmin(cobertura)), int(np.argmax(cobertura))})


def main(argv=None):
    args = parse(argv)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    df0, cfg0 = carregar_q2(args.dados, verificar=not args.sem_integridade)
    seed = args.seed if args.seed is not None else cfg0["seed_final"]
    df, cfg = aplicar_cenario_q2(df0, cfg0, args.preco_concorrente_mult, args.custo_logistico_mult,
                                 args.orcamento_mult, args.divida_mult, args.remover_sku)
    params = ParametrosQ2.de_config(cfg, percentil_risco=args.percentil_risco)
    modelo = ModeloQ2(df, params)
    desemp = {}

    # --- plano completo (MC -> DP -> Greedy -> MC de calibração) com medição das etapas
    plano, t_plano, m_plano = medir(construir_plano, modelo, seed, args.cenarios_construcao)
    desemp["construir_plano_total"] = {"tempo_s": t_plano, "mem_mib": m_plano}
    stats, t, m = medir(estatisticas_por_preco, modelo, args.cenarios_construcao, seed)
    desemp["monte_carlo_estatisticas"] = {"tempo_s": t, "mem_mib": m}
    _, t, m = medir(resolver_dp, modelo, plano.stats, plano.pesos)
    desemp["dynamic_programming"] = {"tempo_s": t, "mem_mib": m}
    _, t, m = medir(greedy_semanal, modelo, plano.trajetoria, plano.stats, plano.dp, plano.pesos)
    desemp["greedy"] = {"tempo_s": t, "mem_mib": m}

    # --- validação independente
    reps = escolher_representativos(modelo)
    res, t, m = medir(validar_plano, modelo, plano, seed, args.cenarios, reps)
    desemp["validacao_monte_carlo"] = {"tempo_s": t, "mem_mib": m, "n_cenarios": args.cenarios}
    resumo = resumir(res, modelo)

    # --- comparativo de políticas (mesmos cenários independentes)
    def validar_politica(nome, pol, promo):
        r = simular_plano(modelo, pol, promo, args.cenarios, seed, TAG_VALIDACAO)
        s = resumir(r, modelo)
        return {"politica": nome, **{k: v for k, v in s.items() if not isinstance(v, dict)},
                **{f"P_{k}": v for k, v in s["prob_pagamento_por_vencimento"].items()}}
    k_base = int(np.argmin(np.abs(modelo.niveis - 1.0)))
    comparativo = pd.DataFrame([
        validar_politica("preço fixo atual (sem DP, sem Greedy)", politica_constante(modelo, k_base), plano_sem_promo(modelo)),
        validar_politica("DP (sem Greedy)", plano.dp.politica, plano_sem_promo(modelo)),
        validar_politica("DP + Greedy (plano completo)", plano.dp.politica, plano.promo),
    ])
    comparativo.to_csv(saida / "comparativo_politicas.csv", index=False, encoding="utf-8")

    # --- teste de estresse: dívidas multiplicadas, plano reconstruído a cada vez
    estresse = None
    if not args.sem_estresse:
        linhas = []
        for dm in (1.0, 1.5, 2.0, 2.5, 3.0):
            mod_e = ModeloQ2(df, ParametrosQ2(**{**params.__dict__,
                             "dividas": tuple((s, v * dm) for s, v in params.dividas)}))
            pl_e = construir_plano(mod_e, seed, args.cenarios_construcao)
            r_e = resumir(validar_plano(mod_e, pl_e, seed, args.cenarios), mod_e)
            r_b = resumir(simular_plano(mod_e, politica_constante(mod_e, k_base), plano_sem_promo(mod_e),
                                        args.cenarios, seed, TAG_VALIDACAO), mod_e)
            linhas.append({"multiplicador_dividas": dm,
                           "P_todos_plano": r_e["prob_pagar_todos_no_prazo"],
                           "P_todos_preco_fixo": r_b["prob_pagar_todos_no_prazo"],
                           "iteracoes_calibracao": len(pl_e.historico),
                           "caixa_final_esperado_plano": r_e["caixa_final_esperado"],
                           "caixa_final_esperado_preco_fixo": r_b["caixa_final_esperado"]})
        estresse = pd.DataFrame(linhas)
        estresse.to_csv(saida / "teste_estresse_dividas.csv", index=False, encoding="utf-8")

    # --- escalabilidade
    if not args.sem_escalabilidade:
        esc = estudo_escalabilidade(modelo, seed)
        esc.to_csv(saida / "escalabilidade.csv", index=False, encoding="utf-8")

    # --- arquivos de evidência
    traj = plano.trajetoria.copy()
    traj["estoque_fim"] = traj["estoque_inicio"] - traj["vendas_esperadas"]
    traj.drop(columns=["i", "k"]).round(3).to_csv(saida / "trajetoria_precos_dp.csv", index=False, encoding="utf-8")
    pd.DataFrame(plano.promo.gasto, columns=modelo.df["sku"], index=np.arange(1, modelo.T + 1)
                 ).rename_axis("semana").round(2).to_csv(saida / "plano_promocional_greedy.csv", encoding="utf-8")
    pd.DataFrame(plano.historico).to_csv(saida / "calibracao_pesos.csv", index=False, encoding="utf-8")

    # gráfico 1: distribuição do caixa final
    cf = res["caixa_final"]
    pr_key = f"caixa_final_p{params.percentil_risco:g}"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cf, bins=50, color="#3b6ea5", alpha=0.85)
    ax.axvline(resumo["caixa_final_esperado"], color="k", lw=2, label=f"média {resumo['caixa_final_esperado']:,.0f}")
    ax.axvline(resumo[pr_key], color="#c0392b", lw=2, ls="--", label=f"P{params.percentil_risco:g} {resumo[pr_key]:,.0f}")
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("caixa final após as 3 dívidas (R$)"); ax.set_ylabel("nº de cenários")
    ax.set_title(f"Distribuição do caixa final — {args.cenarios} cenários independentes")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(saida / "distribuicao_caixa_final.png", dpi=140); plt.close(fig)

    # gráfico 2: trajetórias de produtos representativos
    reg = res["registro"]
    fig, axes = plt.subplots(2, len(reps), figsize=(5 * len(reps), 6.5), squeeze=False)
    semanas = np.arange(0, modelo.T + 1)
    for c, i in enumerate(reps):
        g = traj[traj.sku == modelo.df["sku"].iat[i]]
        axes[0, c].step(g["semana"], g["preco"], where="post", color="#c0392b", lw=2, label="plano DP (esperado)")
        axes[0, c].set_title(f"{modelo.df['sku'].iat[i]} — preço (R$)")
        axes[0, c].set_xlabel("semana"); axes[0, c].grid(alpha=0.3); axes[0, c].legend()
        est = reg["estoque"][:, :, c]
        axes[1, c].fill_between(semanas, np.percentile(est, 10, axis=0), np.percentile(est, 90, axis=0),
                                color="#3b6ea5", alpha=0.25, label="P10–P90 (simulação)")
        axes[1, c].plot(semanas, est.mean(axis=0), color="#3b6ea5", lw=2, label="média (simulação)")
        axes[1, c].plot(np.r_[0, g["semana"]], np.r_[g["estoque_inicio"].iat[0], g["estoque_fim"]], "k--", lw=1.5, label="plano DP (esperado)")
        axes[1, c].set_title("estoque (un.)"); axes[1, c].set_xlabel("semana"); axes[1, c].grid(alpha=0.3); axes[1, c].legend()
    fig.tight_layout(); fig.savefig(saida / "trajetoria_preco_estoque.png", dpi=140); plt.close(fig)

    # gráfico 3: probabilidade por vencimento
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(resumo["prob_pagamento_por_vencimento"].keys())
    vs = [100 * resumo["prob_pagamento_por_vencimento"][k] for k in ks]
    ax.bar([k.replace("_", " ") for k in ks] + ["todas"], vs + [100 * resumo["prob_pagar_todos_no_prazo"]], color="#3b6ea5")
    ax.axhline(100 * (1 - params.limite_risco_falha), color="#c0392b", ls="--", label="meta (1 - limite de risco)")
    ax.set_ylim(0, 105); ax.set_ylabel("P(pagar no prazo) %"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(saida / "probabilidade_pagamento.png", dpi=140); plt.close(fig)

    # --- resultado final
    gasto_total = float(plano.promo.gasto.sum())
    final = {
        "grupo": cfg["grupo"], "seed_usada": seed, "n_skus": modelo.I,
        "cenario": {"preco_concorrente_mult": args.preco_concorrente_mult,
                    "custo_logistico_mult": args.custo_logistico_mult,
                    "orcamento_mult": args.orcamento_mult, "divida_mult": args.divida_mult,
                    "remover_sku": args.remover_sku},
        "cenarios_construcao": args.cenarios_construcao, "cenarios_validacao": args.cenarios,
        "dividas": {f"semana_{s}": v for s, v in params.dividas}, "caixa_inicial": params.caixa_inicial,
        "validacao": resumo,
        "percentil_de_risco_escolhido": params.percentil_risco,
        "calibracao_pesos_urgencia": plano.historico,
        "pesos_finais": [float(x) for x in plano.pesos],
        "dp": {"subproblemas_memoizados": plano.dp.n_subproblemas, "consultas_a_V": plano.dp.n_consultas_V,
               "reuso_medio_por_subproblema": plano.dp.n_consultas_V / plano.dp.n_subproblemas},
        "promocao": {"gasto_total": gasto_total, "orcamento_total": params.orcamento_semanal * modelo.T,
                     "uso_pct": 100 * gasto_total / (params.orcamento_semanal * modelo.T) if params.orcamento_semanal else 0.0},
        "skus_representativos": [modelo.df["sku"].iat[i] for i in reps],
    }
    (saida / "resultado_final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "desempenho.json").write_text(json.dumps(desemp, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"Questão 2 | grupo {cfg['grupo']} | seed {seed} | {modelo.I} SKUs | validação {args.cenarios} cenários")
    for k, v in resumo["prob_pagamento_por_vencimento"].items():
        print(f"  P(pagar {k.replace('_', ' ')}) = {100 * v:.2f}%")
    print(f"  P(pagar as 3) = {100 * resumo['prob_pagar_todos_no_prazo']:.2f}% | cenários de falha: {resumo['cenarios_de_falha']}")
    print(f"  caixa final esperado {resumo['caixa_final_esperado']:,.0f} | P{params.percentil_risco:g} {resumo[pr_key]:,.0f} "
          f"| margem {100 * resumo['margem_esperada']:.1f}% | estoque final {resumo['estoque_final_medio_un']:.0f} un.")
    print(comparativo[["politica", "prob_pagar_todos_no_prazo", "caixa_final_esperado", "margem_esperada"]].to_string(index=False))
    if estresse is not None:
        print(estresse.to_string(index=False))
    print(f"Resultados em: {saida}")
    return final


if __name__ == "__main__":
    main()
