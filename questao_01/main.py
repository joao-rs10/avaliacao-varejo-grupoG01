"""Questão 1 — Operação Caixa 90: Brute Force -> Algoritmo Genético -> Greedy.

Fluxo (integração obrigatória):
  1. Brute Force resolve EXATAMENTE um subconjunto pequeno          -> ótimo de referência
  2. GA roda no mesmo subconjunto                                    -> gap % vs. ótimo (valida o GA)
  3. GA (2 configurações) roda no catálogo completo                  -> política de preços
  4. Greedy recebe a política do GA e distribui o orçamento promocional
  5. Relatório final (caixa, margem, estoque, cobertura da dívida, uso do orçamento)

Uso:
  python questao_01/main.py --seed 1021 --modo completo
  python questao_01/main.py --modo completo --meta-mult 1.10 --orcamento-mult 0.75   # cenário-surpresa
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

from common.cenarios import aplicar_cenario_q1
from common.dados import carregar_q1
from common.medicao import medir
from questao_01.brute_force import brute_force_vetorizado
from questao_01.genetic_algorithm import CONFIGS_PADRAO, ConfigGA, algoritmo_genetico
from questao_01.greedy import greedy_incentivos
from questao_01.modelo import ModeloQ1, ParametrosQ1
from questao_01.otimizacao import comparar_brute_force, comparar_fitness_ga


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Questão 1 — Operação Caixa 90")
    ap.add_argument("--dados", default=str(RAIZ / "data"), help="pasta com os arquivos oficiais")
    ap.add_argument("--seed", type=int, default=None, help="padrão: seed_final do config")
    ap.add_argument("--modo", choices=["rapido", "completo"], default="completo")
    ap.add_argument("--saida", default=str(RAIZ / "questao_01" / "resultados"))
    ap.add_argument("--n-bf", type=int, default=8, help="tamanho do subconjunto do Brute Force (>= 6)")
    ap.add_argument("--n-seeds-gap", type=int, default=10, help="execuções do GA p/ estimar o gap")
    # cenário-surpresa
    ap.add_argument("--meta-mult", type=float, default=1.0)
    ap.add_argument("--orcamento-mult", type=float, default=1.0)
    ap.add_argument("--custo-logistico-mult", type=float, default=1.0)
    ap.add_argument("--remover-sku", action="append", default=[])
    ap.add_argument("--margem-minima", type=float, default=None)
    # parâmetros do GA (aplicados às configurações A e B)
    ap.add_argument("--populacao", type=int, default=None)
    ap.add_argument("--mutacao", type=float, default=None)
    ap.add_argument("--geracoes", type=int, default=None)
    ap.add_argument("--sem-integridade", action="store_true", help="não conferir hashes (apenas debug)")
    return ap.parse_args(argv)


def _cfgs_ga(args):
    cfgs = {}
    for nome, c in CONFIGS_PADRAO.items():
        kw = {}
        if args.populacao:
            kw["populacao"] = args.populacao
        if args.mutacao is not None:
            kw["taxa_mutacao"] = args.mutacao
        if args.geracoes:
            kw["geracoes"] = args.geracoes
        if args.modo == "rapido" and not args.geracoes:
            kw["geracoes"] = 40
            kw["paciencia"] = 15
        cfgs[nome] = ConfigGA(**{**c.__dict__, **kw})
    return cfgs


def selecionar_subconjunto(modelo: ModeloQ1, n: int, seed: int):
    """Sorteio (seed) de n SKUs não retirados, com >= 3 níveis permitidos."""
    cand = [i for i in range(modelo.n) if (not modelo.retirado[i]) and modelo.n_opcoes[i] >= 3]
    if n < 6:
        raise ValueError("O subconjunto do Brute Force deve ter pelo menos 6 produtos.")
    if len(cand) < n:
        raise ValueError(f"Só há {len(cand)} SKUs elegíveis para o subconjunto de {n}.")
    rng = np.random.default_rng(seed + 7)
    return sorted(int(i) for i in rng.choice(cand, size=n, replace=False))


def main(argv=None):
    args = parse(argv)
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    df0, cfg0 = carregar_q1(args.dados, verificar=not args.sem_integridade)
    seed = args.seed if args.seed is not None else cfg0["seed_final"]
    df, cfg = aplicar_cenario_q1(df0, cfg0, args.meta_mult, args.orcamento_mult,
                                 args.custo_logistico_mult, args.remover_sku, args.margem_minima)
    modelo = ModeloQ1(df, ParametrosQ1.de_config(cfg))
    cfgs = _cfgs_ga(args)
    desempenho = {}

    # ---------------------------------------------------------------- A+B no subconjunto
    sub_idx = selecionar_subconjunto(modelo, args.n_bf, seed)
    sub = modelo.subconjunto(sub_idx)
    (pos_bf, f_bf, total_comb), t_bf, m_bf = medir(brute_force_vetorizado, sub)
    desempenho["brute_force"] = {"n_skus": sub.n, "combinacoes": total_comb, "tempo_s": t_bf, "mem_mib": m_bf}

    ga_sub_runs = []
    n_runs = 1 if args.modo == "rapido" else args.n_seeds_gap
    for r in range(n_runs):
        _, f_ga, h = algoritmo_genetico(sub, cfgs["A"], seed + r)
        ga_sub_runs.append({"seed": seed + r, "fitness_ga": f_ga, "gap_pct": max(0.0, 100 * (f_bf - f_ga) / abs(f_bf)),
                            "geracoes_executadas": len(h["melhor"]) - 1})
    (_, f_ga0, _), t_ga0, m_ga0 = medir(algoritmo_genetico, sub, cfgs["A"], seed)
    desempenho["ga_subconjunto"] = {"tempo_s": t_ga0, "mem_mib": m_ga0}
    tab_bf = pd.DataFrame([{"metodo": "Brute Force (exato)", "fitness": f_bf, "gap_pct": 0.0,
                            "tempo_s": t_bf, "combinacoes_avaliadas": total_comb}] +
                          [{"metodo": f"GA config A (seed {x['seed']})", "fitness": x["fitness_ga"],
                            "gap_pct": x["gap_pct"], "tempo_s": t_ga0 if k == 0 else np.nan,
                            "combinacoes_avaliadas": cfgs['A'].populacao * (x["geracoes_executadas"] + 1)}
                           for k, x in enumerate(ga_sub_runs)])
    tab_bf.to_csv(saida / "tabela_bf_vs_ga.csv", index=False, encoding="utf-8")
    gaps = [x["gap_pct"] for x in ga_sub_runs]

    # ---------------------------------------------------------------- B no catálogo completo
    res_ga, hist_cfg = {}, {}
    comp_rows = []
    for nome, c in cfgs.items():
        (pos, f, hist), t, mem = medir(algoritmo_genetico, modelo, c, seed)
        res_ga[nome] = (pos, f)
        hist_cfg[nome] = hist
        desempenho[f"ga_completo_{nome}"] = {"tempo_s": t, "mem_mib": mem}
        ger_melhor = int(np.argmax(np.array(hist["melhor"]) >= max(hist["melhor"]) - 1e-9))
        comp_rows.append({"config": nome, "populacao": c.populacao, "geracoes_max": c.geracoes,
                          "taxa_mutacao": c.taxa_mutacao, "elite": c.elite, "torneio": c.torneio,
                          "taxa_cruzamento": c.taxa_cruzamento, "paciencia": c.paciencia,
                          "geracoes_executadas": len(hist["melhor"]) - 1,
                          "geracao_do_melhor": ger_melhor, "fitness_final": f,
                          "tempo_s": t, "mem_pico_mib": mem})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(saida / "ga_comparacao_configs.csv", index=False, encoding="utf-8")

    melhor_cfg = max(res_ga, key=lambda k: (round(res_ga[k][1], 6), -comp.set_index("config").loc[k, "tempo_s"]))
    pos_ga, f_ga_full = res_ga[melhor_cfg]

    # gráfico fitness por geração
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome, h in hist_cfg.items():
        l, = ax.plot(h["melhor"], label=f"config {nome} — melhor", lw=2)
        ax.plot(h["media"], ls="--", lw=1, color=l.get_color(), alpha=0.6, label=f"config {nome} — média")
    ax.set_xlabel("geração"); ax.set_ylabel("fitness (R$)")
    ax.set_title(f"GA no catálogo completo — fitness por geração (seed {seed})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(saida / "ga_fitness_por_geracao.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- C: Greedy sobre a política do GA
    g, t_g, m_g = medir(greedy_incentivos, modelo, pos_ga)
    desempenho["greedy"] = {"tempo_s": t_g, "mem_mib": m_g}
    ar = np.arange(modelo.n)
    k_ga = modelo.niveis_da_politica(pos_ga)
    preco_ga = modelo.precos[ar, k_ga]
    caixa_ga_sku = modelo.caixa[ar, k_ga]
    antes_depois = pd.DataFrame({
        "sku": modelo.df["sku"], "categoria": modelo.df["categoria"],
        "nivel_preco": modelo.niveis[k_ga], "preco_final": preco_ga.round(2),
        "retirado_do_canal": modelo.retirado,
        "unidades_antes": modelo.unidades[ar, k_ga], "caixa_antes": caixa_ga_sku.round(2),
        "investimento_greedy": g["investimento"].round(2),
        "unidades_extra": g["unidades_extra"].round(2),
        "ganho_bruto_greedy": g["ganho_bruto"].round(2),
        "caixa_depois_liquido": (caixa_ga_sku + g["ganho_bruto"] - g["investimento"]).round(2),
    })
    antes_depois.to_csv(saida / "tabela_antes_depois_greedy.csv", index=False, encoding="utf-8")

    # ---------------------------------------------------------------- resultado final
    base = modelo.avaliar(modelo.politica_base())
    ga_av = modelo.avaliar(pos_ga)
    invest = float(g["investimento"].sum())
    caixa_final = ga_av["caixa_operacional"] + float(g["ganho_bruto"].sum()) - invest
    receita_final = ga_av["receita"] + float((g["unidades_extra"] * preco_ga).sum())
    estoque_final = ga_av["estoque_remanescente"] - float(g["unidades_extra"].sum())
    meta, divida = modelo.p.meta_caixa, modelo.p.divida_total
    orc = modelo.p.orcamento_promocional
    final = {
        "grupo": cfg["grupo"], "seed_usada": seed, "n_skus": modelo.n,
        "cenario": {"meta_mult": args.meta_mult, "orcamento_mult": args.orcamento_mult,
                    "custo_logistico_mult": args.custo_logistico_mult, "remover_sku": args.remover_sku},
        "config_ga_escolhida": melhor_cfg,
        "caixa_politica_preco_atual": base["caixa_operacional"],
        "caixa_apos_ga": ga_av["caixa_operacional"],
        "caixa_final_liquido_pos_greedy": caixa_final,
        "receita_final": receita_final,
        "margem_final_percentual": caixa_final / receita_final if receita_final else 0.0,
        "estoque_remanescente_unidades": estoque_final,
        "meta_caixa_90d": meta,
        "meta_atingida": bool(caixa_final >= meta),
        "folga_ou_insuficiencia": caixa_final - meta,
        "divida_total_90d": divida,
        "cobertura_da_divida": caixa_final / divida if divida else None,
        "divida_coberta": bool(caixa_final >= divida),
        "orcamento_promocional": orc, "orcamento_utilizado": invest,
        "uso_do_orcamento_pct": 100 * invest / orc if orc else 0.0,
        "ganho_bruto_incentivos": float(g["ganho_bruto"].sum()),
        "retorno_por_real_investido": float(g["ganho_bruto"].sum()) / invest if invest else None,
        "skus_retirados_do_canal": int(modelo.retirado.sum()),
        "concessao_de_desconto": ga_av["concessao_desconto"], "teto_de_desconto": ga_av["teto_desconto"],
        "validacao_ga_vs_bf": {"n_skus_subconjunto": sub.n, "skus": [modelo.df['sku'].iat[i] for i in sub_idx],
                               "fitness_bf": f_bf, "gap_pct_medio": float(np.mean(gaps)),
                               "gap_pct_maximo": float(np.max(gaps)), "execucoes_ga": len(gaps)},
    }
    (saida / "resultado_final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- otimização antes/depois
    if args.modo == "completo":
        sub6 = modelo.subconjunto(selecionar_subconjunto(modelo, 6, seed))
        otim = {"brute_force_n6": comparar_brute_force(sub6),
                "fitness_ga_populacao": comparar_fitness_ga(modelo, seed=seed)}
        (saida / "otimizacao_antes_depois.json").write_text(json.dumps(otim, ensure_ascii=False, indent=2), encoding="utf-8")

    (saida / "desempenho.json").write_text(json.dumps(desempenho, ensure_ascii=False, indent=2), encoding="utf-8")

    # tabela comparativa final (base x GA x GA+Greedy)
    comparativo = pd.DataFrame([
        {"politica": "preço atual (sem ação)", "caixa": base["caixa_operacional"], "margem_%": 100 * base["margem_percentual"],
         "estoque_remanescente": base["estoque_remanescente"], "investimento": 0.0},
        {"politica": f"GA (config {melhor_cfg})", "caixa": ga_av["caixa_operacional"], "margem_%": 100 * ga_av["margem_percentual"],
         "estoque_remanescente": ga_av["estoque_remanescente"], "investimento": 0.0},
        {"politica": "GA + Greedy (líquido)", "caixa": caixa_final, "margem_%": 100 * final["margem_final_percentual"],
         "estoque_remanescente": estoque_final, "investimento": invest},
    ])
    comparativo.to_csv(saida / "tabela_comparativa_politicas.csv", index=False, encoding="utf-8")

    print("=" * 72)
    print(f"Questão 1 | grupo {cfg['grupo']} | seed {seed} | {modelo.n} SKUs")
    print(f"Brute Force n={sub.n} ({total_comb:,} combinações): fitness {f_bf:,.2f} em {t_bf:.3f}s")
    print(f"GA vs BF: gap médio {np.mean(gaps):.4f}% (máx {np.max(gaps):.4f}%) em {len(gaps)} execuções")
    print(comp[["config", "populacao", "geracoes_executadas", "fitness_final", "tempo_s", "mem_pico_mib"]].to_string(index=False))
    print(f"Caixa: preço atual {base['caixa_operacional']:,.2f} -> GA {ga_av['caixa_operacional']:,.2f} "
          f"-> GA+Greedy {caixa_final:,.2f}")
    print(f"Meta {meta:,.2f}: {'ATINGIDA' if final['meta_atingida'] else 'NÃO atingida'} "
          f"(folga/insuficiência {final['folga_ou_insuficiencia']:,.2f}) | "
          f"cobertura da dívida {100 * caixa_final / divida:.1f}%")
    print(f"Orçamento promocional usado: {invest:,.2f} de {orc:,.2f} ({final['uso_do_orcamento_pct']:.1f}%)")
    print(f"Resultados em: {saida}")
    return final


if __name__ == "__main__":
    main()
