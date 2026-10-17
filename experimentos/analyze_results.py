"""Calcula estabilidade, estatística, tabelas e figuras dos experimentos."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.experimento import (  # noqa: E402
    carregar_shap,
    id_execucao,
    metricas_estabilidade,
)

METRICAS_PREDITIVAS = ["macro_f1", "pr_auc", "balanced_accuracy", "roc_auc"]
METRICAS_SHAP = [
    "shap_global_spearman",
    "shap_global_jaccard_top5",
    "shap_global_l1_normalized",
    "shap_local_cosine_mean",
    "shap_local_sign_agreement_mean",
]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["piloto", "completo"])
    return parser.parse_args()


def bootstrap_media(dados: np.ndarray, seed: int = 20260929) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    amostras = rng.choice(dados, size=(10_000, len(dados)), replace=True)
    medias = amostras.mean(axis=1)
    return tuple(float(x) for x in np.quantile(medias, [0.025, 0.975]))


def corrigir_holm(pvalores: list[float]) -> list[float]:
    ordem = np.argsort(pvalores)
    ajustados = np.empty(len(pvalores), dtype=float)
    acumulado = 0.0
    m = len(pvalores)
    for posicao, indice in enumerate(ordem):
        valor = min(1.0, (m - posicao) * pvalores[indice])
        acumulado = max(acumulado, valor)
        ajustados[indice] = acumulado
    return ajustados.tolist()


def rank_biserial(diferencas: np.ndarray) -> float:
    diferencas = diferencas[diferencas != 0]
    if len(diferencas) == 0:
        return 0.0
    ranks = rankdata(np.abs(diferencas))
    positivo = ranks[diferencas > 0].sum()
    negativo = ranks[diferencas < 0].sum()
    return float((positivo - negativo) / ranks.sum())


def adicionar_estabilidade(tabela: pd.DataFrame, pasta: Path) -> pd.DataFrame:
    novas = []
    for _, linha in tabela.iterrows():
        if linha["degradacao"] == "baseline":
            novas.append({})
            continue
        baseline_id = id_execucao(
            linha["modelo"], int(linha["semente"]), "baseline", 0.0
        )
        ref_local, ref_global = carregar_shap(pasta / "shap" / f"{baseline_id}.npz")
        cmp_local, cmp_global = carregar_shap(
            pasta / "shap" / f"{linha['run_id']}.npz"
        )
        novas.append(
            metricas_estabilidade(ref_local, ref_global, cmp_local, cmp_global)
        )
    return pd.concat([tabela.reset_index(drop=True), pd.DataFrame(novas)], axis=1)


def variabilidade_baseline(tabela: pd.DataFrame, pasta: Path) -> pd.DataFrame:
    linhas = []
    baselines = tabela[tabela["degradacao"] == "baseline"]
    for modelo, grupo in baselines.groupby("modelo"):
        for (_, a), (_, b) in itertools.combinations(grupo.iterrows(), 2):
            a_local, a_global = carregar_shap(pasta / "shap" / f"{a['run_id']}.npz")
            b_local, b_global = carregar_shap(pasta / "shap" / f"{b['run_id']}.npz")
            linhas.append(
                {
                    "modelo": modelo,
                    "semente_a": int(a["semente"]),
                    "semente_b": int(b["semente"]),
                    **metricas_estabilidade(a_local, a_global, b_local, b_global),
                }
            )
    return pd.DataFrame(linhas)


def resumo_descritivo(tabela: pd.DataFrame) -> pd.DataFrame:
    metricas = METRICAS_PREDITIVAS + METRICAS_SHAP
    linhas = []
    degradados = tabela[tabela["degradacao"] != "baseline"]
    for chaves, grupo in degradados.groupby(["modelo", "degradacao", "nivel"]):
        for metrica in metricas:
            valores = grupo[metrica].dropna().to_numpy(float)
            if not len(valores):
                continue
            baixo, alto = bootstrap_media(valores)
            linhas.append(
                {
                    "modelo": chaves[0],
                    "degradacao": chaves[1],
                    "nivel": chaves[2],
                    "metrica": metrica,
                    "n": len(valores),
                    "media": valores.mean(),
                    "mediana": np.median(valores),
                    "desvio_padrao": valores.std(ddof=1) if len(valores) > 1 else 0,
                    "ic95_media_inferior": baixo,
                    "ic95_media_superior": alto,
                }
            )
    return pd.DataFrame(linhas)


def tabela_resultados_principais(tabela: pd.DataFrame) -> pd.DataFrame:
    base = tabela[tabela["degradacao"] == "baseline"].set_index(["modelo", "semente"])
    degradados = tabela[tabela["degradacao"] != "baseline"].copy()
    degradados["delta_macro_f1"] = degradados.apply(
        lambda r: r["macro_f1"] - base.loc[(r["modelo"], r["semente"]), "macro_f1"],
        axis=1,
    )
    linhas = []
    for (modelo, degradacao, nivel), grupo in degradados.groupby(
        ["modelo", "degradacao", "nivel"]
    ):
        delta = grupo["delta_macro_f1"].to_numpy(float)
        baixo, alto = bootstrap_media(delta)
        linhas.append(
            {
                "modelo": modelo,
                "degradacao": degradacao,
                "nivel": nivel,
                "n_sementes": len(grupo),
                "macro_f1_media": grupo["macro_f1"].mean(),
                "delta_macro_f1_medio": delta.mean(),
                "delta_macro_f1_mediano": np.median(delta),
                "delta_macro_f1_ic95_inferior": baixo,
                "delta_macro_f1_ic95_superior": alto,
                "pr_auc_media": grupo["pr_auc"].mean(),
                "shap_spearman_global_medio": grupo["shap_global_spearman"].mean(),
                "shap_jaccard_top5_medio": grupo["shap_global_jaccard_top5"].mean(),
                "shap_l1_global_media": grupo["shap_global_l1_normalized"].mean(),
                "shap_cosseno_local_medio": grupo["shap_local_cosine_mean"].mean(),
                "shap_concordancia_sinal_media": grupo[
                    "shap_local_sign_agreement_mean"
                ].mean(),
            }
        )
    return pd.DataFrame(linhas).sort_values(["modelo", "degradacao", "nivel"])


def testes_estatisticos(tabela: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for modelo in sorted(tabela["modelo"].unique()):
        base = tabela[
            (tabela["modelo"] == modelo) & (tabela["degradacao"] == "baseline")
        ].set_index("semente")
        for degradacao in sorted(
            tabela.loc[tabela["degradacao"] != "baseline", "degradacao"].unique()
        ):
            bloco = tabela[
                (tabela["modelo"] == modelo)
                & (tabela["degradacao"] == degradacao)
            ]
            for metrica in METRICAS_PREDITIVAS + METRICAS_SHAP:
                colunas = []
                sementes_comuns = set(base.index)
                for nivel in (0.05, 0.10, 0.20, 0.30):
                    serie = bloco[bloco["nivel"] == nivel].set_index("semente")[metrica]
                    sementes_comuns &= set(serie.dropna().index)
                    colunas.append(serie)
                sementes = sorted(sementes_comuns)
                if len(sementes) < 3:
                    continue
                base_val = (
                    np.ones(len(sementes))
                    if metrica in METRICAS_SHAP
                    and metrica != "shap_global_l1_normalized"
                    else np.zeros(len(sementes))
                    if metrica == "shap_global_l1_normalized"
                    else base.loc[sementes, metrica].to_numpy(float)
                )
                niveis_val = [serie.loc[sementes].to_numpy(float) for serie in colunas]
                try:
                    friedman = friedmanchisquare(base_val, *niveis_val)
                    friedman_stat = float(friedman.statistic)
                    friedman_p = float(friedman.pvalue)
                except ValueError:
                    friedman_stat, friedman_p = 0.0, 1.0
                p_brutos = []
                temporarias = []
                for nivel, valores in zip((0.05, 0.10, 0.20, 0.30), niveis_val, strict=True):
                    diferencas = valores - base_val
                    nao_zero = diferencas[diferencas != 0]
                    executar_posthoc = friedman_p < 0.05
                    if not executar_posthoc:
                        estatistica, pvalor = np.nan, np.nan
                    elif len(nao_zero) == 0:
                        estatistica, pvalor = 0.0, 1.0
                    else:
                        teste = wilcoxon(nao_zero, method="exact", alternative="two-sided")
                        estatistica, pvalor = float(teste.statistic), float(teste.pvalue)
                    baixo, alto = bootstrap_media(diferencas)
                    if executar_posthoc:
                        p_brutos.append(pvalor)
                    temporarias.append(
                        {
                            "modelo": modelo,
                            "degradacao": degradacao,
                            "metrica": metrica,
                            "nivel": nivel,
                            "n": len(sementes),
                            "friedman_statistic": friedman_stat,
                            "friedman_p": friedman_p,
                            "posthoc_performed": executar_posthoc,
                            "wilcoxon_statistic": estatistica,
                            "wilcoxon_p": pvalor,
                            "diferenca_media": float(diferencas.mean()),
                            "diferenca_mediana": float(np.median(diferencas)),
                            "ic95_diferenca_media_inferior": baixo,
                            "ic95_diferenca_media_superior": alto,
                            "rank_biserial": rank_biserial(diferencas),
                        }
                    )
                p_ajustados = corrigir_holm(p_brutos) if p_brutos else []
                indice_ajustado = 0
                for linha in temporarias:
                    if linha["posthoc_performed"]:
                        ajustado = p_ajustados[indice_ajustado]
                        indice_ajustado += 1
                        linha["wilcoxon_p_holm"] = ajustado
                        linha["significativo_005"] = bool(ajustado < 0.05)
                    else:
                        linha["wilcoxon_p_holm"] = np.nan
                        linha["significativo_005"] = False
                    linhas.append(linha)
    return pd.DataFrame(linhas)


def associacoes_exploratorias(tabela: pd.DataFrame) -> pd.DataFrame:
    base = tabela[tabela["degradacao"] == "baseline"].set_index(["modelo", "semente"])
    degradados = tabela[tabela["degradacao"] != "baseline"].copy()
    degradados["delta_macro_f1"] = degradados.apply(
        lambda r: r["macro_f1"] - base.loc[(r["modelo"], r["semente"]), "macro_f1"],
        axis=1,
    )
    linhas = []
    for (modelo, degradacao), grupo in degradados.groupby(["modelo", "degradacao"]):
        for metrica in METRICAS_SHAP:
            valido = grupo[["delta_macro_f1", metrica]].dropna()
            teste = spearmanr(valido["delta_macro_f1"], valido[metrica])
            linhas.append(
                {
                    "modelo": modelo,
                    "degradacao": degradacao,
                    "metrica_shap": metrica,
                    "n": len(valido),
                    "spearman_rho": float(teste.statistic),
                    "p_nao_corrigido": float(teste.pvalue),
                    "interpretacao": "exploratoria; nao implica causalidade",
                }
            )
    return pd.DataFrame(linhas)


def avaliar_hipoteses(tabela: pd.DataFrame, variabilidade: pd.DataFrame) -> dict:
    base = tabela[tabela["degradacao"] == "baseline"].set_index(["modelo", "semente"])
    degradados = tabela[tabela["degradacao"] != "baseline"].copy()
    degradados["delta_macro_f1"] = degradados.apply(
        lambda r: r["macro_f1"] - base.loc[(r["modelo"], r["semente"]), "macro_f1"],
        axis=1,
    )
    h1_grupos = (
        degradados[
            (degradados["degradacao"] == "label_noise")
            & (degradados["nivel"] == 0.30)
        ]
        .groupby("modelo")["delta_macro_f1"]
        .mean()
    )
    h1_modelos = h1_grupos[h1_grupos <= -0.05].index.tolist()
    cenarios_medios = (
        degradados.groupby(["modelo", "degradacao", "nivel"], as_index=False)
        .agg(
            delta_macro_f1=("delta_macro_f1", "mean"),
            shap_global_spearman=("shap_global_spearman", "mean"),
        )
    )
    h2 = cenarios_medios[
        (cenarios_medios["nivel"].isin([0.20, 0.30]))
        & (cenarios_medios["delta_macro_f1"].abs() <= 0.02)
        & (cenarios_medios["shap_global_spearman"] <= 0.80)
    ]
    limites_l1 = (
        variabilidade.groupby("modelo")["shap_global_l1_normalized"]
        .quantile(0.95)
        .to_dict()
        if not variabilidade.empty
        else {}
    )
    h3_detalhes = []
    h3_medias = (
        degradados[degradados["nivel"] == 0.30]
        .groupby(["modelo", "degradacao"], as_index=False)["shap_global_l1_normalized"]
        .mean()
    )
    for _, r in h3_medias.iterrows():
        limite = limites_l1.get(r["modelo"], np.nan)
        if pd.notna(limite) and r["shap_global_l1_normalized"] > limite:
            h3_detalhes.append(
                {
                    "modelo": r["modelo"],
                    "degradacao": r["degradacao"],
                    "mean_shap_global_l1_normalized": r["shap_global_l1_normalized"],
                    "baseline_p95": limite,
                }
            )
    h3_tipos = sorted({item["degradacao"] for item in h3_detalhes})
    return {
        "H1": {
            "confirmed": len(h1_modelos) >= 2,
            "models_meeting_threshold": h1_modelos,
            "mean_delta_by_model": h1_grupos.to_dict(),
        },
        "H2": {
            "confirmed": not h2.empty,
            "matching_scenarios": h2[
                ["modelo", "degradacao", "nivel", "delta_macro_f1", "shap_global_spearman"]
            ].to_dict("records"),
        },
        "H3": {
            "confirmed": len(h3_tipos) >= 2,
            "degradation_types_meeting_threshold": h3_tipos,
            "matching_runs": h3_detalhes,
            "baseline_l1_p95_by_model": limites_l1,
        },
    }


def criar_figuras(tabela: pd.DataFrame, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    degradados = tabela[tabela["degradacao"] != "baseline"].copy()
    degradados["nivel_percentual"] = (degradados["nivel"] * 100).astype(int)
    degradados["Nível (%)"] = degradados["nivel_percentual"]
    degradados["Degradação"] = degradados["degradacao"].map(
        {
            "missing_values": "Valores ausentes",
            "attribute_errors": "Erros em atributos",
            "label_noise": "Ruído em rótulos",
        }
    )
    degradados["Modelo"] = degradados["modelo"].map(
        {
            "logistic_regression": "Regressão Logística",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        }
    )
    paleta = {
        "Valores ausentes": "#3568A8",
        "Erros em atributos": "#D38B2C",
        "Ruído em rótulos": "#8A5AA8",
    }

    for metrica, rotulo in [
        ("macro_f1", "Macro-F1"),
        ("pr_auc", "PR-AUC"),
        ("shap_global_spearman", "Spearman SHAP global"),
        ("shap_local_cosine_mean", "Cosseno SHAP local médio"),
    ]:
        grade = sns.relplot(
            data=degradados,
            x="nivel_percentual",
            y=metrica,
            hue="Degradação",
            col="Modelo",
            kind="line",
            marker="o",
            errorbar=("ci", 95),
            palette=paleta,
            facet_kws={"sharey": True},
            height=4.1,
            aspect=0.95,
        )
        grade.set_axis_labels("Nível de degradação (%)", rotulo)
        grade.set_titles("{col_name}")
        grade.fig.suptitle(f"{rotulo} por modelo e degradação", y=1.04)
        grade.savefig(destino / f"{metrica}_por_nivel.png", dpi=220, bbox_inches="tight")
        plt.close(grade.fig)

    base = tabela[tabela["degradacao"] == "baseline"].set_index(["modelo", "semente"])
    degradados["delta_macro_f1"] = degradados.apply(
        lambda r: r["macro_f1"] - base.loc[(r["modelo"], r["semente"]), "macro_f1"],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    sns.scatterplot(
        data=degradados,
        x="delta_macro_f1",
        y="shap_global_spearman",
        hue="Degradação",
        style="Modelo",
        size="Nível (%)",
        sizes=(35, 130),
        palette=paleta,
        ax=ax,
    )
    ax.axvline(0, color="#444444", linewidth=1)
    ax.axhline(0.80, color="#444444", linewidth=1, linestyle="--")
    ax.set(
        title="Perda preditiva e estabilidade global das explicações",
        xlabel="Variação de Macro-F1 em relação ao baseline",
        ylabel="Correlação de Spearman do ranking SHAP",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(
        destino / "desempenho_vs_estabilidade.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = argumentos()
    pasta = RAIZ / "resultados" / args.modo
    metricas_path = pasta / "metricas_execucoes.csv"
    if not metricas_path.exists():
        raise FileNotFoundError(f"Resultados não encontrados: {metricas_path}")
    tabela = pd.read_csv(metricas_path)
    tabela = adicionar_estabilidade(tabela, pasta)
    tabela.to_csv(pasta / "metricas_com_estabilidade.csv", index=False)
    variabilidade = variabilidade_baseline(tabela, pasta)
    variabilidade.to_csv(pasta / "variabilidade_baseline_shap.csv", index=False)
    if not variabilidade.empty:
        (
            variabilidade.groupby("modelo")[METRICAS_SHAP]
            .quantile([0.05, 0.95])
            .rename_axis(index=["modelo", "quantil"])
            .reset_index()
            .to_csv(pasta / "limites_variabilidade_baseline.csv", index=False)
        )
    resumo_descritivo(tabela).to_csv(pasta / "resumo_descritivo.csv", index=False)
    tabela_resultados_principais(tabela).to_csv(
        pasta / "tabela_resultados_principais.csv", index=False
    )
    if args.modo == "completo":
        testes_estatisticos(tabela).to_csv(
            pasta / "testes_estatisticos.csv", index=False
        )
        associacoes_exploratorias(tabela).to_csv(
            pasta / "associacoes_exploratorias.csv", index=False
        )
        hipoteses = avaliar_hipoteses(tabela, variabilidade)
        (pasta / "avaliacao_hipoteses.json").write_text(
            json.dumps(hipoteses, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    criar_figuras(tabela, pasta / "figuras")
    print(f"Análise concluída em {pasta}")


if __name__ == "__main__":
    main()
