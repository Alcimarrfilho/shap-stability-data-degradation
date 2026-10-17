"""Audita a completude e a integridade estrutural dos artefatos experimentais."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.experimento import (  # noqa: E402
    DEGRADACOES,
    MODELOS,
    NIVEIS,
    PREDITORES,
    SEMENTES,
    id_execucao,
)


def cenarios(modo: str):
    if modo == "piloto":
        return (11, 29, 47), ("random_forest",), (
            ("baseline", 0.0),
            ("missing_values", 0.10),
            ("missing_values", 0.30),
        )
    return SEMENTES, MODELOS, (("baseline", 0.0),) + tuple(
        (d, n) for d in DEGRADACOES for n in NIVEIS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["piloto", "completo"])
    args = parser.parse_args()
    pasta = RAIZ / "resultados" / args.modo
    tabela = pd.read_csv(pasta / "metricas_execucoes.csv")
    sementes, modelos, combinacoes = cenarios(args.modo)
    esperados = {
        id_execucao(modelo, semente, degradacao, nivel)
        for semente in sementes
        for modelo in modelos
        for degradacao, nivel in combinacoes
    }
    encontrados = set(tabela["run_id"])
    erros: list[str] = []
    if encontrados != esperados:
        erros.append(
            f"IDs divergentes: faltam={sorted(esperados-encontrados)}, "
            f"sobram={sorted(encontrados-esperados)}"
        )
    if tabela["run_id"].duplicated().any():
        erros.append("Há IDs de execução duplicados.")
    for metrica in ["macro_f1", "pr_auc", "balanced_accuracy", "roc_auc", "accuracy"]:
        if not tabela[metrica].between(0, 1).all():
            erros.append(f"Métrica fora de [0,1]: {metrica}")
    calibracao = json.loads(
        (RAIZ / "modelos" / "configuracao_congelada.json").read_text(encoding="utf-8")
    )
    indices_esperados = np.asarray(calibracao["shap_indices"])
    for run_id in esperados:
        caminho = pasta / "shap" / f"{run_id}.npz"
        if not caminho.exists():
            erros.append(f"SHAP ausente: {run_id}")
            continue
        with np.load(caminho, allow_pickle=False) as dados:
            if dados["local"].shape != (500, len(PREDITORES)):
                erros.append(f"Forma SHAP local inválida: {run_id}")
            if dados["global_abs"].shape != (len(PREDITORES),):
                erros.append(f"Forma SHAP global inválida: {run_id}")
            if not np.array_equal(dados["sample_indices"], indices_esperados):
                erros.append(f"Amostra SHAP divergente: {run_id}")
            if not np.isfinite(dados["local"]).all():
                erros.append(f"SHAP contém valor não finito: {run_id}")
    mascaras_esperadas = {
        f"seed-{s}__{d}__nivel-{int(n*100):02d}"
        for s in sementes
        for d, n in combinacoes
        if d != "baseline"
    }
    for mask_id in mascaras_esperadas:
        meta_path = RAIZ / "mascaras" / args.modo / f"{mask_id}.json"
        csv_path = RAIZ / "mascaras" / args.modo / f"{mask_id}.csv.gz"
        if not meta_path.exists() or not csv_path.exists():
            erros.append(f"Máscara incompleta: {mask_id}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        quantidade = len(pd.read_csv(csv_path))
        if quantidade != int(meta["changed_items"]):
            erros.append(f"Contagem de máscara divergente: {mask_id}")
    relatorio = {
        "modo": args.modo,
        "expected_runs": len(esperados),
        "found_runs": len(tabela),
        "expected_masks": len(mascaras_esperadas),
        "errors": erros,
        "valid": not erros,
    }
    (pasta / "validacao_artefatos.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if erros:
        raise RuntimeError("A validação dos artefatos falhou.")


if __name__ == "__main__":
    main()
