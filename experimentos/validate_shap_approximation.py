"""Valida a aproximação TreeSHAP contra o cálculo exato em amostra fixa."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.experimento import (  # noqa: E402
    CATEGORICAS,
    PREDITORES,
    TEST_SEED,
    calcular_shap_agregado,
    carregar_base,
    criar_particao_fixa,
    criar_pipeline,
    dividir_desenvolvimento,
    metricas_estabilidade,
)


def main() -> None:
    protocolo = json.loads((RAIZ / "protocolo.json").read_text(encoding="utf-8"))
    calibracao = json.loads(
        (RAIZ / "modelos" / "configuracao_congelada.json").read_text(encoding="utf-8")
    )
    dados = carregar_base(RAIZ / protocolo["dataset"]["file"])
    partes = criar_particao_fixa(dados)
    treino, _ = dividir_desenvolvimento(partes.desenvolvimento, 11)
    categorias = [calibracao["categories"][c] for c in CATEGORICAS]
    parametros = calibracao["models"]["random_forest"]["best_parameters"]
    pipeline = criar_pipeline("random_forest", categorias, 11, parametros)
    pipeline.fit(treino[PREDITORES], (treino["y"] == "yes").astype(int))
    indices = calibracao["shap_indices"][:50]
    explicacao = partes.teste.loc[indices, PREDITORES]
    background = treino[PREDITORES].sample(n=200, random_state=TEST_SEED)

    inicio = time.perf_counter()
    local_exato, global_exato = calcular_shap_agregado(
        pipeline, "random_forest", explicacao, background, approximate_trees=False
    )
    tempo_exato = time.perf_counter() - inicio
    inicio = time.perf_counter()
    local_aprox, global_aprox = calcular_shap_agregado(
        pipeline, "random_forest", explicacao, background, approximate_trees=True
    )
    tempo_aprox = time.perf_counter() - inicio
    metricas = metricas_estabilidade(
        local_exato, global_exato, local_aprox, global_aprox
    )
    resultado = {
        "model": "random_forest",
        "seed": 11,
        "n_observations": 50,
        "exact_seconds": tempo_exato,
        "approximate_seconds": tempo_aprox,
        **metricas,
    }
    resultado["accepted"] = bool(
        resultado["shap_global_spearman"] >= 0.95
        and resultado["shap_local_cosine_mean"] >= 0.90
    )
    destino = RAIZ / "resultados" / "validacao_shap_aproximado.json"
    destino.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if not resultado["accepted"]:
        raise RuntimeError("A aproximação SHAP não atingiu os critérios pré-definidos.")


if __name__ == "__main__":
    main()
