"""Pipeline reproduzível do experimento de qualidade de dados e SHAP.

Este módulo implementa o protocolo aprovado sem executar trabalho ao ser
importado. A execução é feita por ``run_experiment.py`` e produz um artefato
independente por treinamento, permitindo interrupção e retomada.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .degradacoes import (
    introduzir_erros_atributos,
    introduzir_ruido_rotulos,
    introduzir_valores_ausentes,
)


NUMERICAS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
PREDITORES = NUMERICAS + CATEGORICAS
MODELOS = ("logistic_regression", "random_forest", "xgboost")
DEGRADACOES = ("missing_values", "attribute_errors", "label_noise")
NIVEIS = (0.05, 0.10, 0.20, 0.30)
SEMENTES = (11, 29, 47, 71, 101, 131, 173, 211, 257, 307)
TEST_SEED = 20260929
SHAP_SAMPLE_SIZE = 500


@dataclass(frozen=True)
class Particoes:
    desenvolvimento: pd.DataFrame
    teste: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def carregar_base(caminho: Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho, sep=";")
    esperadas = set(PREDITORES + ["duration", "y"])
    faltantes = sorted(esperadas - set(dados.columns))
    if faltantes:
        raise ValueError(f"A base não contém as colunas esperadas: {faltantes}")
    if set(dados["y"].unique()) != {"yes", "no"}:
        raise ValueError("A variável y deve conter exatamente 'yes' e 'no'.")
    return dados.copy()


def hash_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def criar_particao_fixa(dados: pd.DataFrame) -> Particoes:
    desenvolvimento, teste = train_test_split(
        dados,
        test_size=0.20,
        random_state=TEST_SEED,
        stratify=dados["y"],
    )
    return Particoes(
        desenvolvimento=desenvolvimento.sort_index(),
        teste=teste.sort_index(),
    )


def dividir_desenvolvimento(
    desenvolvimento: pd.DataFrame, semente: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treino, validacao = train_test_split(
        desenvolvimento,
        test_size=0.25,
        random_state=semente,
        stratify=desenvolvimento["y"],
    )
    return treino.sort_index(), validacao.sort_index()


def categorias_congeladas(treino_calibracao: pd.DataFrame) -> list[list[str]]:
    return [
        sorted(treino_calibracao[coluna].dropna().astype(str).unique().tolist())
        for coluna in CATEGORICAS
    ]


def criar_preprocessador(
    modelo: str, categorias: list[list[str]]
) -> ColumnTransformer:
    passos_numericos: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if modelo == "logistic_regression":
        passos_numericos.append(("scaler", StandardScaler()))

    numerico = Pipeline(passos_numericos)
    categorico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=categorias,
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numerico, NUMERICAS),
            ("cat", categorico, CATEGORICAS),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def criar_estimador(modelo: str, semente: int, parametros: dict[str, Any] | None = None):
    if modelo == "logistic_regression":
        estimador = LogisticRegression(
            solver="liblinear", max_iter=3000, random_state=semente
        )
    elif modelo == "random_forest":
        estimador = RandomForestClassifier(
            n_jobs=-1, random_state=semente, class_weight="balanced_subsample"
        )
    elif modelo == "xgboost":
        estimador = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=semente,
        )
    else:
        raise ValueError(f"Modelo desconhecido: {modelo}")
    if parametros:
        estimador.set_params(**parametros)
    return estimador


def criar_pipeline(
    modelo: str,
    categorias: list[list[str]],
    semente: int,
    parametros: dict[str, Any] | None = None,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessador", criar_preprocessador(modelo, categorias)),
            ("modelo", criar_estimador(modelo, semente, parametros)),
        ]
    )


def espaco_busca(modelo: str) -> dict[str, list[Any]]:
    if modelo == "logistic_regression":
        return {
            "modelo__C": [0.01, 0.1, 1.0, 10.0],
            "modelo__class_weight": [None, "balanced"],
        }
    if modelo == "random_forest":
        return {
            "modelo__n_estimators": [200],
            "modelo__max_depth": [None, 12, 24],
            "modelo__min_samples_leaf": [1, 5],
            "modelo__max_features": ["sqrt"],
        }
    if modelo == "xgboost":
        return {
            "modelo__n_estimators": [200, 400],
            "modelo__max_depth": [3, 6],
            "modelo__learning_rate": [0.03, 0.10],
            "modelo__min_child_weight": [1, 5],
            "modelo__subsample": [0.8],
            "modelo__colsample_bytree": [0.8],
        }
    raise ValueError(modelo)


def escolher_limiar(y: pd.Series, probabilidades: np.ndarray) -> tuple[float, float]:
    melhores: list[tuple[float, float]] = []
    for limiar in np.round(np.arange(0.05, 0.951, 0.01), 2):
        previsto = np.where(probabilidades >= limiar, "yes", "no")
        macro_f1 = f1_score(y, previsto, average="macro")
        melhores.append((float(macro_f1), float(limiar)))
    melhor_f1 = max(valor for valor, _ in melhores)
    empatados = [
        limiar for valor, limiar in melhores if np.isclose(valor, melhor_f1)
    ]
    limiar = min(empatados, key=lambda valor: (abs(valor - 0.50), valor))
    return limiar, melhor_f1


def calibrar_modelos(
    dados: pd.DataFrame,
    arquivo_saida: Path,
    modelos_dir: Path,
) -> dict[str, Any]:
    particoes = criar_particao_fixa(dados)
    treino, validacao = dividir_desenvolvimento(particoes.desenvolvimento, TEST_SEED)
    categorias = categorias_congeladas(treino)
    shap_indices, _ = train_test_split(
        particoes.teste.index.to_numpy(),
        train_size=SHAP_SAMPLE_SIZE,
        random_state=TEST_SEED,
        stratify=particoes.teste["y"],
    )
    calibracao: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "calibration_seed": TEST_SEED,
        "test_seed": TEST_SEED,
        "predictors": PREDITORES,
        "numeric_predictors": NUMERICAS,
        "categorical_predictors": CATEGORICAS,
        "categories": dict(zip(CATEGORICAS, categorias, strict=True)),
        "test_indices": [int(i) for i in particoes.teste.index],
        "shap_indices": sorted(int(i) for i in shap_indices),
        "models": {},
    }
    modelos_dir.mkdir(parents=True, exist_ok=True)
    X_treino, y_treino = treino[PREDITORES], treino["y"]
    X_validacao, y_validacao = validacao[PREDITORES], validacao["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=TEST_SEED)

    for nome in MODELOS:
        pipeline = criar_pipeline(nome, categorias, TEST_SEED)
        busca = GridSearchCV(
            pipeline,
            espaco_busca(nome),
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        inicio = time.perf_counter()
        # A codificação binária é usada em todos os estimadores para manter a
        # mesma classe positiva (1 = "yes") e compatibilidade com o XGBoost.
        busca.fit(X_treino, (y_treino == "yes").astype(int))
        prob = busca.best_estimator_.predict_proba(X_validacao)[:, 1]
        limiar, f1_validacao = escolher_limiar(y_validacao, prob)
        parametros = {
            chave.removeprefix("modelo__"): valor
            for chave, valor in busca.best_params_.items()
        }
        calibracao["models"][nome] = {
            "best_parameters": parametros,
            "best_cv_macro_f1": float(busca.best_score_),
            "threshold": limiar,
            "validation_macro_f1": float(f1_validacao),
            "elapsed_seconds": time.perf_counter() - inicio,
            "search_space": espaco_busca(nome),
        }
        joblib.dump(
            busca.best_estimator_, modelos_dir / f"calibracao_{nome}.joblib"
        )

    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    arquivo_saida.write_text(
        json.dumps(calibracao, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return calibracao


def seed_cenario(semente: int, degradacao: str, nivel: float) -> int:
    texto = f"{semente}|{degradacao}|{nivel:.2f}".encode("utf-8")
    return int(hashlib.sha256(texto).hexdigest()[:8], 16)


def aplicar_degradacao(
    treino: pd.DataFrame,
    degradacao: str,
    nivel: float,
    semente: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    resultado = treino.copy(deep=True)
    semente_derivada = seed_cenario(semente, degradacao, nivel)
    if degradacao == "missing_values":
        X, log = introduzir_valores_ausentes(
            treino[PREDITORES], PREDITORES, nivel, semente_derivada
        )
        resultado = pd.concat(
            [X, treino.drop(columns=PREDITORES)], axis=1
        ).loc[:, treino.columns]
        return resultado, log, None
    if degradacao == "attribute_errors":
        X, log = introduzir_erros_atributos(
            treino[PREDITORES], PREDITORES, nivel, semente_derivada
        )
        resultado = pd.concat(
            [X, treino.drop(columns=PREDITORES)], axis=1
        ).loc[:, treino.columns]
        return resultado, log, None
    if degradacao == "label_noise":
        y, log, contagens = introduzir_ruido_rotulos(
            treino["y"], nivel, semente_derivada
        )
        resultado.loc[:, "y"] = y
        return resultado, log, contagens
    raise ValueError(f"Degradação desconhecida: {degradacao}")


def metricas_preditivas(
    y_real: pd.Series, probabilidades: np.ndarray, limiar: float
) -> dict[str, Any]:
    previsto = np.where(probabilidades >= limiar, "yes", "no")
    y_binario = (y_real == "yes").astype(int)
    pr_auc = float(average_precision_score(y_binario, probabilidades))
    matriz = confusion_matrix(y_real, previsto, labels=["no", "yes"])
    return {
        "macro_f1": float(f1_score(y_real, previsto, average="macro")),
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_real, previsto)),
        "roc_auc": float(roc_auc_score(y_binario, probabilidades)),
        "accuracy": float(accuracy_score(y_real, previsto)),
        "tn": int(matriz[0, 0]),
        "fp": int(matriz[0, 1]),
        "fn": int(matriz[1, 0]),
        "tp": int(matriz[1, 1]),
        "predicted_no": int((previsto == "no").sum()),
        "predicted_yes": int((previsto == "yes").sum()),
    }


def mapa_atributos_transformados(
    preprocessador: ColumnTransformer,
) -> tuple[list[str], list[str]]:
    nomes = list(preprocessador.get_feature_names_out())
    originais: list[str] = []
    for nome in nomes:
        sem_prefixo = nome.split("__", 1)[-1]
        encontrado = next(
            (
                coluna
                for coluna in PREDITORES
                if sem_prefixo == coluna or sem_prefixo.startswith(f"{coluna}_")
            ),
            None,
        )
        if encontrado is None:
            raise ValueError(f"Não foi possível mapear o atributo transformado {nome}")
        originais.append(encontrado)
    return nomes, originais


def _normalizar_saida_shap(valores: Any) -> np.ndarray:
    if isinstance(valores, list):
        valores = valores[-1]
    arr = np.asarray(getattr(valores, "values", valores))
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    if arr.ndim != 2:
        raise ValueError(f"Formato SHAP inesperado: {arr.shape}")
    return arr.astype(np.float64, copy=False)


def calcular_shap_agregado(
    pipeline: Pipeline,
    modelo: str,
    X_explicacao: pd.DataFrame,
    X_background: pd.DataFrame,
    approximate_trees: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    import shap

    preprocessador: ColumnTransformer = pipeline.named_steps["preprocessador"]
    estimador = pipeline.named_steps["modelo"]
    X_t = np.asarray(preprocessador.transform(X_explicacao), dtype=np.float64)
    background_t = np.asarray(
        preprocessador.transform(X_background), dtype=np.float64
    )
    if modelo == "logistic_regression":
        mascara = shap.maskers.Independent(
            background_t, max_samples=len(background_t)
        )
        explicador = shap.LinearExplainer(estimador, mascara)
        valores = explicador(X_t)
    else:
        explicador = shap.TreeExplainer(estimador)
        valores = explicador.shap_values(
            X_t,
            approximate=approximate_trees,
            check_additivity=False,
        )
    shap_codificado = _normalizar_saida_shap(valores)
    _, mapeamento = mapa_atributos_transformados(preprocessador)
    shap_agregado = np.zeros((len(X_explicacao), len(PREDITORES)), dtype=np.float64)
    for posicao, atributo in enumerate(mapeamento):
        shap_agregado[:, PREDITORES.index(atributo)] += shap_codificado[:, posicao]
    global_abs = np.mean(np.abs(shap_agregado), axis=0)
    return shap_agregado, global_abs


def id_execucao(modelo: str, semente: int, degradacao: str, nivel: float) -> str:
    nivel_txt = f"{int(round(nivel * 100)):02d}"
    return f"{modelo}__seed-{semente}__{degradacao}__nivel-{nivel_txt}"


def _salvar_mascara(
    log: pd.DataFrame,
    contagens: pd.DataFrame | None,
    destino: Path,
    metadados: dict[str, Any],
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    prefixo = destino / metadados["mask_id"]
    log.to_csv(prefixo.with_suffix(".csv.gz"), index=False, compression="gzip")
    if contagens is not None:
        contagens.to_csv(prefixo.with_name(prefixo.name + "__contagens.csv"))
    prefixo.with_suffix(".json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def executar_treinamento(
    modelo: str,
    semente: int,
    degradacao: str,
    nivel: float,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    shap_teste: pd.DataFrame,
    calibracao: dict[str, Any],
    diretorio_saida: Path,
    diretorio_mascaras: Path,
    forcar: bool = False,
) -> dict[str, Any]:
    run_id = id_execucao(modelo, semente, degradacao, nivel)
    runs_dir = diretorio_saida / "runs"
    shap_dir = diretorio_saida / "shap"
    for pasta in (runs_dir, shap_dir):
        pasta.mkdir(parents=True, exist_ok=True)
    resultado_path = runs_dir / f"{run_id}.json"
    if resultado_path.exists() and not forcar:
        return json.loads(resultado_path.read_text(encoding="utf-8"))

    inicio = time.perf_counter()
    categorias = [calibracao["categories"][coluna] for coluna in CATEGORICAS]
    parametros = calibracao["models"][modelo]["best_parameters"]
    limiar = float(calibracao["models"][modelo]["threshold"])
    pipeline = criar_pipeline(modelo, categorias, semente, parametros)
    pipeline.fit(treino[PREDITORES], (treino["y"] == "yes").astype(int))
    probabilidades = pipeline.predict_proba(teste[PREDITORES])[:, 1]
    metricas = metricas_preditivas(teste["y"], probabilidades, limiar)

    background = treino[PREDITORES].sample(
        n=min(200, len(treino)), random_state=TEST_SEED
    )
    shap_local, shap_global = calcular_shap_agregado(
        pipeline, modelo, shap_teste[PREDITORES], background
    )
    np.savez_compressed(
        shap_dir / f"{run_id}.npz",
        local=shap_local,
        global_abs=shap_global,
        feature_names=np.asarray(PREDITORES),
        sample_indices=shap_teste.index.to_numpy(),
    )
    resultado = {
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "modelo": modelo,
        "semente": semente,
        "degradacao": degradacao,
        "nivel": nivel,
        "threshold": limiar,
        "n_train": len(treino),
        "n_test": len(teste),
        "n_shap": len(shap_teste),
        "train_yes": int((treino["y"] == "yes").sum()),
        "train_no": int((treino["y"] == "no").sum()),
        "elapsed_seconds": time.perf_counter() - inicio,
        **metricas,
    }
    resultado_path.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultado


def executar_matriz(
    dados: pd.DataFrame,
    calibracao: dict[str, Any],
    modo: str,
    raiz_resultados: Path,
    raiz_mascaras: Path,
    forcar: bool = False,
    callback=None,
) -> list[dict[str, Any]]:
    particoes = criar_particao_fixa(dados)
    shap_indices = calibracao["shap_indices"]
    shap_teste = particoes.teste.loc[shap_indices]
    if len(shap_teste) != SHAP_SAMPLE_SIZE:
        raise ValueError("A amostra SHAP não contém exatamente 500 observações.")

    if modo == "piloto":
        sementes = (11, 29, 47)
        modelos = ("random_forest",)
        cenarios = (("baseline", 0.0), ("missing_values", 0.10), ("missing_values", 0.30))
    elif modo == "completo":
        sementes = SEMENTES
        modelos = MODELOS
        cenarios = (("baseline", 0.0),) + tuple(
            (degradacao, nivel)
            for degradacao in DEGRADACOES
            for nivel in NIVEIS
        )
    else:
        raise ValueError("modo deve ser 'piloto' ou 'completo'")

    saida = raiz_resultados / modo
    resultados: list[dict[str, Any]] = []
    for semente in sementes:
        treino_limpo, _ = dividir_desenvolvimento(
            particoes.desenvolvimento, semente
        )
        for degradacao, nivel in cenarios:
            if degradacao == "baseline":
                treino_cenario = treino_limpo.copy()
            else:
                treino_cenario, log, contagens = aplicar_degradacao(
                    treino_limpo, degradacao, nivel, semente
                )
                mask_id = f"seed-{semente}__{degradacao}__nivel-{int(nivel*100):02d}"
                _salvar_mascara(
                    log,
                    contagens,
                    raiz_mascaras / modo,
                    {
                        "mask_id": mask_id,
                        "created_at_utc": utc_now(),
                        "seed": semente,
                        "derived_seed": seed_cenario(semente, degradacao, nivel),
                        "degradation": degradacao,
                        "level": nivel,
                        "changed_items": len(log),
                        "train_rows": len(treino_limpo),
                    },
                )
            for modelo in modelos:
                resultado = executar_treinamento(
                    modelo,
                    semente,
                    degradacao,
                    nivel,
                    treino_cenario,
                    particoes.teste,
                    shap_teste,
                    calibracao,
                    saida,
                    raiz_mascaras,
                    forcar=forcar,
                )
                resultados.append(resultado)
                if callback:
                    callback(resultado)
    consolidar_execucoes(saida)
    return resultados


def consolidar_execucoes(diretorio_saida: Path) -> pd.DataFrame:
    arquivos = sorted((diretorio_saida / "runs").glob("*.json"))
    linhas = [json.loads(arquivo.read_text(encoding="utf-8")) for arquivo in arquivos]
    tabela = pd.DataFrame(linhas)
    if not tabela.empty:
        tabela = tabela.sort_values(
            ["modelo", "semente", "degradacao", "nivel"]
        ).reset_index(drop=True)
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(diretorio_saida / "metricas_execucoes.csv", index=False)
    return tabela


def ambiente_execucao() -> dict[str, str]:
    import matplotlib
    import scipy
    import shap
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def metricas_estabilidade(
    referencia_local: np.ndarray,
    referencia_global: np.ndarray,
    comparacao_local: np.ndarray,
    comparacao_global: np.ndarray,
) -> dict[str, float]:
    spearman = spearmanr(referencia_global, comparacao_global).statistic
    top_ref = set(np.argsort(referencia_global)[-5:])
    top_cmp = set(np.argsort(comparacao_global)[-5:])
    jaccard = len(top_ref & top_cmp) / len(top_ref | top_cmp)
    ref_norm = referencia_global / max(referencia_global.sum(), np.finfo(float).eps)
    cmp_norm = comparacao_global / max(comparacao_global.sum(), np.finfo(float).eps)
    l1 = float(np.abs(ref_norm - cmp_norm).sum())
    numerador = np.sum(referencia_local * comparacao_local, axis=1)
    denominador = np.linalg.norm(referencia_local, axis=1) * np.linalg.norm(
        comparacao_local, axis=1
    )
    cosseno = np.divide(
        numerador,
        denominador,
        out=np.ones_like(numerador, dtype=float),
        where=denominador > 0,
    )
    sinais_validos = (referencia_local != 0) | (comparacao_local != 0)
    concordancias = []
    for i in range(len(referencia_local)):
        validos = sinais_validos[i]
        if not np.any(validos):
            concordancias.append(1.0)
        else:
            concordancias.append(
                float(
                    np.mean(
                        np.sign(referencia_local[i, validos])
                        == np.sign(comparacao_local[i, validos])
                    )
                )
            )
    return {
        "shap_global_spearman": float(spearman),
        "shap_global_jaccard_top5": float(jaccard),
        "shap_global_l1_normalized": l1,
        "shap_local_cosine_mean": float(np.mean(cosseno)),
        "shap_local_cosine_median": float(np.median(cosseno)),
        "shap_local_sign_agreement_mean": float(np.mean(concordancias)),
    }


def carregar_shap(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(caminho, allow_pickle=False) as dados:
        return dados["local"], dados["global_abs"]

