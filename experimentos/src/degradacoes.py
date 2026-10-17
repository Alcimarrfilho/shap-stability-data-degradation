"""Degradações controladas definidas no protocolo experimental.

As funções alteram apenas cópias dos dados e retornam registros suficientes
para auditar quais células ou rótulos foram modificados. Elas não executam o
piloto nem treinam modelos.
"""

from __future__ import annotations

from math import floor
from typing import Iterable

import numpy as np
import pandas as pd


def _validar_nivel(nivel: float) -> None:
    if not 0 < nivel < 1:
        raise ValueError("O nível deve estar estritamente entre 0 e 1.")


def _validar_colunas(dados: pd.DataFrame, colunas: Iterable[str]) -> list[str]:
    selecionadas = list(colunas)
    ausentes = sorted(set(selecionadas) - set(dados.columns))
    if ausentes:
        raise KeyError(f"Colunas inexistentes: {ausentes}")
    if not selecionadas:
        raise ValueError("Informe pelo menos uma coluna preditora.")
    return selecionadas


def introduzir_valores_ausentes(
    dados: pd.DataFrame,
    colunas: Iterable[str],
    nivel: float,
    semente: int,
    categorias_legitimas: tuple[str, ...] = ("unknown",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Substitui uma fração exata das células elegíveis por ``NaN``."""

    _validar_nivel(nivel)
    colunas = _validar_colunas(dados, colunas)
    resultado = dados.copy(deep=True)
    elegiveis: list[tuple[object, str]] = []

    for coluna in colunas:
        for indice, valor in dados[coluna].items():
            if pd.isna(valor) or valor in categorias_legitimas:
                continue
            elegiveis.append((indice, coluna))

    quantidade = floor(nivel * len(elegiveis))
    gerador = np.random.default_rng(semente)
    escolhidas = gerador.choice(len(elegiveis), size=quantidade, replace=False)
    registros = []

    for posicao in escolhidas:
        indice, coluna = elegiveis[int(posicao)]
        anterior = resultado.at[indice, coluna]
        resultado.at[indice, coluna] = np.nan
        registros.append(
            {"indice": indice, "atributo": coluna, "valor_anterior": anterior}
        )

    log = pd.DataFrame(registros, columns=["indice", "atributo", "valor_anterior"])
    return resultado, log


def introduzir_erros_atributos(
    dados: pd.DataFrame,
    colunas: Iterable[str],
    nivel: float,
    semente: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Troca células por valores diferentes oriundos da mesma coluna."""

    _validar_nivel(nivel)
    colunas = _validar_colunas(dados, colunas)
    resultado = dados.copy(deep=True)
    elegiveis = [
        (indice, coluna)
        for coluna in colunas
        for indice, valor in dados[coluna].items()
        if pd.notna(valor)
    ]
    quantidade = floor(nivel * len(elegiveis))
    gerador = np.random.default_rng(semente)
    posicoes = gerador.choice(len(elegiveis), size=quantidade, replace=False)
    selecionadas = [elegiveis[int(posicao)] for posicao in posicoes]
    registros = []

    por_coluna: dict[str, list[object]] = {}
    for indice, coluna in selecionadas:
        por_coluna.setdefault(coluna, []).append(indice)

    for coluna, indices in por_coluna.items():
        universo = dados[coluna].dropna()
        if universo.nunique(dropna=True) < 2:
            raise ValueError(f"A coluna {coluna!r} não possui valores alternativos.")

        valores = universo.to_numpy()
        indices_doador = universo.index.to_numpy()
        atuais = dados.loc[indices, coluna].to_numpy()
        sorteios = gerador.integers(0, len(valores), size=len(indices))
        substitutos = valores[sorteios]
        iguais = substitutos == atuais

        while np.any(iguais):
            sorteios[iguais] = gerador.integers(0, len(valores), size=int(iguais.sum()))
            substitutos = valores[sorteios]
            iguais = substitutos == atuais

        for indice, anterior, substituto, posicao_doador in zip(
            indices, atuais, substitutos, sorteios, strict=True
        ):
            resultado.at[indice, coluna] = substituto
            registros.append(
                {
                    "indice": indice,
                    "atributo": coluna,
                    "valor_anterior": anterior,
                    "valor_substituto": substituto,
                    "indice_doador": indices_doador[int(posicao_doador)],
                }
            )

    log = pd.DataFrame(
        registros,
        columns=[
            "indice",
            "atributo",
            "valor_anterior",
            "valor_substituto",
            "indice_doador",
        ],
    )
    return resultado, log


def introduzir_ruido_rotulos(
    rotulos: pd.Series,
    nivel: float,
    semente: int,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Inverte a mesma proporção de exemplos dentro de cada classe binária."""

    _validar_nivel(nivel)
    classes = set(rotulos.dropna().unique())
    if classes != {"yes", "no"}:
        raise ValueError("Os rótulos devem conter exatamente as classes 'yes' e 'no'.")

    resultado = rotulos.copy(deep=True)
    gerador = np.random.default_rng(semente)
    registros = []

    for classe in ("no", "yes"):
        indices_classe = rotulos.index[rotulos == classe].to_numpy()
        quantidade = floor(nivel * len(indices_classe))
        escolhidos = gerador.choice(indices_classe, size=quantidade, replace=False)
        nova_classe = "yes" if classe == "no" else "no"
        resultado.loc[escolhidos] = nova_classe
        registros.extend(
            {
                "indice": indice,
                "rotulo_anterior": classe,
                "rotulo_novo": nova_classe,
            }
            for indice in escolhidos
        )

    log = pd.DataFrame(
        registros, columns=["indice", "rotulo_anterior", "rotulo_novo"]
    )
    contagens = pd.DataFrame(
        {
            "antes": rotulos.value_counts().reindex(["no", "yes"], fill_value=0),
            "depois": resultado.value_counts().reindex(["no", "yes"], fill_value=0),
        }
    )
    contagens.index.name = "classe"
    return resultado, log, contagens
