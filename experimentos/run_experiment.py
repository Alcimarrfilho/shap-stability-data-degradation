"""Executa calibração, piloto ou matriz completa do protocolo aprovado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.experimento import (  # noqa: E402
    ambiente_execucao,
    calibrar_modelos,
    carregar_base,
    executar_matriz,
    hash_arquivo,
)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("acao", choices=["calibrar", "piloto", "completo"])
    parser.add_argument("--forcar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    protocolo_path = RAIZ / "protocolo.json"
    protocolo = json.loads(protocolo_path.read_text(encoding="utf-8"))
    if not protocolo.get("pilot_execution_allowed", False):
        raise RuntimeError("O protocolo ainda não autoriza a execução do piloto.")

    base_path = RAIZ / protocolo["dataset"]["file"]
    calibracao_path = RAIZ / "modelos" / "configuracao_congelada.json"
    dados = carregar_base(base_path)

    manifesto = {
        "environment": ambiente_execucao(),
        "dataset_path": str(base_path.relative_to(RAIZ)),
        "dataset_sha256": hash_arquivo(base_path),
        "protocol_status": protocolo["status"],
        "protocol_approval": protocolo.get("approval"),
    }
    (RAIZ / "resultados").mkdir(parents=True, exist_ok=True)
    (RAIZ / "resultados" / "manifesto_execucao.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.acao == "calibrar" or not calibracao_path.exists():
        print("Calibrando hiperparâmetros e limiares somente nos dados limpos...")
        calibracao = calibrar_modelos(dados, calibracao_path, RAIZ / "modelos")
        print(f"Configuração congelada: {calibracao_path}")
        if args.acao == "calibrar":
            return
    else:
        calibracao = json.loads(calibracao_path.read_text(encoding="utf-8"))

    concluidos = 0

    def progresso(resultado: dict) -> None:
        nonlocal concluidos
        concluidos += 1
        print(
            f"[{concluidos}] {resultado['run_id']} "
            f"Macro-F1={resultado['macro_f1']:.4f} "
            f"tempo={resultado['elapsed_seconds']:.1f}s",
            flush=True,
        )

    executar_matriz(
        dados,
        calibracao,
        args.acao,
        RAIZ / "resultados",
        RAIZ / "mascaras",
        forcar=args.forcar,
        callback=progresso,
    )


if __name__ == "__main__":
    main()
