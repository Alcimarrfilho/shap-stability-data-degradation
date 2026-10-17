import sys
import unittest
from math import floor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.degradacoes import (  # noqa: E402
    introduzir_erros_atributos,
    introduzir_ruido_rotulos,
    introduzir_valores_ausentes,
)


class TestDegradacoes(unittest.TestCase):
    def setUp(self):
        self.dados = pd.DataFrame(
            {
                "idade": [20, 30, 40, 50, 60, 70],
                "trabalho": ["a", "b", "c", "a", "b", "unknown"],
            }
        )

    def test_ausentes_respeitam_quantidade_e_unknown(self):
        degradado, log = introduzir_valores_ausentes(
            self.dados, self.dados.columns, 0.25, 11
        )
        elegiveis = self.dados.size - 1
        self.assertEqual(len(log), floor(0.25 * elegiveis))
        self.assertEqual(degradado.loc[5, "trabalho"], "unknown")
        self.assertFalse(self.dados.isna().any().any())

    def test_erros_preservam_dominio_e_trocam_valor(self):
        degradado, log = introduzir_erros_atributos(
            self.dados, self.dados.columns, 0.25, 29
        )
        self.assertEqual(len(log), floor(0.25 * self.dados.size))
        self.assertTrue((log["valor_anterior"] != log["valor_substituto"]).all())
        for coluna in self.dados.columns:
            self.assertTrue(set(degradado[coluna]).issubset(set(self.dados[coluna])))

    def test_ruido_e_estratificado_e_auditavel(self):
        rotulos = pd.Series(["no"] * 8 + ["yes"] * 4)
        degradado, log, contagens = introduzir_ruido_rotulos(rotulos, 0.25, 47)
        self.assertEqual((log["rotulo_anterior"] == "no").sum(), 2)
        self.assertEqual((log["rotulo_anterior"] == "yes").sum(), 1)
        self.assertEqual(contagens.loc["yes", "antes"], 4)
        self.assertEqual(contagens.loc["yes", "depois"], 5)
        self.assertEqual(len(degradado), len(rotulos))


if __name__ == "__main__":
    unittest.main()
