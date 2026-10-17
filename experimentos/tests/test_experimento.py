import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experimento import (  # noqa: E402
    CATEGORICAS,
    PREDITORES,
    aplicar_degradacao,
    calcular_shap_agregado,
    categorias_congeladas,
    criar_particao_fixa,
    criar_pipeline,
    dividir_desenvolvimento,
    id_execucao,
    metricas_estabilidade,
    metricas_preditivas,
    seed_cenario,
)


def dados_sinteticos(n=120):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(index=np.arange(n))
    frame["age"] = rng.integers(18, 80, n)
    frame["balance"] = rng.normal(1000, 500, n)
    frame["day"] = rng.integers(1, 31, n)
    frame["campaign"] = rng.integers(1, 8, n)
    frame["pdays"] = rng.choice([-1, 3, 15, 90], n)
    frame["previous"] = rng.integers(0, 5, n)
    frame["duration"] = rng.integers(20, 500, n)
    dominios = {
        "job": ["admin", "services", "unknown"],
        "marital": ["single", "married"],
        "education": ["primary", "secondary"],
        "default": ["yes", "no"],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["cellular", "unknown"],
        "month": ["jan", "feb", "mar"],
        "poutcome": ["success", "failure", "unknown"],
    }
    for coluna, valores in dominios.items():
        frame[coluna] = rng.choice(valores, n)
    frame["y"] = np.where(np.arange(n) % 5 == 0, "yes", "no")
    return frame


class TestExperimento(unittest.TestCase):
    def setUp(self):
        self.dados = dados_sinteticos()

    def test_particao_fixa_e_estratificada(self):
        primeira = criar_particao_fixa(self.dados)
        segunda = criar_particao_fixa(self.dados)
        self.assertEqual(len(primeira.teste), 24)
        self.assertListEqual(primeira.teste.index.tolist(), segunda.teste.index.tolist())
        self.assertTrue(set(primeira.teste.index).isdisjoint(primeira.desenvolvimento.index))
        proporcao_geral = (self.dados["y"] == "yes").mean()
        proporcao_teste = (primeira.teste["y"] == "yes").mean()
        self.assertLessEqual(abs(proporcao_teste - proporcao_geral), 1 / len(primeira.teste))

    def test_divisao_por_semente_e_reprodutivel(self):
        desenvolvimento = criar_particao_fixa(self.dados).desenvolvimento
        treino_a, validacao_a = dividir_desenvolvimento(desenvolvimento, 11)
        treino_b, validacao_b = dividir_desenvolvimento(desenvolvimento, 11)
        self.assertListEqual(treino_a.index.tolist(), treino_b.index.tolist())
        self.assertListEqual(validacao_a.index.tolist(), validacao_b.index.tolist())
        self.assertTrue(set(treino_a.index).isdisjoint(validacao_a.index))

    def test_seed_de_cenario_e_estavel_e_especifica(self):
        a = seed_cenario(11, "missing_values", 0.10)
        self.assertEqual(a, seed_cenario(11, "missing_values", 0.10))
        self.assertNotEqual(a, seed_cenario(11, "missing_values", 0.30))
        self.assertNotEqual(a, seed_cenario(29, "missing_values", 0.10))

    def test_degradacao_nao_muda_validacao_ou_teste(self):
        partes = criar_particao_fixa(self.dados)
        treino, validacao = dividir_desenvolvimento(partes.desenvolvimento, 11)
        validacao_original = validacao.copy(deep=True)
        teste_original = partes.teste.copy(deep=True)
        degradado, log, _ = aplicar_degradacao(treino, "missing_values", 0.10, 11)
        self.assertGreater(len(log), 0)
        self.assertTrue(degradado[PREDITORES].isna().any().any())
        pd.testing.assert_frame_equal(validacao, validacao_original)
        pd.testing.assert_frame_equal(partes.teste, teste_original)

    def test_pipeline_random_forest_treina_sem_duration(self):
        partes = criar_particao_fixa(self.dados)
        treino, _ = dividir_desenvolvimento(partes.desenvolvimento, 11)
        categorias = categorias_congeladas(treino)
        pipeline = criar_pipeline(
            "random_forest",
            categorias,
            11,
            {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
        )
        pipeline.fit(treino[PREDITORES], (treino["y"] == "yes").astype(int))
        probabilidades = pipeline.predict_proba(partes.teste[PREDITORES])[:, 1]
        self.assertEqual(len(probabilidades), len(partes.teste))
        self.assertTrue(np.all((probabilidades >= 0) & (probabilidades <= 1)))
        self.assertNotIn("duration", PREDITORES)

    def test_xgboost_e_compativel_com_tree_explainer(self):
        partes = criar_particao_fixa(self.dados)
        treino, _ = dividir_desenvolvimento(partes.desenvolvimento, 11)
        categorias = categorias_congeladas(treino)
        pipeline = criar_pipeline(
            "xgboost",
            categorias,
            11,
            {
                "n_estimators": 10,
                "max_depth": 2,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        )
        pipeline.fit(treino[PREDITORES], (treino["y"] == "yes").astype(int))
        local, global_abs = calcular_shap_agregado(
            pipeline,
            "xgboost",
            partes.teste[PREDITORES].head(5),
            treino[PREDITORES].head(20),
        )
        self.assertEqual(local.shape, (5, len(PREDITORES)))
        self.assertEqual(global_abs.shape, (len(PREDITORES),))
        self.assertTrue(np.isfinite(local).all())

    def test_metricas_preditivas_e_matriz_confusao(self):
        y = pd.Series(["no", "no", "yes", "yes"])
        probabilidades = np.array([0.1, 0.8, 0.7, 0.9])
        resultado = metricas_preditivas(y, probabilidades, 0.5)
        self.assertEqual(resultado["tn"], 1)
        self.assertEqual(resultado["fp"], 1)
        self.assertEqual(resultado["fn"], 0)
        self.assertEqual(resultado["tp"], 2)
        self.assertGreater(resultado["macro_f1"], 0)

    def test_metricas_shap_identicas_sao_estaveis(self):
        local = np.array([[1.0, -1.0], [0.5, 0.25]])
        global_abs = np.mean(np.abs(local), axis=0)
        resultado = metricas_estabilidade(local, global_abs, local, global_abs)
        self.assertAlmostEqual(resultado["shap_global_spearman"], 1.0)
        self.assertAlmostEqual(resultado["shap_global_l1_normalized"], 0.0)
        self.assertAlmostEqual(resultado["shap_local_cosine_mean"], 1.0)
        self.assertAlmostEqual(resultado["shap_local_sign_agreement_mean"], 1.0)

    def test_id_execucao_e_inequivoco(self):
        self.assertEqual(
            id_execucao("random_forest", 11, "missing_values", 0.10),
            "random_forest__seed-11__missing_values__nivel-10",
        )


if __name__ == "__main__":
    unittest.main()
