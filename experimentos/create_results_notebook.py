"""Gera e executa o caderno reproduzível dos resultados completos."""

from pathlib import Path

import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

RAIZ = Path(__file__).resolve().parent
destino = RAIZ / "02_resultados_experimento.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Resultados do experimento completo\n\n"
        "Este caderno lê somente os artefatos auditados da matriz de 390 "
        "treinamentos. Ele não recalibra modelos nem altera resultados."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "RAIZ = Path.cwd()\n"
        "PASTA = RAIZ / 'resultados' / 'completo'\n"
        "validacao = json.loads((PASTA / 'validacao_artefatos.json').read_text(encoding='utf-8'))\n"
        "assert validacao['valid'] and validacao['found_runs'] == 390\n"
        "validacao"
    ),
    nbf.v4.new_markdown_cell("## Ambiente, base e configuração congelada"),
    nbf.v4.new_code_cell(
        "manifesto = json.loads((RAIZ / 'resultados' / 'manifesto_execucao.json').read_text(encoding='utf-8'))\n"
        "calibracao = json.loads((RAIZ / 'modelos' / 'configuracao_congelada.json').read_text(encoding='utf-8'))\n"
        "display(pd.Series(manifesto['environment'], name='versão').to_frame())\n"
        "display(pd.DataFrame({\n"
        "    nome: {\n"
        "        'Macro-F1 CV': dados['best_cv_macro_f1'],\n"
        "        'limiar congelado': dados['threshold'],\n"
        "        'Macro-F1 validação': dados['validation_macro_f1'],\n"
        "        'parâmetros': str(dados['best_parameters']),\n"
        "    } for nome, dados in calibracao['models'].items()\n"
        "}).T)"
    ),
    nbf.v4.new_markdown_cell("## Resultados principais no nível de 30%"),
    nbf.v4.new_code_cell(
        "resultados = pd.read_csv(PASTA / 'tabela_resultados_principais.csv')\n"
        "colunas = ['modelo', 'degradacao', 'macro_f1_media', 'delta_macro_f1_medio',\n"
        "           'delta_macro_f1_ic95_inferior', 'delta_macro_f1_ic95_superior',\n"
        "           'shap_spearman_global_medio', 'shap_l1_global_media',\n"
        "           'shap_cosseno_local_medio']\n"
        "resultados.loc[resultados['nivel'].eq(0.30), colunas].round(4)"
    ),
    nbf.v4.new_markdown_cell("## Avaliação pré-registrada das hipóteses"),
    nbf.v4.new_code_cell(
        "hipoteses = json.loads((PASTA / 'avaliacao_hipoteses.json').read_text(encoding='utf-8'))\n"
        "pd.DataFrame([\n"
        "    {'hipótese': chave, 'confirmada': valor['confirmed']}\n"
        "    for chave, valor in hipoteses.items()\n"
        "])"
    ),
    nbf.v4.new_markdown_cell("## Testes estatísticos"),
    nbf.v4.new_code_cell(
        "testes = pd.read_csv(PASTA / 'testes_estatisticos.csv')\n"
        "pd.Series({\n"
        "    'comparações registradas': len(testes),\n"
        "    'pós-testes executados após Friedman significativo': int(testes['posthoc_performed'].sum()),\n"
        "    'comparações significativas após Holm': int(testes['significativo_005'].sum()),\n"
        "}).to_frame('quantidade')"
    ),
    nbf.v4.new_markdown_cell("## Figuras geradas automaticamente"),
    nbf.v4.new_code_cell(
        "for nome in [\n"
        "    'macro_f1_por_nivel.png', 'pr_auc_por_nivel.png',\n"
        "    'shap_global_spearman_por_nivel.png',\n"
        "    'shap_local_cosine_mean_por_nivel.png',\n"
        "    'desempenho_vs_estabilidade.png',\n"
        "]:\n"
        "    print(nome)\n"
        "    display(Image(filename=str(PASTA / 'figuras' / nome), width=950))"
    ),
    nbf.v4.new_markdown_cell(
        "## Limites de interpretação\n\n"
        "Os resultados são válidos para esta base, estes modelos, esta definição "
        "de degradação e este protocolo. Degradações sintéticas não representam "
        "todos os mecanismos reais; uma única base limita a validade externa; "
        "e o TreeSHAP aproximado foi usado após validação registrada contra o "
        "cálculo exato em uma subamostra fixa."
    ),
]

nbf.write(nb, destino)
executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
executor.preprocess(nb, {"metadata": {"path": str(RAIZ)}})
nbf.write(nb, destino)
exportador = HTMLExporter()
html, _ = exportador.from_notebook_node(nb)
(RAIZ / "02_resultados_experimento.html").write_text(html, encoding="utf-8")
print(destino)
