# Pipeline observável — Pima Indians Diabetes

**Aluno:** Filipe Martins Henrique  
**Tema:** Pima Indians Diabetes  
**Disciplina:** Inteligência Artificial / Sistemas de Informação — UNITINS

## Objetivo

Construir um pipeline reprodutível de classificação binária com **PyTorch** e **MLflow**, registrando parâmetros, métricas, artefatos e traces para comparar diferentes configurações de treinamento.

O objetivo preditivo é classificar o atributo `Outcome` do Pima Indians Diabetes Dataset.

## Experimentos

A arquitetura, o split, a semente e o número de épocas permanecem iguais. As runs alteram apenas a taxa de aprendizado e a regularização L2.

| Run | Learning rate | Weight decay (L2) | Pergunta |
|---|---:|---:|---|
| `baseline` | 0.01 | 0 | Qual é o desempenho da configuração de referência? |
| `lr_menor` | 0.001 | 0 | Uma taxa de aprendizado menor melhora a generalização? |
| `regularizacao_l2` | 0.01 | 1e-5 | A regularização L2 melhora a generalização? |

## Pipeline

1. Carregar o dataset.
2. Transformar zeros inválidos em ausentes nas variáveis `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI`.
3. Dividir os dados em treino/validação/teste de forma estratificada (70/15/15).
4. Ajustar imputação pela mediana **somente no treino**.
5. Ajustar `StandardScaler` **somente no treino**.
6. Executar três runs da MLP.
7. Registrar no MLflow parâmetros e métricas por época.
8. Comparar os modelos usando somente a validação.
9. Escolher a maior `val_f1` (com `val_accuracy` como desempate).
10. Avaliar apenas o modelo escolhido no conjunto de teste reservado.

Cada run refaz deterministicamente o mesmo preparo com a mesma seed. Isso permite que o trace de cada run mostre claramente `prepare_data -> train -> validate`, sem alterar o split entre experimentos.

## Métricas rastreadas

Por época:

- `train_loss`
- `val_loss`
- `val_accuracy`
- `val_f1`

Ao final da validação de cada run:

- `final_val_loss`
- `final_val_accuracy`
- `final_val_precision`
- `final_val_recall`
- `final_val_f1`

Somente para a run escolhida:

- `test_loss`
- `test_accuracy`
- `test_precision`
- `test_recall`
- `test_f1`

## Artefatos

Cada run registra:

- curva de `train_loss` e `val_loss`;
- curva de `val_accuracy` e `val_f1`;
- `model_state_dict.pt` da própria run;
- `preprocessing_summary.json`;
- `scaler.joblib`;
- `imputer.joblib`.

A run selecionada também registra:

- matriz de confusão do teste;
- relatório de classificação do teste;
- cópia dos pesos do modelo selecionado;
- resumo comparativo dos experimentos.

## Traces

As etapas principais instrumentadas no MLflow são:

- `prepare_data`
- `train`
- `validate`
- `final_test` — apenas na run escolhida

As métricas de treino/validação continuam sendo registradas por época, sem criar dezenas de spans no trace.

## Ambiente utilizado

- **Python:** `3.12.15`
- **PyTorch:** `2.5.1+cpu` (instalado via índice oficial de CPU do PyTorch)
- **MLflow:** `3.17.0` (`sqlite:///mlflow.db`)
- **Scikit-learn:** `1.5.2` | **NumPy:** `1.26.4` | **Pandas:** `2.2.3` | **Matplotlib:** `3.9.2`

## Como executar

### 1. Criar ambiente virtual (Python 3.12)

Windows PowerShell (com Miniconda/Conda Python 3.12):

```powershell
conda create -n py312 python=3.12 -y
C:\Users\Filipe\miniconda3\envs\py312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Ou diretamente caso o Python 3.12 seja o padrão no PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar PyTorch (CPU oficial) e dependências

O `requirements.txt` já aponta para `--extra-index-url https://download.pytorch.org/whl/cpu` e fixa versões estáveis (`torch==2.5.1+cpu`, `numpy==1.26.4`, `scipy==1.14.1`, `pandas==2.2.3`, `scikit-learn==1.5.2`, `matplotlib==3.9.2`), evitando bloqueios de DLL pelo *Smart App Control* do Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para validar a instalação do PyTorch:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.rand(2,3))"
```

### 3. Executar os três experimentos

```powershell
.\.venv\Scripts\python.exe main.py
```

O arquivo `data/diabetes.csv` não é versionado no Git (está no `.gitignore`), pois se não existir localmente o script `src/data.py` baixa automaticamente uma cópia pública do dataset na primeira execução (768 registros: 537 treino, 115 validação e 116 teste).

### 4. Abrir o MLflow

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

Depois acesse `http://127.0.0.1:5000` no navegador.

## Estrutura

```text
pima-diabetes-mlflow/
├── artifacts/
│   ├── baseline/
│   ├── lr_menor/
│   ├── regularizacao_l2/
│   ├── final_test/
│   └── experiment_summary.json
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── pipeline.py
├── main.py
├── requirements.txt
├── ROTEIRO_VIDEO.md
├── CHECKLIST_ENTREGA.md
├── .gitignore
└── README.md
```

No repositório são mantidos os gráficos (`.png`) e relatórios (`.json`) em `artifacts/` como evidência dos experimentos. Os binários locais (`*.pt`, `*.joblib`, `mlflow.db`, `mlruns/` e `data/diabetes.csv`) são ignorados pelo Git e gerados automaticamente ao executar `python main.py`.

## Critério de escolha do modelo

O conjunto de teste **não participa da escolha**. As três runs são comparadas pela validação sob o mesmo split. A maior `val_f1` determina a run selecionada; em caso de empate, utiliza-se `val_accuracy`. Somente depois disso o conjunto de teste reservado é utilizado uma vez.

## Resultados

Comparação das três configurações no conjunto de validação (`n_val = 115`):

| Run | Learning rate | Weight decay (L2) | Final Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.01 | 0.0 | 0.0278 | 1.7854 | 0.7217 (72,17%) | 0.5870 (58,70%) | 0.6750 (67,50%) | 0.6279 (62,79%) |
| `lr_menor` | 0.001 | 0.0 | 0.3397 | 0.6096 | 0.7217 (72,17%) | 0.5952 (59,52%) | 0.6250 (62,50%) | 0.6098 (60,98%) |
| **`regularizacao_l2`** | **0.01** | **1e-5** | **0.0394** | **2.0419** | **0.7826 (78,26%)** | **0.6744 (67,44%)** | **0.7250 (72,50%)** | **0.6988 (69,88%)** |

**Run selecionada:** `regularizacao_l2` (maior `val_f1` = `0.6988` e maior `val_accuracy` = `0.7826`)

### Avaliação única no conjunto de teste reservado (`n_test = 116`)

- **Teste final — Loss:** `2.3072`
- **Teste final — Accuracy:** `0.7328` (`73,28%`)
- **Teste final — Precision:** `0.6389` (`63,89%`)
- **Teste final — Recall:** `0.5610` (`56,10%`)
- **Teste final — F1:** `0.5974` (`59,74%`)

**Matriz de confusão no teste (`regularizacao_l2`):**
- Verdadeiros Negativos (TN, classe 0 correta): **62**
- Falsos Positivos (FP, classe 0 prevista como 1): **13**
- Falsos Negativos (FN, classe 1 prevista como 0): **18**
- Verdadeiros Positivos (TP, classe 1 correta): **23**

### Análise crítica

- Nas runs com `learning_rate = 0.01` (`baseline` e `regularizacao_l2`), observa-se queda acentuada da `train_loss` (chegando a `0.0278` e `0.0394` na época 100) acompanhada de elevação da `val_loss` (`1.7854` e `2.0419`), indicando sobreajuste (*overfitting*) nas épocas finais em termos de calibração da perda. Ainda assim, a regularização L2 (`weight_decay = 1e-5`) melhorou substancialmente a fronteira de decisão na validação, elevando a `val_f1` de `0.6279` para `0.6988` e a `val_accuracy` de `0.7217` para `0.7826`.
- Já a run `lr_menor` (`learning_rate = 0.001`) apresentou convergência mais suave e menor divergência entre treino e validação (`train_loss = 0.3397` e `val_loss = 0.6096`), porém com menor revocação (`0.6250`) e menor `val_f1` (`0.6098`) ao final das 100 épocas.

## Evidências para o vídeo

No vídeo, mostrar:

1. estrutura do projeto;
2. comando `.\.venv\Scripts\python.exe main.py`;
3. as três runs no MLflow (`baseline`, `lr_menor`, `regularizacao_l2`);
4. comparação de parâmetros;
5. curvas e métricas de validação;
6. traces `prepare_data`, `train` e `validate` em cada run;
7. justificativa da run escolhida (`regularizacao_l2`);
8. `final_test`, métricas e matriz de confusão da run selecionada.

## Observação

Este projeto tem finalidade acadêmica. O modelo não deve ser interpretado como ferramenta clínica de diagnóstico.

