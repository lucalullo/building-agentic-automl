# Building Agentic AutoML

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Baseline-2E8B57)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)

A progressive educational project that shows how to build an **Agentic AutoML system step by step**, starting from a minimal baseline agent and gradually evolving toward a more autonomous machine learning workflow.

The project focuses on understanding how an AutoML agent can inspect tabular data, make modeling decisions, prepare features, select an appropriate model and metric, train and evaluate the model, and return the complete result as a structured state.

The architecture intentionally starts simple. Each version introduces one new concept while preserving the ideas developed in the previous versions.

## Project roadmap

The project follows one main principle:

> **One version, one concept, one architectural improvement.**

Version 1 establishes the fundamental AutoML workflow:

1. receive a dataset and target;
2. detect the machine learning task;
3. identify feature types;
4. prepare the data;
5. select a baseline model;
6. select an evaluation metric;
7. train and evaluate the model;
8. return a structured state.

Future versions will progressively introduce more robust preprocessing, model comparison, validation strategies, optimization and more advanced agentic components.

## Repository

The project is available on GitHub:

### [Open Building Agentic AutoML on GitHub](https://github.com/lucalullo/building-agentic-automl)

## Current capabilities

The current agent can:

- load a tabular CSV dataset;
- receive the target column from the user;
- detect whether the problem is classification or regression;
- identify numerical features automatically;
- identify categorical features automatically;
- convert categorical features to the pandas `category` dtype;
- select a LightGBM baseline model according to the detected task;
- select an evaluation metric automatically;
- create a train-validation split;
- preserve class proportions during classification through stratification;
- train the selected model;
- evaluate the model;
- adapt automatically between classification and regression;
- return the complete result as a structured state.

For classification:

```text
Model  → LGBMClassifier
Metric → ROC AUC
```

For regression:

```text
Model  → LGBMRegressor
Metric → RMSE
```

## Current architecture

```text
Dataset + Target
       ↓
 Task Detection
       ↓
Feature Detection
       ↓
 Data Preparation
       ↓
 Model Selection
       +
Metric Selection
       ↓
Train / Validation Split
       ↓
Training + Evaluation
       ↓
      State
```

The current architecture is intentionally compact.

The `agent()` function coordinates the complete workflow.

It first loads the dataset and determines whether the target represents a classification or regression problem.

The agent then detects numerical and categorical features directly from the dataframe dtypes.

Categorical columns are converted to the pandas `category` dtype so they can be handled directly by LightGBM.

The model and metric are selected according to the detected task.

The dataset is then divided into training and validation subsets, the baseline model is trained, its performance is evaluated, and the result is returned as a structured state.

A typical classification state looks like:

```python
{
    "task": "classification",
    "numerical_features": [
        "num_1",
        "num_2",
        "num_3",
        "num_4"
    ],
    "categorical_features": [
        "category_1",
        "category_2"
    ],
    "model": "LGBMClassifier",
    "metric": "roc_auc",
    "score": 0.9068167604752971
}
```

The same agent can also receive a regression dataset without changing its internal workflow:

```python
{
    "task": "regression",
    "model": "LGBMRegressor",
    "metric": "rmse",
    "score": 1.8760451113860066
}
```

This is the first important AutoML behavior of the project: the system changes its modeling decisions according to the characteristics of the target.

## Project versions

| Version | Main concept | Status | Folder |
|---|---|---|---|
| Version 1 | Baseline Agent | Completed | [`v01-baseline-agent`](v01-baseline-agent/) |

## Version 1 - Baseline Agent

The first version introduces the complete baseline AutoML flow:

```text
Dataset + Target
       ↓
 Detect Task
       ↓
Detect Features
       ↓
 Prepare Data
       ↓
Select Model
       ↓
Select Metric
       ↓
Train + Evaluate
       ↓
    State
```

The objective is not yet to create a complete production AutoML platform.

Instead, Version 1 establishes the smallest functional architecture capable of observing a tabular dataset, making basic modeling decisions and returning an evaluated machine learning baseline.

### Task detection

The agent determines the task by inspecting the number of unique values in the target column.

```python
def detect_task(df, target):
    n_unique = df[target].nunique()

    if n_unique <= 20:
        return "classification"

    return "regression"
```

The rule is intentionally simple.

Targets with at most 20 unique values are currently interpreted as classification problems. Otherwise, the agent treats the task as regression.

### Feature detection

Numerical and categorical features are identified directly from the dataframe dtypes:

```python
def detect_features(df, target):
    X = df.drop(columns=target)

    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    return numerical, categorical
```

This allows the agent to adapt automatically to the structure of a tabular dataset without requiring a manually defined feature schema.

### Data preparation

Categorical features are converted to the pandas `category` dtype:

```python
def prepare_data(df, target, categorical_features):
    data = df.copy()

    for column in categorical_features:
        data[column] = data[column].astype("category")

    X = data.drop(columns=target)
    y = data[target]

    return X, y
```

LightGBM can then use these categorical features directly.

### Model selection

The baseline model depends on the detected task:

```python
def select_model(task):
    if task == "classification":
        return LGBMClassifier(random_state=42, verbosity=-1)

    return LGBMRegressor(random_state=42, verbosity=-1)
```

Version 1 deliberately uses only one model family.

No model comparison or hyperparameter optimization is performed yet.

### Metric selection

The evaluation metric also depends on the task:

```python
def select_metric(task):
    if task == "classification":
        return "roc_auc"

    return "rmse"
```

The current mapping is:

```text
classification → ROC AUC
regression     → RMSE
```

### Training and evaluation

The dataset is divided into training and validation sets using:

```text
test_size    = 0.2
random_state = 42
```

Classification tasks additionally use:

```python
stratify=y
```

to preserve the target class distribution.

Evaluation then depends on the detected task:

```text
classification → predict_proba → ROC AUC
regression     → predict       → RMSE
```

### Agent

The individual functions are combined into the main agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    X, y = prepare_data(df, target, categorical)

    model = select_model(task)
    metric = select_metric(task)
    score = train_and_evaluate(X, y, model, task)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "model": model.__class__.__name__,
        "metric": metric,
        "score": float(score)
    }
```

In Version 1, the Agent still coordinates every operation directly.

This is intentional: the objective is to understand the complete baseline workflow before separating responsibilities into more advanced components.

Open the folder:

[`v01-baseline-agent`](v01-baseline-agent/)

## Component responsibilities

| Component | Responsibility |
|---|---|
| `detect_task()` | Determines whether the problem is classification or regression |
| `detect_features()` | Identifies numerical and categorical features |
| `prepare_data()` | Prepares features and target for modeling |
| `select_model()` | Selects the LightGBM baseline model |
| `select_metric()` | Selects the evaluation metric |
| `train_and_evaluate()` | Creates the split, trains the model and computes the score |
| `agent()` | Coordinates the complete AutoML workflow |
| State | Stores the task, detected features, model, metric and evaluation score |

## Documentation

Each completed version contains:

- a Jupyter Notebook;
- an Italian technical report;
- an English technical report;
- a version-specific architecture diagram.

For Version 1:

- [`building-agentic-automl.ipynb`](v01-baseline-agent/building-agentic-automl.ipynb)
- [`Relazione Versione 1 - Baseline Agent.pdf`](v01-baseline-agent/Relazione%20Versione%201%20-%20Baseline%20Agent.pdf)
- [`Report Version 1 - Baseline Agent.pdf`](v01-baseline-agent/Report%20Version%201%20-%20Baseline%20Agent.pdf)
- [`Version 1.png`](v01-baseline-agent/Version%201.png)

## Repository structure

```text
building-agentic-automl/
├── v01-baseline-agent/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 1 - Baseline Agent.pdf
│   ├── Report Version 1 - Baseline Agent.pdf
│   └── Version 1.png
│
├── README.md
├── LICENSE
└── .gitignore
```

The repository structure will grow progressively as new versions are introduced.

Each completed version remains available independently so that every architectural step can be studied and compared with the following versions.

## Run locally

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- LightGBM
- scikit-learn

Install the main dependencies:

```bash
pip install pandas lightgbm scikit-learn jupyter
```

Clone the repository:

```bash
git clone https://github.com/lucalullo/building-agentic-automl.git
cd building-agentic-automl
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
v01-baseline-agent/building-agentic-automl.ipynb
```

and run the cells in order.

When running locally, update the dataset paths used inside the notebook so they point to the local CSV files.

## Development approach

The project follows one main principle:

> **One version, one concept, one architectural improvement.**

Instead of immediately building a complex AutoML platform, the system evolves through small and understandable steps.

Version 1 intentionally keeps the architecture minimal:

```text
observe
   ↓
decide
   ↓
prepare
   ↓
train
   ↓
evaluate
   ↓
return
```

This makes each decision explicit and provides a clear foundation for progressively more autonomous AutoML behavior.

Each completed version will remain available as an independent learning resource.

## Roadmap

- [x] Baseline Agent
- [ ] More robust task detection
- [ ] More advanced data validation and preprocessing
- [ ] Multiple model families
- [ ] Model comparison and selection
- [ ] Cross-validation
- [ ] Hyperparameter optimization
- [ ] Richer experiment state and artifacts
- [ ] More modular decision components
- [ ] More advanced agentic orchestration

The exact architecture of future versions can evolve as new concepts are introduced.

The guiding rule remains:

> **One version, one concept, one architectural improvement.**

## Current limitations

Version 1 is intentionally compact and educational:

- task detection relies only on the number of unique target values;
- only one model family, LightGBM, is available;
- there is no comparison between multiple algorithms;
- there is no hyperparameter optimization;
- categorical conversion is the only explicit preprocessing strategy;
- evaluation uses a single train-validation split;
- there is no cross-validation;
- there is no advanced missing-value strategy;
- there is no dedicated outlier handling;
- there is no explicit data-leakage detection;
- model selection is based only on the detected task;
- all decisions are still coordinated directly by one `agent()` function;
- there is no planner or multi-agent architecture yet.

These limitations are intentional.

They define the starting point from which the architecture can evolve progressively.

## Project status

**Version 1 - Baseline Agent is completed.**

The project currently provides the first functional foundation of Building Agentic AutoML.

The agent can move automatically from a tabular dataset and target column to an evaluated LightGBM baseline while adapting between classification and regression.

The project is intentionally not considered complete at Version 1.

Future versions can progressively introduce additional AutoML and agentic capabilities while preserving the educational structure of the project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
