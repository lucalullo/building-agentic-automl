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

Version 2 extends this baseline with a dedicated **Data Inspector**:

1. inspect dataset dimensions;
2. inspect target distribution or summary statistics;
3. inspect feature data types;
4. detect missing values;
5. measure feature cardinality;
6. store the inspection inside the agent state.

Version 3 adds a **Preprocessing Agent** while keeping the model and validation protocol unchanged:

1. try a small set of preprocessing strategies;
2. keep missing values natively or apply simple imputation;
3. learn preprocessing parameters only from the training data;
4. evaluate every strategy using the same model and validation split;
5. compare the experiment results;
6. select the best preprocessing strategy;
7. store the experiments and selected strategy inside the agent state.

Each version preserves the previous workflow as much as possible so that the effect of the new capability can be studied in isolation.

Future versions will progressively introduce model comparison, smarter validation, feature engineering, hyperparameter optimization and more advanced agentic components.

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
- inspect dataset dimensions;
- inspect target distribution for classification tasks;
- inspect target summary statistics for regression tasks;
- inspect feature data types;
- detect missing values;
- measure feature cardinality;
- try multiple preprocessing strategies;
- keep missing values and let LightGBM handle them natively;
- impute missing numerical values with the training median;
- impute missing categorical values with the most frequent training value;
- learn preprocessing parameters only from the training data;
- align categorical vocabularies between training and validation data;
- select a LightGBM baseline model according to the detected task;
- select an evaluation metric automatically;
- create a train-validation split;
- preserve class proportions during classification through stratification;
- train and evaluate each preprocessing experiment;
- compare experiment results using the appropriate metric direction;
- select the best preprocessing strategy;
- adapt automatically between classification and regression;
- return the complete result, dataset inspection, experiments and selected preprocessing strategy as a structured state.

For classification:

```text
Model  → LGBMClassifier
Metric → ROC AUC
Best   → highest score
```

For regression:

```text
Model  → LGBMRegressor
Metric → RMSE
Best   → lowest score
```

## Current architecture

```text
Dataset + Target
       ↓
 Task Detection
       ↓
Feature Detection
       ↓
 Data Inspection
       ↓
  Prepare X / y
       ↓
Try Preprocessing Strategies
  native / impute
       ↓
Train / Validation Split
       ↓
Preprocess Training + Validation
       ↓
LightGBM Training + Evaluation
       ↓
 Compare Experiments
       ↓
Select Best Preprocessing
       ↓
State + Experiments
```

The current architecture remains intentionally compact.

The `agent()` function still coordinates the complete workflow.

It first loads the dataset, determines whether the target represents a classification or regression problem, detects numerical and categorical features, and runs the Data Inspector introduced in Version 2.

Version 3 keeps the inspection step unchanged and adds a small preprocessing experiment loop.

The raw features and target are first separated. For every available preprocessing strategy, the same train-validation split is recreated with `test_size=0.2` and `random_state=42`. Classification also uses stratification.

Two preprocessing strategies are currently available:

```text
native → keep missing values and let LightGBM handle them
impute → numerical median + categorical most frequent value
```

Preprocessing parameters are learned only from the training subset. Categorical vocabularies are also built from training data and then applied consistently to training and validation data.

Each strategy is evaluated with the same LightGBM model family and the metric selected for the detected task. The agent stores every experiment, compares the scores, and selects the best preprocessing result.

A typical classification state from the Adult Income dataset looks like:

```python
{
    "task": "classification",
    "numerical_features": [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week"
    ],
    "categorical_features": [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ],
    "inspection": {
        "shape": {
            "rows": 48842,
            "columns": 15
        },
        "target": {
            "name": "income",
            "dtype": "object",
            "unique_values": 2,
            "distribution": {
                "<=50K": 37155,
                ">50K": 11687
            }
        },
        "dtypes": {...},
        "missing_values": {...},
        "cardinality": {...}
    },
    "model": "LGBMClassifier",
    "metric": "roc_auc",
    "experiments": [
        {
            "preprocessing": "native",
            "score": 0.9311658417981501
        },
        {
            "preprocessing": "impute",
            "score": 0.930687503244851
        }
    ],
    "best_preprocessing": "native",
    "score": 0.9311658417981501
}
```

The same agent can also receive a regression dataset without changing its internal workflow:

```python
{
    "task": "regression",
    "model": "LGBMRegressor",
    "metric": "rmse",
    "experiments": [
        {
            "preprocessing": "native",
            "score": 1.8760451113860066
        },
        {
            "preprocessing": "impute",
            "score": 1.8776921187920332
        }
    ],
    "best_preprocessing": "native",
    "score": 1.8760451113860066
}
```

Version 3 therefore introduces the first explicit experiment loop in the project: the agent no longer uses one fixed preparation choice, but tries alternatives, evaluates them under the same protocol, and retains the evidence behind the selected result.

## Project versions

| Version | Main concept | Status | Folder |
|---|---|---|---|
| Version 1 | Baseline Agent | Completed | [`v01-baseline-agent`](v01-baseline-agent/) |
| Version 2 | Data Inspector | Completed | [`v02-data-inspector`](v02-data-inspector/) |
| Version 3 | Preprocessing Agent | Completed | [`v03-preprocessing-agent`](v03-preprocessing-agent/) |

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


## Version 2 - Data Inspector

Version 2 introduces the first new capability on top of the baseline agent: **dataset inspection before training**.

The complete flow becomes:

```text
Dataset + Target
       ↓
  Detect Task
       ↓
 Detect Features
       ↓
 Inspect Dataset
       ↓
  Prepare Data
       ↓
 Select Model
       ↓
 Select Metric
       ↓
Train + Evaluate
       ↓
 State + Inspection
```

The objective is to make the agent more informed without changing the baseline modeling strategy introduced in Version 1.

The main classification example now uses the **Adult Income** dataset, which contains 48,842 rows, 14 input features and the target column `income`.

### Data inspection

The new component is `inspect_dataset()`:

```python
def inspect_dataset(df, target):
    X = df.drop(columns=target)
    y = df[target]

    target_info = {
        "name": target,
        "dtype": str(y.dtype),
        "unique_values": int(y.nunique())
    }

    if y.nunique() <= 20:
        target_info["distribution"] = y.value_counts(dropna=False).to_dict()
    else:
        target_info["summary"] = y.describe().to_dict()

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "target": target_info,
        "dtypes": {column: str(dtype) for column, dtype in X.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "cardinality": X.nunique(dropna=True).to_dict()
    }
```

The inspector records five groups of information:

```text
shape          → rows and columns
target         → name, dtype, unique values, distribution or summary
dtypes         → feature data types
missing_values → missing count for every column
cardinality    → unique non-null values for every feature
```

For Adult Income, the inspector detects:

```text
rows    = 48,842
columns = 15

target distribution:
<=50K = 37,155
>50K  = 11,687

main missing values:
workclass      = 2,799
occupation     = 2,809
native_country = 857
```

The inspection is descriptive only. Version 2 reports these signals but does not yet automatically impute missing values, remove outliers or transform features.

### Target inspection

The inspector adapts its target summary according to the target structure.

For classification, it stores the class distribution:

```python
"distribution": y.value_counts(dropna=False).to_dict()
```

For regression, it stores descriptive statistics:

```python
"summary": y.describe().to_dict()
```

This allows the same inspection logic to work with both discrete and continuous targets.

### Missing values

Missing values are counted for every column:

```python
"missing_values": df.isna().sum().to_dict()
```

This makes data-quality signals visible in the state without adding remediation logic yet.

### Feature cardinality

The inspector also measures the number of unique non-null values for every input feature:

```python
"cardinality": X.nunique(dropna=True).to_dict()
```

Cardinality provides a first structural signal that future versions can use for preprocessing and feature-handling decisions.

### Agent

Version 2 integrates the inspection step directly into the existing agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    inspection = inspect_dataset(df, target)
    X, y = prepare_data(df, target, categorical)

    model = select_model(task)
    metric = select_metric(task)
    score = train_and_evaluate(X, y, model, task)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "inspection": inspection,
        "model": model.__class__.__name__,
        "metric": metric,
        "score": float(score)
    }
```

Only two architectural changes are required compared with Version 1:

```python
inspection = inspect_dataset(df, target)
```

and:

```python
"inspection": inspection,
```

The rest of the baseline pipeline remains unchanged.

### Classification result

On Adult Income, Version 2 obtains:

```text
task   = classification
model  = LGBMClassifier
metric = roc_auc
score  = 0.9311658417981501
```

### Regression test

The same agent is tested on `simple_regression.csv` using the target column `target`:

```python
REGRESSION_PATH = "/kaggle/input/datasets/lucalullo/agentic-automl-datasets/simple_regression.csv"

regression_state = agent(REGRESSION_PATH, "target")
```

The agent automatically detects regression and returns:

```text
task   = regression
model  = LGBMRegressor
metric = rmse
score  = 1.8760451113860066
```

The regression inspection reports 1,200 rows and 7 columns and summarizes the continuous target with descriptive statistics.

Version 2 therefore adds the first explicit **dataset-awareness layer** to the project while preserving the minimal architecture of Version 1.

Open the folder:

[`v02-data-inspector`](v02-data-inspector/)

## Version 3 - Preprocessing Agent

Version 3 introduces the next capability on top of the Data Inspector: **preprocessing experimentation before training**.

The complete flow becomes:

```text
Dataset + Target
       ↓
  Detect Task
       ↓
 Detect Features
       ↓
 Inspect Dataset
       ↓
   Prepare X / y
       ↓
Try Preprocessing
 native / impute
       ↓
Train + Evaluate
 each strategy
       ↓
Compare Experiments
       ↓
 Select Best
       ↓
State + Experiments
```

The objective is to let the agent test simple preparation choices while keeping the model family and validation strategy unchanged.

The main classification example continues to use the **Adult Income** dataset so that preprocessing is the only new concept introduced in this version.

### Preprocessing strategies

Version 3 compares two simple strategies:

```text
native → keep missing values and let LightGBM handle them
impute → numerical median + categorical most frequent value
```

The available strategies are defined explicitly:

```python
PREPROCESSING_STRATEGIES = ["native", "impute"]
```

The preprocessing logic is implemented in `preprocess_data()`:

```python
def preprocess_data(X_train, X_valid, numerical_features, categorical_features, strategy):
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    if strategy == "impute":
        for column in numerical_features:
            value = X_train[column].median()
            X_train[column] = X_train[column].fillna(value)
            X_valid[column] = X_valid[column].fillna(value)

        for column in categorical_features:
            value = X_train[column].mode().iloc[0]
            X_train[column] = X_train[column].fillna(value)
            X_valid[column] = X_valid[column].fillna(value)

    for column in categorical_features:
        categories = X_train[column].dropna().unique()
        dtype = pd.CategoricalDtype(categories=categories)

        X_train[column] = X_train[column].astype(dtype)
        X_valid[column] = X_valid[column].astype(dtype)

    return X_train, X_valid
```

### Leakage-safe preprocessing

The train-validation split is created before preprocessing.

This means that medians, modes and categorical vocabularies are learned only from the training subset.

The validation data is transformed using the parameters learned from training data, preventing validation leakage.

Categorical features use a shared `pd.CategoricalDtype` built from categories observed in the training set:

```python
categories = X_train[column].dropna().unique()
dtype = pd.CategoricalDtype(categories=categories)

X_train[column] = X_train[column].astype(dtype)
X_valid[column] = X_valid[column].astype(dtype)
```

### Preprocessing experiments

Every strategy is evaluated independently using the same model family and the same validation protocol:

```python
experiments = []

for preprocessing in PREPROCESSING_STRATEGIES:
    model = select_model(task)

    score = train_and_evaluate(
        X, y, model, task,
        numerical_features,
        categorical_features,
        preprocessing
    )

    experiments.append({
        "preprocessing": preprocessing,
        "score": float(score)
    })
```

This creates a small experiment history instead of returning only one training result.

### Best experiment selection

The agent selects the best preprocessing strategy according to the metric direction:

```python
def select_best_experiment(experiments, metric):
    if metric == "rmse":
        return min(experiments, key=lambda experiment: experiment["score"])

    return max(experiments, key=lambda experiment: experiment["score"])
```

Therefore:

```text
ROC AUC → maximize
RMSE    → minimize
```

### Agent

Version 3 integrates the preprocessing experiment loop directly into the existing agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    inspection = inspect_dataset(df, target)
    X, y = prepare_data(df, target)

    metric = select_metric(task)
    experiments = []

    for preprocessing in PREPROCESSING_STRATEGIES:
        model = select_model(task)

        score = train_and_evaluate(
            X, y, model, task,
            numerical,
            categorical,
            preprocessing
        )

        experiments.append({
            "preprocessing": preprocessing,
            "score": float(score)
        })

    best_experiment = select_best_experiment(experiments, metric)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "inspection": inspection,
        "model": select_model(task).__class__.__name__,
        "metric": metric,
        "experiments": experiments,
        "best_preprocessing": best_experiment["preprocessing"],
        "score": best_experiment["score"]
    }
```

The architecture remains intentionally simple: Version 3 adds preprocessing experiments without introducing model comparison, cross-validation, hyperparameter optimization or a planner.

### Classification result

On Adult Income, Version 3 obtains:

```text
native → ROC AUC = 0.9311658417981501
impute → ROC AUC = 0.930687503244851

best_preprocessing = native
```

Because ROC AUC is maximized, `native` is selected.

The result also demonstrates an important AutoML principle: imputation is not assumed to be better automatically. The strategies are compared empirically.

### Regression test

The same agent is tested on `simple_regression.csv`:

```text
native → RMSE = 1.8760451113860066
impute → RMSE = 1.8776921187920332

best_preprocessing = native
```

Because RMSE is minimized, `native` is again selected.

The regression test confirms that the experiment-selection logic adapts correctly to both metric directions.

Version 3 therefore adds the first explicit **preprocessing experimentation layer** while preserving the minimal architecture established in the previous versions.

Open the folder:

[`v03-preprocessing-agent`](v03-preprocessing-agent/)

## Component responsibilities

| Component | Responsibility |
|---|---|
| `detect_task()` | Determines whether the problem is classification or regression |
| `detect_features()` | Identifies numerical and categorical features |
| `inspect_dataset()` | Profiles shape, target, dtypes, missing values and feature cardinality |
| `prepare_data()` | Separates input features `X` from target `y` |
| `preprocess_data()` | Applies the selected preprocessing strategy using parameters learned from training data |
| `select_model()` | Selects the LightGBM baseline model |
| `select_metric()` | Selects the evaluation metric |
| `train_and_evaluate()` | Creates the split, applies preprocessing, trains the model and computes the score |
| `select_best_experiment()` | Selects the best preprocessing experiment according to the metric direction |
| `agent()` | Coordinates the complete AutoML workflow and preprocessing experiment loop |
| State | Stores task, features, inspection, model, metric, experiments, selected preprocessing and final score |

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

For Version 2:

- [`building-agentic-automl.ipynb`](v02-data-inspector/building-agentic-automl.ipynb)
- [`Relazione Versione 2 - Data Inspector.pdf`](v02-data-inspector/Relazione%20Versione%202%20-%20Data%20Inspector.pdf)
- [`Report Version 2 - Data Inspector.pdf`](v02-data-inspector/Report%20Version%202%20-%20Data%20Inspector.pdf)
- [`Version 2.png`](v02-data-inspector/Version%202.png)

For Version 3:

- [`building-agentic-automl.ipynb`](v03-preprocessing-agent/building-agentic-automl.ipynb)
- [`Relazione Versione 3 - Preprocessing Agent.pdf`](v03-preprocessing-agent/Relazione%20Versione%203%20-%20Preprocessing%20Agent.pdf)
- [`Report Version 3 - Preprocessing Agent.pdf`](v03-preprocessing-agent/Report%20Version%203%20-%20Preprocessing%20Agent.pdf)
- [`Version 3.png`](v03-preprocessing-agent/Version%203.png)

## Repository structure

```text
building-agentic-automl/
├── v01-baseline-agent/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 1 - Baseline Agent.pdf
│   ├── Report Version 1 - Baseline Agent.pdf
│   └── Version 1.png
│
├── v02-data-inspector/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 2 - Data Inspector.pdf
│   ├── Report Version 2 - Data Inspector.pdf
│   └── Version 2.png
│
├── v03-preprocessing-agent/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 3 - Preprocessing Agent.pdf
│   ├── Report Version 3 - Preprocessing Agent.pdf
│   └── Version 3.png
│
├── README.md
├── LICENSE
└── .gitignore
```

The repository structure grows progressively as new versions are introduced.

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

Then open one of the completed versions:

```text
v01-baseline-agent/building-agentic-automl.ipynb
v02-data-inspector/building-agentic-automl.ipynb
v03-preprocessing-agent/building-agentic-automl.ipynb
```

and run the cells in order.

When running locally, update the dataset paths used inside the notebooks so they point to the local CSV files.

## Development approach

The project follows one main principle:

> **One version, one concept, one architectural improvement.**

Instead of immediately building a complex AutoML platform, the system evolves through small and understandable steps.

Version 1 establishes the minimal agentic loop:

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

Version 2 adds an explicit inspection step:

```text
observe
   ↓
inspect
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

Version 3 adds a minimal experimentation loop over preprocessing:

```text
observe
   ↓
inspect
   ↓
try
   ↓
evaluate
   ↓
compare
   ↓
select
   ↓
return
```

The state now becomes richer not only because it stores dataset information, but also because it records alternative preprocessing experiments and the selected result.

Each completed version remains available as an independent learning resource.

## Roadmap

- [x] Version 1 - Baseline Agent
- [x] Version 2 - Data Inspector
- [x] Version 3 - Preprocessing Agent
- [ ] Version 4 - Model Selection
- [ ] Version 5 - Smart Validation
- [ ] Version 6 - Feature Engineering
- [ ] Version 7 - Hyperparameter Optimization
- [ ] Version 8 - Senior Agent

The exact architecture of future versions can evolve as new concepts are introduced.

The guiding rule remains:

> **One version, one concept, one architectural improvement.**

## Current limitations

Version 3 is intentionally compact and educational:

- task detection still relies only on the number of unique target values;
- the Data Inspector remains descriptive and does not decide which preprocessing strategies should be tried;
- only two preprocessing strategies are compared: native missing-value handling and simple imputation;
- numerical imputation uses only the median;
- categorical imputation uses only the most frequent value;
- there is no scaling or comparison of encoding strategies;
- there is no dedicated outlier detection or treatment;
- there is no explicit data-leakage detection beyond fitting preprocessing parameters only on training data;
- duplicate rows, skewness and semantic feature types are not inspected yet;
- only one model family, LightGBM, is available;
- there is no comparison between multiple algorithms;
- there is no hyperparameter optimization;
- evaluation uses a single train-validation split;
- there is no cross-validation;
- model selection is still based only on the detected task;
- all decisions are coordinated directly by one `agent()` function;
- there is no planner or multi-agent architecture yet.

These limitations are intentional.

Version 3 focuses specifically on preprocessing experimentation. Model comparison, stronger validation and optimization remain separate concepts for later versions.

## Project status

**Version 1 - Baseline Agent is completed.**

**Version 2 - Data Inspector is completed.**

**Version 3 - Preprocessing Agent is completed.**

The project currently provides a functional Agentic AutoML baseline, a dataset-awareness layer and its first preprocessing experiment loop.

The agent can move automatically from a tabular dataset and target column to an evaluated LightGBM baseline while adapting between classification and regression.

Before training, it can inspect dataset dimensions, target behavior, feature types, missing values and feature cardinality.

It can now also compare native missing-value handling with simple imputation, learn preprocessing parameters only from training data, evaluate both strategies under the same validation protocol, store the experiments and return the best preprocessing result.

The project is intentionally not considered complete at Version 3.

Future versions can progressively add model selection, smarter validation, feature engineering, hyperparameter optimization and more advanced agentic orchestration while preserving the educational structure of the project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
