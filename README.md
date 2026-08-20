# Building Agentic AutoML

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-2E8B57)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-EB5B28)
![CatBoost](https://img.shields.io/badge/CatBoost-Model-FFCC00)
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

Version 4 adds **Model Selection** while keeping preprocessing and validation unchanged:

1. build candidate models for the detected task;
2. compare LightGBM, XGBoost and CatBoost;
3. evaluate every preprocessing-model combination using the same validation split;
4. compare results using the same task-specific metric;
5. select the best preprocessing-model combination;
6. store all experiments, the selected preprocessing strategy, model and score inside the agent state.

Version 5 adds **Smart Validation** while keeping preprocessing, model candidates and hyperparameter configurations unchanged:

1. replace the single train-validation split with task-aware cross-validation;
2. use `StratifiedKFold` for classification and `KFold` for regression;
3. evaluate every preprocessing-model combination across 5 folds;
4. store fold scores, mean score and standard deviation for every experiment;
5. select the best experiment using the mean validation score;
6. return the selected validation strategy together with `best_preprocessing`, `best_model`, `best_score` and `best_std`.

Version 6 adds **Feature Engineering** while keeping preprocessing, models, metrics, validation and hyperparameter configurations unchanged:

1. compare the original feature space with an engineered feature space;
2. use `none` to preserve the original features;
3. use `interactions` to add pairwise multiplication features between numerical variables;
4. apply feature engineering inside each validation fold after preprocessing;
5. evaluate every preprocessing-feature engineering-model combination across 5 folds;
6. return `best_feature_engineering` together with the previously selected experiment information.

Version 7 adds **Hyperparameter Optimization** while keeping preprocessing, feature engineering, models, metrics and validation unchanged:

1. define a small model-specific hyperparameter search space;
2. preserve the original Version 6 configuration as a baseline for every model;
3. compare one additional configuration for LightGBM, XGBoost and CatBoost;
4. evaluate every preprocessing-feature engineering-model-hyperparameter combination across 5 folds;
5. store the hyperparameters together with fold scores, mean score and standard deviation;
6. return `best_params` together with the best preprocessing, feature engineering strategy, model and validation results.

Each version preserves the previous workflow as much as possible so that the effect of the new capability can be studied in isolation.

Future versions will progressively introduce more advanced agentic decision-making and orchestration.

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
- preserve missing values whenever the selected model supports them directly;
- impute missing numerical values with the training median;
- impute missing categorical values with the most frequent training value;
- learn preprocessing parameters only from the training data or training fold;
- align categorical vocabularies between training and validation data;
- compare multiple feature engineering strategies;
- preserve the original feature space with `none`;
- generate pairwise numerical interaction features with `interactions`;
- apply feature engineering independently inside each validation fold after preprocessing;
- build LightGBM, XGBoost and CatBoost candidates according to the detected task;
- compare multiple hyperparameter configurations for each model;
- preserve the original model configuration as an HPO baseline;
- store the hyperparameter configuration tested in every experiment;
- select an evaluation metric automatically;
- select a validation strategy automatically;
- use `StratifiedKFold` for classification and `KFold` for regression;
- evaluate every preprocessing-feature engineering-model-hyperparameter combination across 5 folds;
- store fold scores, mean score and standard deviation for every experiment;
- compare experiment results using the appropriate metric direction;
- select the best experiment using the mean validation score;
- adapt automatically between classification and regression;
- return the complete result, dataset inspection, experiments, selected validation strategy, selected preprocessing strategy, selected feature engineering strategy, selected model, selected hyperparameters, best score and score variability as a structured state.

For classification:

```text
Models              → LightGBM / XGBoost / CatBoost
Feature engineering → none / interactions
Hyperparameters     → baseline / alternative configuration
Validation          → StratifiedKFold (5 folds)
Metric              → ROC AUC
Best                → highest mean score
```

For regression:

```text
Models              → LightGBM / XGBoost / CatBoost
Feature engineering → none / interactions
Hyperparameters     → baseline / alternative configuration
Validation          → KFold (5 folds)
Metric              → RMSE
Best                → lowest mean score
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
Try Feature Engineering Strategies
  none / interactions
       ↓
 Build Model Candidates
LightGBM / XGBoost / CatBoost
       ↓
Try Hyperparameter Configurations
 baseline / alternative
       ↓
 Select Validation Strategy
 StratifiedKFold / KFold
       ↓
Preprocess Inside Each Fold
       ↓
Engineer Features Inside Each Fold
       ↓
Create Fresh Model with Parameters
       ↓
 Train + Evaluate
 every fold, every combination
       ↓
 Compare Experiments
 mean score + std
       ↓
 Select Best Experiment
       ↓
State + Experiments
```

The current architecture remains intentionally compact.

The `agent()` function still coordinates the complete workflow.

It loads the dataset, detects the task and feature types, runs the Data Inspector introduced in Version 2, keeps the preprocessing experiment logic introduced in Version 3, preserves the model-selection layer introduced in Version 4, uses the smart-validation layer introduced in Version 5, and retains the feature-engineering layer introduced in Version 6.

Version 7 adds a controlled hyperparameter-optimization layer.

For every preprocessing strategy, feature engineering strategy and candidate model, the agent evaluates two hyperparameter configurations:

```text
baseline    → original Version 6 model configuration
alternative → one predefined model-specific configuration
```

The current search spaces are:

```text
LightGBM
baseline    → {}
alternative → n_estimators=200, learning_rate=0.05, num_leaves=31

XGBoost
baseline    → {}
alternative → n_estimators=200, learning_rate=0.05, max_depth=4

CatBoost
baseline    → {}
alternative → iterations=500, learning_rate=0.05, depth=6
```

Validation remains task-aware:

```text
classification → StratifiedKFold
regression     → KFold
```

Both strategies use 5 folds, `shuffle=True` and `random_state=42`.

Preprocessing and feature engineering remain isolated inside each fold. A fresh model instance is then created using the hyperparameter configuration being evaluated.

The experiment grid is now:

```text
2 preprocessing
× 2 feature engineering
× 3 models
× 2 hyperparameter configurations
= 24 experiments
```

Each experiment is evaluated across 5 folds:

```text
24 experiments × 5 folds = 120 model fits per dataset
```

Every experiment stores:

```text
preprocessing
feature_engineering
model
hyperparameters
fold_scores
mean_score
std_score
```

A typical classification state from the Adult Income dataset now looks like:

```python
{
    "task": "classification",
    "numerical_features": [...],
    "categorical_features": [...],
    "inspection": {...},
    "metric": "roc_auc",
    "validation": "StratifiedKFold",
    "n_splits": 5,
    "experiments": [
        {
            "preprocessing": "native",
            "feature_engineering": "none",
            "model": "CatBoost",
            "hyperparameters": {},
            "fold_scores": [...],
            "mean_score": 0.930630626301291,
            "std_score": 0.002246310473216
        },
        {
            "preprocessing": "native",
            "feature_engineering": "none",
            "model": "CatBoost",
            "hyperparameters": {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6
            },
            "fold_scores": [...],
            "mean_score": 0.9299103499067497,
            "std_score": 0.002476258487812128
        },
        ...
    ],
    "best_preprocessing": "native",
    "best_feature_engineering": "none",
    "best_model": "CatBoost",
    "best_params": {},
    "best_score": 0.930630626301291,
    "best_std": 0.002246310473216
}
```

On classification, the original CatBoost configuration remains the best experiment.

The same agent can also receive a regression dataset without changing its overall workflow. On `simple_regression.csv`, the best experiment is:

```text
validation               = KFold
best_preprocessing       = impute
best_feature_engineering = none
best_model               = CatBoost
best_params              = iterations=500, learning_rate=0.05, depth=6
best_score               = 1.6078463494
best_std                 = 0.1001204396
metric                   = RMSE
```

Version 7 therefore adds the first explicit **hyperparameter-optimization layer** while preserving all previous architectural decisions.

## Project versions

| Version | Main concept | Status | Folder |
|---|---|---|---|
| Version 1 | Baseline Agent | Completed | [`v01-baseline-agent`](v01-baseline-agent/) |
| Version 2 | Data Inspector | Completed | [`v02-data-inspector`](v02-data-inspector/) |
| Version 3 | Preprocessing Agent | Completed | [`v03-preprocessing-agent`](v03-preprocessing-agent/) |
| Version 4 | Model Selection | Completed | [`v04-model-selection`](v04-model-selection/) |
| Version 5 | Smart Validation | Completed | [`v05-smart-validation`](v05-smart-validation/) |
| Version 6 | Feature Engineering | Completed | [`v06-feature-engineering`](v06-feature-engineering/) |
| Version 7 | Hyperparameter Optimization | Completed | [`v07-hyperparameter-optimization`](v07-hyperparameter-optimization/) |

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

## Version 4 - Model Selection

Version 4 introduces the next capability on top of preprocessing experimentation: **automatic model comparison**.

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
 Build Candidate Models
LightGBM / XGBoost / CatBoost
       ↓
 Train + Evaluate
 every combination
       ↓
 Compare Experiments
       ↓
 Select Best
       ↓
 State + Experiments
```

The objective is to let the agent choose among multiple strong model candidates while keeping preprocessing and validation unchanged.

The main classification example continues to use the **Adult Income** dataset so that model selection is the only new concept introduced in this version.

### Model candidates

Version 4 compares three high-performance boosting models:

```text
LightGBM
XGBoost
CatBoost
```

The available models depend on the detected task:

```python
def select_models(task, categorical_features):
    if task == "classification":
        return {
            "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
            "XGBoost": XGBClassifier(
                random_state=42,
                tree_method="hist",
                enable_categorical=True,
                verbosity=0
            ),
            "CatBoost": CatBoostClassifier(
                random_seed=42,
                cat_features=categorical_features,
                verbose=False,
                allow_writing_files=False
            )
        }

    return {
        "LightGBM": LGBMRegressor(random_state=42, verbosity=-1),
        "XGBoost": XGBRegressor(
            random_state=42,
            tree_method="hist",
            enable_categorical=True,
            verbosity=0
        ),
        "CatBoost": CatBoostRegressor(
            random_seed=42,
            cat_features=categorical_features,
            verbose=False,
            allow_writing_files=False
        )
    }
```

Hyperparameter optimization is deliberately excluded from this version.

### Model-selection experiments

Every preprocessing strategy is evaluated with every candidate model:

```python
experiments = []

for preprocessing in PREPROCESSING_STRATEGIES:
    for model_name, model in models.items():
        score = train_and_evaluate(
            X, y, model, model_name, task,
            numerical_features,
            categorical_features,
            preprocessing
        )

        experiments.append({
            "preprocessing": preprocessing,
            "model": model_name,
            "score": float(score)
        })
```

This produces a compact experiment grid:

```text
preprocessing × model → score
```

The same train-validation split and metric logic are preserved from Version 3.

### Best experiment selection

The existing experiment-selection logic remains unchanged:

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

Version 4 integrates model comparison directly into the existing agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    inspection = inspect_dataset(df, target)
    X, y = prepare_data(df, target)

    metric = select_metric(task)
    models = select_models(task, categorical)
    experiments = []

    for preprocessing in PREPROCESSING_STRATEGIES:
        for model_name, model in models.items():
            score = train_and_evaluate(
                X, y, model, model_name, task,
                numerical, categorical, preprocessing
            )

            experiments.append({
                "preprocessing": preprocessing,
                "model": model_name,
                "score": float(score)
            })

    best_experiment = select_best_experiment(experiments, metric)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "inspection": inspection,
        "metric": metric,
        "experiments": experiments,
        "best_preprocessing": best_experiment["preprocessing"],
        "best_model": best_experiment["model"],
        "best_score": best_experiment["score"]
    }
```

### Classification result

On Adult Income, Version 4 selects:

```text
best_preprocessing = impute
best_model         = CatBoost
best_score         = 0.932104618262178
metric             = ROC AUC
```

### Regression test

On `simple_regression.csv`, Version 4 selects:

```text
best_preprocessing = impute
best_model         = CatBoost
best_score         = 1.6542186741
metric             = RMSE
```

Version 4 therefore adds an explicit **model-selection layer** while keeping task detection, inspection, preprocessing strategies, metrics and validation protocol unchanged.

Open the folder:

[`v04-model-selection`](v04-model-selection/)

## Version 5 - Smart Validation

Version 5 introduces the next capability on top of model selection: **task-aware cross-validation**.

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
 Build Candidate Models
LightGBM / XGBoost / CatBoost
       ↓
Select Validation
StratifiedKFold / KFold
       ↓
 Train + Evaluate
 every fold, every combination
       ↓
 Compare Experiments
 mean score + std
       ↓
 Select Best
       ↓
 State + Experiments
```

The objective is to make model selection more reliable by replacing the single train-validation split with a more robust validation strategy, while keeping preprocessing and model candidates unchanged.

The main classification example continues to use the **Adult Income** dataset so that smart validation is the only new concept introduced in this version.

### Validation strategy

Version 5 introduces task-aware validation:

```python
def select_validation(task, n_splits=5):
    if task == "classification":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

    return KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )
```

Therefore:

```text
classification → StratifiedKFold
regression     → KFold
```

Both use 5 folds, `shuffle=True` and `random_state=42`.

### Smart-validation experiments

Every preprocessing strategy is evaluated with every candidate model across all folds:

```python
experiments = []

for preprocessing in PREPROCESSING_STRATEGIES:
    for model_name in models:
        result = train_and_evaluate(
            X,
            y,
            model_name,
            task,
            numerical_features,
            categorical_features,
            preprocessing,
            validation
        )

        experiments.append({
            "preprocessing": preprocessing,
            "model": model_name,
            "fold_scores": result["fold_scores"],
            "mean_score": result["mean_score"],
            "std_score": result["std_score"]
        })
```

This produces a richer experiment grid:

```text
preprocessing × model × folds → fold_scores, mean_score, std_score
```

Preprocessing is refit inside each fold and every fold uses a fresh model instance.

### Best experiment selection

Selection now uses the mean score instead of the score from one split:

```python
def select_best_experiment(experiments, metric):
    if metric == "rmse":
        return min(experiments, key=lambda experiment: experiment["mean_score"])

    return max(experiments, key=lambda experiment: experiment["mean_score"])
```

Therefore:

```text
ROC AUC → maximize mean_score
RMSE    → minimize mean_score
```

### Agent

Version 5 integrates smart validation directly into the existing agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    inspection = inspect_dataset(df, target)
    X, y = prepare_data(df, target)

    metric = select_metric(task)
    validation = select_validation(task)
    models = select_models(task, categorical)

    experiments = []

    for preprocessing in PREPROCESSING_STRATEGIES:
        for model_name in models:
            result = train_and_evaluate(
                X,
                y,
                model_name,
                task,
                numerical,
                categorical,
                preprocessing,
                validation
            )

            experiments.append({
                "preprocessing": preprocessing,
                "model": model_name,
                "fold_scores": result["fold_scores"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"]
            })

    best_experiment = select_best_experiment(experiments, metric)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "inspection": inspection,
        "metric": metric,
        "validation": validation.__class__.__name__,
        "n_splits": validation.n_splits,
        "experiments": experiments,
        "best_preprocessing": best_experiment["preprocessing"],
        "best_model": best_experiment["model"],
        "best_score": best_experiment["mean_score"],
        "best_std": best_experiment["std_score"]
    }
```

### Classification result

On Adult Income, Version 5 selects:

```text
validation         = StratifiedKFold
best_preprocessing = native
best_model         = CatBoost
best_score         = 0.930630626301291
best_std           = 0.002246310473216
metric             = ROC AUC
```

### Regression test

On `simple_regression.csv`, Version 5 selects:

```text
validation         = KFold
best_preprocessing = impute
best_model         = CatBoost
best_score         = 1.6283421164
best_std           = 0.1019398476
metric             = RMSE
```

Version 5 therefore adds an explicit **smart-validation layer** while keeping task detection, inspection, preprocessing strategies, models, metrics and hyperparameter configurations unchanged.

Open the folder:

[`v05-smart-validation`](v05-smart-validation/)

## Version 6 - Feature Engineering

Version 6 introduces the next capability on top of smart validation: **automatic feature engineering**.

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
Try Feature Engineering
 none / interactions
       ↓
 Build Candidate Models
LightGBM / XGBoost / CatBoost
       ↓
 Select Validation
StratifiedKFold / KFold
       ↓
 Train + Evaluate
 every fold, every combination
       ↓
 Compare Experiments
 mean score + std
       ↓
 Select Best
       ↓
 State + Experiments
```

The objective is to let the agent compare alternative feature spaces while keeping preprocessing, model candidates, metrics, validation and hyperparameter configurations unchanged.

The main classification example continues to use the **Adult Income** dataset so that feature engineering is the only new concept introduced in this version.

### Feature engineering strategies

Version 6 compares two strategies:

```text
none         → preserve the original feature space
interactions → add pairwise multiplication features between numerical variables
```

The available strategies are defined explicitly:

```python
FEATURE_ENGINEERING_STRATEGIES = ["none", "interactions"]
```

Feature engineering is implemented in `apply_feature_engineering()`:

```python
def apply_feature_engineering(
    X_train,
    X_valid,
    numerical_features,
    strategy
):
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    if strategy == "interactions":
        for i in range(len(numerical_features)):
            for j in range(i + 1, len(numerical_features)):
                feature_a = numerical_features[i]
                feature_b = numerical_features[j]

                new_feature = f"{feature_a}_x_{feature_b}"

                X_train[new_feature] = (
                    X_train[feature_a] * X_train[feature_b]
                )

                X_valid[new_feature] = (
                    X_valid[feature_a] * X_valid[feature_b]
                )

    return X_train, X_valid
```

The original features are always preserved. The `interactions` strategy only adds new numerical columns.

### Fold-safe feature engineering

Feature engineering is applied inside each validation fold after preprocessing:

```text
split fold
   ↓
preprocess training + validation
   ↓
apply feature engineering
   ↓
fit fresh model
   ↓
evaluate validation fold
```

This preserves the fold-isolated architecture introduced in Version 5.

The current interaction transformation has no learned parameters, but keeping it inside each fold makes the pipeline consistent with future feature transformations that may require fitting.

### Feature-engineering experiments

Every preprocessing strategy is evaluated with every feature engineering strategy and every candidate model:

```python
experiments = []

for preprocessing in PREPROCESSING_STRATEGIES:
    for feature_engineering in FEATURE_ENGINEERING_STRATEGIES:
        for model_name in models:
            result = train_and_evaluate(
                X,
                y,
                model_name,
                task,
                numerical_features,
                categorical_features,
                preprocessing,
                feature_engineering,
                validation
            )

            experiments.append({
                "preprocessing": preprocessing,
                "feature_engineering": feature_engineering,
                "model": model_name,
                "fold_scores": result["fold_scores"],
                "mean_score": result["mean_score"],
                "std_score": result["std_score"]
            })
```

This expands the experiment grid to:

```text
preprocessing × feature engineering × model × folds
→ fold_scores, mean_score, std_score
```

With 2 preprocessing strategies, 2 feature engineering strategies, 3 models and 5 folds, one dataset evaluation performs 60 model fits.

### Best experiment selection

The selection logic remains unchanged from Version 5:

```python
def select_best_experiment(experiments, metric):
    if metric == "rmse":
        return min(experiments, key=lambda experiment: experiment["mean_score"])

    return max(experiments, key=lambda experiment: experiment["mean_score"])
```

Therefore:

```text
ROC AUC → maximize mean_score
RMSE    → minimize mean_score
```

The selected experiment now also exposes:

```python
"best_feature_engineering": best_experiment["feature_engineering"]
```

### Agent

Version 6 integrates feature engineering directly into the existing agent:

```python
def agent(data_path, target):
    df = pd.read_csv(data_path)

    task = detect_task(df, target)
    numerical, categorical = detect_features(df, target)
    inspection = inspect_dataset(df, target)
    X, y = prepare_data(df, target)

    metric = select_metric(task)
    validation = select_validation(task)
    models = select_models(task, categorical)

    experiments = []

    for preprocessing in PREPROCESSING_STRATEGIES:
        for feature_engineering in FEATURE_ENGINEERING_STRATEGIES:
            for model_name in models:
                result = train_and_evaluate(
                    X,
                    y,
                    model_name,
                    task,
                    numerical,
                    categorical,
                    preprocessing,
                    feature_engineering,
                    validation
                )

                experiments.append({
                    "preprocessing": preprocessing,
                    "feature_engineering": feature_engineering,
                    "model": model_name,
                    "fold_scores": result["fold_scores"],
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"]
                })

    best_experiment = select_best_experiment(experiments, metric)

    return {
        "task": task,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "inspection": inspection,
        "metric": metric,
        "validation": validation.__class__.__name__,
        "n_splits": validation.n_splits,
        "experiments": experiments,
        "best_preprocessing": best_experiment["preprocessing"],
        "best_feature_engineering": best_experiment["feature_engineering"],
        "best_model": best_experiment["model"],
        "best_score": best_experiment["mean_score"],
        "best_std": best_experiment["std_score"]
    }
```

### Classification result

On Adult Income, Version 6 selects:

```text
validation               = StratifiedKFold
best_preprocessing       = native
best_feature_engineering = none
best_model               = CatBoost
best_score               = 0.930630626301291
best_std                 = 0.002246310473216
metric                   = ROC AUC
```

The best interaction-based classification experiment is `native + interactions + CatBoost`, with mean ROC AUC `0.9297864575`.

The generated interactions therefore do not improve the best classification score in this version. This is a valid AutoML outcome: the agent evaluates the engineered feature space rather than assuming that it must be better.

### Regression test

On `simple_regression.csv`, Version 6 selects:

```text
validation               = KFold
best_preprocessing       = impute
best_feature_engineering = none
best_model               = CatBoost
best_score               = 1.6283421164
best_std                 = 0.1019398476
metric                   = RMSE
```

The best interaction-based regression experiment is `impute + interactions + CatBoost`, with mean RMSE `1.6816779756`.

The original feature space therefore remains best on both current datasets.

Version 6 adds an explicit **feature-engineering experimentation layer** while keeping task detection, inspection, preprocessing strategies, models, metrics, smart validation and hyperparameter configurations unchanged.

Open the folder:

[`v06-feature-engineering`](v06-feature-engineering/)

## Version 7 - Hyperparameter Optimization

Version 7 introduces the next capability on top of feature engineering: **automatic comparison of hyperparameter configurations**.

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
Try Feature Engineering
 none / interactions
       ↓
 Build Candidate Models
LightGBM / XGBoost / CatBoost
       ↓
Try Hyperparameters
 baseline / alternative
       ↓
 Select Validation
StratifiedKFold / KFold
       ↓
 Train + Evaluate
 every fold, every combination
       ↓
 Compare Experiments
 mean score + std
       ↓
 Select Best
       ↓
 State + Experiments
```

The objective is to let the agent compare model configurations while keeping task detection, data inspection, preprocessing, feature engineering, model families, metrics and validation unchanged.

The main classification example continues to use the **Adult Income** dataset so that hyperparameter optimization is the only new concept introduced in this version.

### Hyperparameter search spaces

Version 7 uses a deliberately small and explicit search space.

Each model keeps its original Version 6 configuration as a baseline and compares it with one alternative configuration:

```python
HYPERPARAMETER_SEARCH_SPACES = {
    "LightGBM": [
        {},
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31
        }
    ],
    "XGBoost": [
        {},
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4
        }
    ],
    "CatBoost": [
        {},
        {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6
        }
    ]
}
```

The search is intentionally compact.

Version 7 does not introduce random search, Bayesian optimization, Optuna, early stopping or a larger tuning framework. Those additions would represent separate architectural concepts.

### Parameter-aware model construction

The model factory now accepts model-specific parameters:

```python
def select_models(task, categorical_features, model_params=None):
    model_params = model_params or {}

    lightgbm_params = model_params.get("LightGBM", {})
    xgboost_params = model_params.get("XGBoost", {})
    catboost_params = model_params.get("CatBoost", {})

    ...
```

Only the parameters belonging to the model currently being evaluated are applied.

Inside each validation fold, a fresh model instance is created using the selected configuration:

```python
fold_model = select_models(
    task,
    categorical_features,
    {
        model_name: model_params
    }
)[model_name]
```

This preserves the fold-isolated training logic introduced in Version 5.

### Hyperparameter experiments

The experiment grid now includes a fourth decision dimension:

```python
experiments = []

for preprocessing in PREPROCESSING_STRATEGIES:
    for feature_engineering in FEATURE_ENGINEERING_STRATEGIES:
        for model_name in models:
            for model_params in HYPERPARAMETER_SEARCH_SPACES[model_name]:
                result = train_and_evaluate(
                    X,
                    y,
                    model_name,
                    task,
                    numerical_features,
                    categorical_features,
                    preprocessing,
                    feature_engineering,
                    model_params,
                    validation
                )

                experiments.append({
                    "preprocessing": preprocessing,
                    "feature_engineering": feature_engineering,
                    "model": model_name,
                    "hyperparameters": model_params,
                    "fold_scores": result["fold_scores"],
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"]
                })
```

The full grid becomes:

```text
preprocessing
× feature engineering
× model
× hyperparameter configuration
× folds
```

With 2 preprocessing strategies, 2 feature engineering strategies, 3 models, 2 hyperparameter configurations and 5 folds:

```text
2 × 2 × 3 × 2 = 24 experiments
24 × 5 = 120 model fits per dataset
```

### Best experiment selection

The selection rule remains unchanged from Version 5 and Version 6:

```python
def select_best_experiment(experiments, metric):
    if metric == "rmse":
        return min(experiments, key=lambda experiment: experiment["mean_score"])

    return max(experiments, key=lambda experiment: experiment["mean_score"])
```

Therefore:

```text
ROC AUC → maximize mean_score
RMSE    → minimize mean_score
```

The selected state now also exposes:

```python
"best_params": best_experiment["hyperparameters"]
```

### Agent

Version 7 extends the existing `agent()` experiment loop with hyperparameter configurations while preserving the previous decision logic:

```python
for preprocessing in PREPROCESSING_STRATEGIES:
    for feature_engineering in FEATURE_ENGINEERING_STRATEGIES:
        for model_name in models:
            for model_params in HYPERPARAMETER_SEARCH_SPACES[model_name]:
                result = train_and_evaluate(
                    X,
                    y,
                    model_name,
                    task,
                    numerical,
                    categorical,
                    preprocessing,
                    feature_engineering,
                    model_params,
                    validation
                )
```

The final state includes:

```text
task
features
inspection
metric
validation
experiments
best_preprocessing
best_feature_engineering
best_model
best_params
best_score
best_std
```

### Classification result

On Adult Income, Version 7 selects:

```text
validation               = StratifiedKFold
best_preprocessing       = native
best_feature_engineering = none
best_model               = CatBoost
best_params              = {}
best_score               = 0.930630626301291
best_std                 = 0.002246310473216
metric                   = ROC AUC
```

The baseline CatBoost configuration remains the best classification experiment.

This is a valid HPO result: optimization does not guarantee that an alternative parameter configuration will outperform the original configuration.

For example, with `native + none + CatBoost`:

```text
baseline
mean ROC AUC = 0.9306306263
std          = 0.0022463105

alternative
iterations   = 500
learning_rate= 0.05
depth        = 6
mean ROC AUC = 0.9299103499
std          = 0.0024762585
```

The agent therefore correctly keeps the baseline configuration.

### Regression test

On `simple_regression.csv`, Version 7 selects:

```text
validation               = KFold
best_preprocessing       = impute
best_feature_engineering = none
best_model               = CatBoost
best_params              = {"iterations": 500, "learning_rate": 0.05, "depth": 6}
best_score               = 1.6078463494
best_std                 = 0.1001204396
metric                   = RMSE
```

Version 6 previously obtained:

```text
CatBoost + impute + none
mean RMSE = 1.6283421164
```

Version 7 improves this to:

```text
CatBoost + impute + none
iterations    = 500
learning_rate = 0.05
depth         = 6
mean RMSE     = 1.6078463494
```

The absolute RMSE reduction is approximately `0.02050`, corresponding to about `1.26%`.

This demonstrates both possible HPO outcomes in the same version:

```text
classification → baseline parameters remain best
regression     → alternative parameters improve the result
```

Version 7 therefore adds an explicit **hyperparameter-optimization layer** while keeping task detection, inspection, preprocessing, feature engineering, model families, metrics and smart validation unchanged.

Open the folder:

[`v07-hyperparameter-optimization`](v07-hyperparameter-optimization/)

## Component responsibilities

| Component | Responsibility |
|---|---|
| `detect_task()` | Determines whether the problem is classification or regression |
| `detect_features()` | Identifies numerical and categorical features |
| `inspect_dataset()` | Profiles shape, target, dtypes, missing values and feature cardinality |
| `prepare_data()` | Separates input features `X` from target `y` |
| `preprocess_data()` | Applies the selected preprocessing strategy using parameters learned from training data or training folds |
| `apply_feature_engineering()` | Preserves the original feature space or adds pairwise numerical interaction features |
| `HYPERPARAMETER_SEARCH_SPACES` | Stores the baseline and alternative hyperparameter configurations for each model |
| `select_models()` | Builds LightGBM, XGBoost and CatBoost candidates and applies model-specific hyperparameters |
| `select_metric()` | Selects the evaluation metric |
| `select_validation()` | Selects the validation strategy according to the task |
| `train_and_evaluate()` | Applies preprocessing and feature engineering inside each fold, creates a fresh parameterized model instance and returns fold scores, mean and standard deviation |
| `select_best_experiment()` | Selects the best preprocessing-feature engineering-model-hyperparameter experiment according to the metric direction and mean score |
| `agent()` | Coordinates the complete AutoML workflow and cross-validated experiment grid |
| State | Stores task, features, inspection, metric, validation, experiments, selected preprocessing, selected feature engineering strategy, selected model, selected hyperparameters, best score and best standard deviation |

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

For Version 4:

- [`building-agentic-automl.ipynb`](v04-model-selection/building-agentic-automl.ipynb)
- [`Relazione Versione 4 - Model Selection.pdf`](v04-model-selection/Relazione%20Versione%204%20-%20Model%20Selection.pdf)
- [`Report Version 4 - Model Selection.pdf`](v04-model-selection/Report%20Version%204%20-%20Model%20Selection.pdf)
- [`Version 4.png`](v04-model-selection/Version%204.png)

For Version 5:

- [`building-agentic-automl.ipynb`](v05-smart-validation/building-agentic-automl.ipynb)
- [`Relazione Versione 5 - Smart Validation.pdf`](v05-smart-validation/Relazione%20Versione%205%20-%20Smart%20Validation.pdf)
- [`Report Version 5 - Smart Validation.pdf`](v05-smart-validation/Report%20Version%205%20-%20Smart%20Validation.pdf)
- [`Version 5.png`](v05-smart-validation/Version%205.png)

For Version 6:

- [`building-agentic-automl.ipynb`](v06-feature-engineering/building-agentic-automl.ipynb)
- [`Relazione Versione 6 - Feature Engineering.pdf`](v06-feature-engineering/Relazione%20Versione%206%20-%20Feature%20Engineering.pdf)
- [`Report Version 6 - Feature Engineering.pdf`](v06-feature-engineering/Report%20Version%206%20-%20Feature%20Engineering.pdf)
- [`Version 6.png`](v06-feature-engineering/Version%206.png)

For Version 7:

- [`building-agentic-automl.ipynb`](v07-hyperparameter-optimization/building-agentic-automl.ipynb)
- [`Relazione Versione 7 - Hyperparameter Optimization.pdf`](v07-hyperparameter-optimization/Relazione%20Versione%207%20-%20Hyperparameter%20Optimization.pdf)
- [`Report Version 7 - Hyperparameter Optimization.pdf`](v07-hyperparameter-optimization/Report%20Version%207%20-%20Hyperparameter%20Optimization.pdf)
- [`Version 7.png`](v07-hyperparameter-optimization/Version%207.png)

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
├── v04-model-selection/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 4 - Model Selection.pdf
│   ├── Report Version 4 - Model Selection.pdf
│   └── Version 4.png
│
├── v05-smart-validation/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 5 - Smart Validation.pdf
│   ├── Report Version 5 - Smart Validation.pdf
│   └── Version 5.png
│
├── v06-feature-engineering/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 6 - Feature Engineering.pdf
│   ├── Report Version 6 - Feature Engineering.pdf
│   └── Version 6.png
│
├── v07-hyperparameter-optimization/
│   ├── building-agentic-automl.ipynb
│   ├── Relazione Versione 7 - Hyperparameter Optimization.pdf
│   ├── Report Version 7 - Hyperparameter Optimization.pdf
│   └── Version 7.png
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
- XGBoost
- CatBoost
- scikit-learn

Install the main dependencies:

```bash
pip install pandas lightgbm xgboost catboost scikit-learn jupyter
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
v04-model-selection/building-agentic-automl.ipynb
v05-smart-validation/building-agentic-automl.ipynb
v06-feature-engineering/building-agentic-automl.ipynb
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
try preprocessing
   ↓
evaluate
   ↓
compare
   ↓
select
   ↓
return
```

Version 4 extends the experiment loop to model selection:

```text
observe
   ↓
inspect
   ↓
try preprocessing
   ↓
try models
   ↓
evaluate
   ↓
compare
   ↓
select
   ↓
return
```

Version 5 extends the experiment loop to smart validation:

```text
observe
   ↓
inspect
   ↓
try preprocessing
   ↓
try models
   ↓
validate across folds
   ↓
compare
   ↓
select
   ↓
return
```

Version 6 extends the experiment loop to feature engineering:

```text
observe
   ↓
inspect
   ↓
try preprocessing
   ↓
try feature engineering
   ↓
try models
   ↓
validate across folds
   ↓
compare
   ↓
select
   ↓
return
```

Version 7 extends the experiment loop to hyperparameter optimization:

```text
observe
   ↓
inspect
   ↓
try preprocessing
   ↓
try feature engineering
   ↓
try models
   ↓
try hyperparameters
   ↓
validate across folds
   ↓
compare
   ↓
select
   ↓
return
```

The state now records dataset information, alternative preprocessing-feature engineering-model-hyperparameter experiments, the selected validation strategy, the selected preprocessing strategy, the selected feature engineering strategy, the selected model, the selected hyperparameters, the best score and the score variability.

Each completed version remains available as an independent learning resource.

## Roadmap

- [x] Version 1 - Baseline Agent
- [x] Version 2 - Data Inspector
- [x] Version 3 - Preprocessing Agent
- [x] Version 4 - Model Selection
- [x] Version 5 - Smart Validation
- [x] Version 6 - Feature Engineering
- [x] Version 7 - Hyperparameter Optimization
- [ ] Version 8 - Senior Agent

The exact architecture of future versions can evolve as new concepts are introduced.

The guiding rule remains:

> **One version, one concept, one architectural improvement.**

## Current limitations

Version 7 is intentionally compact and educational:

- task detection still relies only on the number of unique target values;
- the Data Inspector remains descriptive and does not decide which preprocessing, feature engineering, models or hyperparameter spaces should be tried;
- only two preprocessing strategies are compared: native missing-value handling and simple imputation;
- numerical imputation uses only the median;
- categorical imputation uses only the most frequent value;
- there is no scaling or comparison of encoding strategies;
- there is no dedicated outlier detection or treatment;
- duplicate rows, skewness and semantic feature types are not inspected yet;
- feature engineering is limited to pairwise multiplication between numerical features;
- there are no ratio, logarithmic, polynomial, datetime or categorical interaction strategies;
- the number of interaction features grows quadratically with the number of numerical features;
- there is no feature selection after feature generation;
- model comparison is limited to LightGBM, XGBoost and CatBoost;
- hyperparameter optimization uses only two predefined configurations per model;
- hyperparameter search spaces are manually defined and intentionally small;
- there is no random search, Bayesian optimization, Optuna or adaptive search strategy;
- there is no early stopping or resource-aware tuning logic;
- cross-validation is fixed to 5 folds;
- there is no group-aware or time-series-aware validation logic;
- there is no nested cross-validation or separate final holdout for an unbiased post-selection performance estimate;
- all decisions are coordinated directly by one `agent()` function;
- there is no planner or multi-agent architecture yet;
- the best fitted pipeline is not yet persisted as a reusable artifact.

These limitations are intentional.

Version 7 focuses specifically on hyperparameter optimization. More advanced agentic decision-making and orchestration remain separate concepts for Version 8.

## Project status

**Version 1 - Baseline Agent is completed.**

**Version 2 - Data Inspector is completed.**

**Version 3 - Preprocessing Agent is completed.**

**Version 4 - Model Selection is completed.**

**Version 5 - Smart Validation is completed.**

**Version 6 - Feature Engineering is completed.**

**Version 7 - Hyperparameter Optimization is completed.**

The project currently provides a functional Agentic AutoML baseline, a dataset-awareness layer, preprocessing experimentation, automatic model comparison, task-aware cross-validation, automatic numerical feature engineering and controlled hyperparameter optimization.

The agent can move automatically from a tabular dataset and target column to cross-validated experiments while adapting between classification and regression.

Before training, it can inspect dataset dimensions, target behavior, feature types, missing values and feature cardinality.

It can compare native missing-value handling with simple imputation, compare the original feature space with pairwise numerical interactions, evaluate LightGBM, XGBoost and CatBoost, compare multiple hyperparameter configurations and validate every complete experiment across multiple folds.

For classification, Version 7 keeps the original CatBoost configuration as the best experiment with mean ROC AUC `0.9306306263`.

For regression, Version 7 selects CatBoost with `iterations=500`, `learning_rate=0.05` and `depth=6`, improving mean RMSE from `1.6283421164` in Version 6 to `1.6078463494`.

The state stores every experiment together with hyperparameters, fold scores, mean score and standard deviation, and returns the best preprocessing strategy, best feature engineering strategy, validation strategy, best model, best hyperparameters, best score and best standard deviation.

The project is intentionally not considered complete at Version 7.

Version 8 can focus on more advanced agentic decision-making while preserving the progressive educational structure of the project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
