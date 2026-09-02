# ML Models

A lightweight machine-learning framework implemented in Python, with NumPy as the primary numerical computing library.

## Overview

An independent implementation of machine-learning algorithms and supporting tools, focused on understanding and implementing the underlying algorithms rather than relying on high-level machine-learning frameworks.

The project includes models, preprocessing utilities, evaluation metrics, optimization, composable pipelines, and hyperparameter search.

## Features

### Models

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Trees — ID3 and CART
* Naive Bayes

### Infrastructure

* Model and Transformer protocols
* Composable pipelines
* Gradient Descent optimizer
* K-fold cross-validation
* Hyperparameter search with `GridSearchCV`

### Preprocessing & Evaluation

* Train/test and K-fold splitting
* Normalization and standardization
* Label and ordinal encoding
* Accuracy, precision, recall and F1
* Confusion matrices

## Architecture

```text
                       ML Framework
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       Models          Preprocessing       Metrics
          │                 │                 │
     ┌────┴────┐       ┌────┴────┐            │
     │         │       │         │            │
 ID3/CART     KNN   Normalize  Encoding    Scoring
     │         │
 Logistic   Naive Bayes
 Regression
     │
     └────────── Pipeline ──────────┐
                                    │
                Hyperparameter Search / GridSearchCV
```

## Example

```python
from models import NaiveBayesClassifier
from core import Pipeline
from preprocessing import train_test_split, normalize

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

pipe = Pipeline([
    ("scaling", normalize.z_score()),
    ("estimator", NaiveBayesClassifier())
])

pipe.fit(X_train, Y_train)

results = pipe.score(
    X_test,
    Y_test,
    scores=["f1", "accuracy"]
)

print(f"Accuracy: {results['accuracy']}")
print(f"F1-score: {results['f1']}")
```

## Benchmarks

## Benchmarks

The implementations were evaluated on several datasets using different models and preprocessing configurations.

| Dataset   | Model               | Configuration                | Result          |
|           |                     |                              |                 |
| Iris      | KNN                 | K = 5                        | 96.19% accuracy |
| Wine      | KNN                 | K = 5, distance weighting    | 95.17% accuracy |
| Mushrooms | Decision Tree       | ID3, max_depth, gini/entropy | 99.59% accuracy |
| Mushrooms | Naive Bayes         | —                            | 95.41% accuracy |
| Digits    | Logistic Regression | GridSearchCV, F1             | 94.45% F1       |
| Digits    | Decision Tree       | CART, max_depth, gini        | 85.74% accuracy |

### Decision Tree vs. scikit-learn

On the Digits dataset, the custom Decision Tree implementation was compared against the corresponding scikit-learn implementation under the same benchmark setup.

| Implementation      | Train Accuracy | Test Accuracy | Fit Time |
|                     |                |               |          |
| This implementation | 100.00%        | 85.74%        | 1.3295 s |
| scikit-learn        | 100.00%        | 86.59%        | 0.0544 s |

The benchmark demonstrates comparable predictive performance while also providing a reference point for computational performance.

## Project Structure

```text
ML-models/
├── core/             # Framework infrastructure
├── models/           # Model implementations
├── metrics/          # Evaluation metrics
├── preprocessing/    # Data preprocessing
├── visuals/          # Visualization utilities
├── notebooks/        # Experiments and benchmarks
├── data/             # Dataset files
└── requirements.txt  # Python dependecies
```
