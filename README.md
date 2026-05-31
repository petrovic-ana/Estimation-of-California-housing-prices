# Estimation of California Housing Prices

Machine Learning course project — School of Electrical Engineering, University of Belgrade.

**Author:** Ana Petrović

## Description

This project predicts housing prices in California using demographic and geographic features. It is inspired by *"Analysis and Forecasting of California Housing"* (Yucong Chen, 2022, International Conference on Management Science and Industrial Engineering).

Three regression models are implemented:

- **Ridge Regression** — linear model with L2 regularization
- **Random Forest** — ensemble of decision trees
- **Gradient Boosting** — sequential ensemble with boosting

Models are evaluated using **RMSE** (Root Mean Squared Error) on the log-transformed target variable.

## Project Structure

```
Projekat/
├── main.ipynb                          # Main notebook with analysis and models
├── housing.csv                         # Dataset (20,640 instances)
├── estimation-of-california-housing-prices.pdf  # Project report
└── README.md
```

## Dataset

The `housing.csv` file contains **20,640** samples with **10** columns:

| Column | Description |
|--------|-------------|
| `longitude` | Block longitude |
| `latitude` | Block latitude |
| `housing_median_age` | Median age of housing units in the block |
| `total_rooms` | Total number of rooms in the block |
| `total_bedrooms` | Total number of bedrooms in the block |
| `population` | Block population |
| `households` | Number of households in the block |
| `median_income` | Median household income (in tens of thousands of USD) |
| `median_house_value` | Median house value (USD) — **target variable** |
| `ocean_proximity` | Categorical feature — proximity to the ocean |

## Data Preprocessing

1. **Missing values** — 207 instances are missing `total_bedrooms`. Missing values are filled with the median `total_bedrooms` of the 20 most similar instances based on `households`.
2. **Categorical feature** — `ocean_proximity` is extended: `NEAR OCEAN` and `<1H OCEAN` are split into northern and southern California categories (`NEAR OCEAN NORTH`, `<1H OCEAN NORTH`) for better informativeness.
3. **Log transformation** — applied to `households`, `median_income`, `median_house_value`, `total_rooms`, `total_bedrooms`, and `population` due to non-normal distributions.
4. **Category encoding** — `LabelEncoder` for `ocean_proximity`.
5. **Standardization** — `StandardScaler` applied to all input features.
6. **Train/test split** — 80% training / 20% test (`random_state=42`).

## Models and Hyperparameters

### Ridge Regression

- Cross-validation (15 folds) for selecting `alpha` in the range \(10^{-5}\)–\(10^{4}\)
- Final model: `alpha=1`
- Analysis of training set size impact (20–10,000 instances)

### Random Forest

- Validation: **out-of-bag (OOB)** score
- Tuned hyperparameters:
  - `n_estimators=250`
  - `max_depth=14`
  - `max_features=6`

### Gradient Boosting

- Cross-validation (5 folds)
- Tuned hyperparameters:
  - `n_estimators=250`
  - `max_depth=10`
  - `learning_rate=0.07`

## Results

| Model | RMSE (this project) | RMSE (reference paper) |
|-------|---------------------|-------------------------|
| Ridge | 0.33078 | 0.37310 |
| Random Forest | 0.23283 | 0.29093 |
| Gradient Boosting | **0.22165** | 0.29484 |

**Gradient Boosting** achieves the best result. All three models produce RMSE values consistent with the reference paper.

## Getting Started

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Run the Project

```bash
jupyter notebook main.ipynb
```

The notebook loads `housing.csv` from the same directory — run it from the project root.

## References

- Chen, Y. (2022). *Analysis and Forecasting of California Housing*. International Conference on Management Science and Industrial Engineering.
- California Housing dataset — classic dataset for housing price regression in California (1990 US Census).
