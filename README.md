# Insurance Growth Analytics

**Neighborhood-level marketing strategy for health insurance plans in NYC**, using real Census data, deep learning, and statistical rigor to identify high-risk communities and project campaign ROI.

Built by **Mark Anthony Anglin**

---

## What This Does

Pulls live ACS 5-Year Census data for all 55 NYC PUMAs (Public Use Microdata Areas), engineers features around health insurance coverage gaps, clusters neighborhoods by risk profile, and serves an interactive Flask dashboard with:

- **Risk scoring** via a PyTorch neural network classifier (validated with Leave-One-Out CV)
- **Customer segmentation** via K-Means on autoencoder-compressed latent space
- **Campaign ROI projections** with bootstrap confidence intervals
- **Statistical validation** at every step (Shapiro-Wilk, Spearman, Kruskal-Wallis)
- **AuADHD-friendly UI** with progressive disclosure, large touch targets, and reduced motion support

## Live Data Pipeline

Data is pulled at runtime from the **U.S. Census Bureau ACS API** (no static CSVs):

| ACS Table | What It Provides |
|-----------|-----------------|
| B27001 | Health insurance coverage by age and sex |
| B17001 | Poverty status (18-64) |
| B18101 | Disability status by age and sex |
| B01003 | Total population |
| B11001 | Total households |

The pipeline filters to NYC PUMAs (`NAME` contains `NYC-`), engineers 15+ features (uninsured rates by age band, poverty rate, disability rate), and assigns borough labels.

## Technical Stack

### Deep Learning (PyTorch)

**Autoencoder** for dimensionality reduction:
```
NeighborhoodAutoencoder(
    Encoder: Linear(7, 8) -> ReLU -> Linear(8, 3) -> ReLU
    Decoder: Linear(3, 8) -> ReLU -> Linear(8, 7)
)
```
Compresses 7 cluster features into a 3D latent space. K-Means runs on the encodings, not raw features, capturing nonlinear neighborhood patterns.

**Binary Classifier** for risk prediction:
```
InsuranceRiskClassifier(
    Linear(8, 32) -> ReLU -> Dropout(0.3)
    Linear(32, 16) -> ReLU -> Dropout(0.2)
    Linear(16, 1) -> Sigmoid
)
Optimizer: Adam (lr=0.001, weight_decay=1e-4)
Loss: Binary Cross-Entropy
```

### Cross-Validation Strategy

With only n=55 neighborhoods, standard train/test splits waste data and produce unstable estimates. **Leave-One-Out CV** trains 55 separate models, each holding out one neighborhood. Every prediction is genuinely out-of-sample.

| Metric | LOO-CV Score |
|--------|-------------|
| Accuracy | 0.909 |
| AUC | 0.972 |
| AUC 95% CI | [0.922, 1.000] (bootstrap, 1000 resamples) |
| Precision | 0.900+ |
| Recall | 0.900+ |

### Statistical Methods

- **Shapiro-Wilk**: Tests normality of uninsured rate, poverty rate, disability rate. Non-normal distributions trigger nonparametric methods.
- **Spearman Rank Correlation**: Robust to non-normality. Tests poverty-uninsured, disability-uninsured, and young adult-uninsured associations with p-values and effect sizes.
- **Kruskal-Wallis H-test**: Nonparametric ANOVA testing whether borough groups and cluster groups differ significantly on uninsured rate.
- **Bootstrap Confidence Intervals**: 1000 resamples for AUC CI, 1000 resamples for cluster mean rate CIs, 20 resamples for cluster stability (silhouette).
- **Permutation Feature Importance**: 10 shuffles per feature, reports mean +/- std drop in accuracy.
- **Silhouette Analysis**: Tests k=2 through k=6, selects optimal cluster count.

### Clustering

K-Means on autoencoder latent representations, with:
- Silhouette-optimized k selection (k=2 through 6)
- Bootstrap stability analysis (20 resamples, reports CI)
- Kruskal-Wallis test confirming clusters differ significantly
- Clusters sorted by uninsured rate (Cluster 0 = highest risk)

### Web Application (Flask + Vega-Lite)

Five dashboard pages:

| Page | What It Shows |
|------|--------------|
| **Dashboard** | KPI cards, top 10 neighborhoods, cluster breakdown, model performance, statistical findings |
| **Neighborhood Data** | Filterable/searchable table with cluster badges, risk levels, inline rate bars |
| **Explore** | Interactive scatter plots (configurable axes), age group distributions by borough, feature histograms, correlation panel, normality tests |
| **Risk Predictor** | Input form for neighborhood metrics, live PyTorch inference, confusion matrix, feature importance, model architecture |
| **Campaign Strategy** | Per-cluster ROI projections with bootstrap CIs, cost vs revenue charts, phased campaign recommendations |

### Frontend / Accessibility

- **AuADHD-friendly design**: Progressive disclosure (collapsible sections), 48px minimum touch targets, `prefers-reduced-motion` media query, high contrast dark theme, consistent color language
- **Responsive**: Sidebar collapses on mobile (<992px)
- **Vega-Lite**: Interactive charts with tooltips, container-width responsive sizing

## SQL-Equivalent Transformations

The data pipeline performs operations equivalent to SQL aggregation and transformation. For example, the feature engineering maps to:

```sql
-- Uninsured rate by neighborhood
SELECT
    neighborhood,
    CAST(SUM(uninsured_male + uninsured_female) AS FLOAT) / NULLIF(total_pop, 0) AS uninsured_rate,
    CAST(poverty_pop AS FLOAT) / NULLIF(poverty_universe, 0) AS poverty_rate,
    CAST(SUM(disabled_male + disabled_female) AS FLOAT) / NULLIF(disability_universe, 0) AS disability_rate
FROM acs_census_data
WHERE name LIKE 'NYC-%'
GROUP BY neighborhood;

-- Borough-level aggregation
SELECT
    borough,
    AVG(uninsured_rate) AS avg_uninsured,
    COUNT(*) AS neighborhood_count
FROM neighborhoods
GROUP BY borough
HAVING COUNT(*) >= 2;

-- Risk classification
SELECT *,
    CASE WHEN uninsured_rate > (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY uninsured_rate) FROM neighborhoods)
         THEN 'HIGH RISK' ELSE 'LOW RISK'
    END AS risk_label
FROM neighborhoods
ORDER BY uninsured_rate DESC;
```

## Project Structure

```
INSURANCEGROWTHANALYTICS/
├── app/
│   ├── app.py                          # Flask app, data pipeline, models, all routes
│   ├── INSURANCEGROWTHANALYTICS.ipynb  # Jupyter notebook (EDA + analysis)
│   ├── templates/
│   │   ├── layout.html                 # Base template with design system
│   │   ├── home.html                   # Dashboard
│   │   ├── data.html                   # Neighborhood data table
│   │   ├── view.html                   # Explore & visualize
│   │   ├── model.html                  # Risk predictor
│   │   └── campaign.html               # Campaign strategy
│   ├── Static/static/images/           # Favicon and static assets
│   └── Tickets/
│       └── Project_Description.MD      # Original project requirements
├── README.md
└── requirements.txt
```

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| **Python** | Entire pipeline — data loading, feature engineering, modeling, web app |
| **PyTorch** | Autoencoder (dimensionality reduction), neural net classifier, LOO-CV training loop |
| **Statistical Analysis** | Shapiro-Wilk, Spearman, Kruskal-Wallis, bootstrap CIs, permutation importance |
| **Customer Segmentation** | K-Means on autoencoder latent space, silhouette optimization, stability analysis |
| **Marketing Analytics** | Campaign ROI projections, phased rollout strategy, cost/revenue modeling |
| **Data Engineering** | Live Census API pipeline, feature engineering from raw ACS tables |
| **SQL** | Equivalent transformations documented (aggregation, filtering, CASE logic) |
| **Visualization** | Vega-Lite interactive charts, responsive design, dark theme |
| **Web Development** | Flask, Jinja2 templates, Bootstrap 5, CSS design system |
| **Accessibility** | AuADHD-friendly UX (progressive disclosure, large targets, reduced motion) |

## Getting Started

```bash
# clone and set up
git clone <repo-url>
cd INSURANCEGROWTHANALYTICS

# create virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install flask pandas numpy torch scikit-learn scipy requests

# run the dashboard
cd app
python app.py
# visit http://localhost:5000
```

The app pulls Census data on startup (requires internet). First load takes ~30 seconds for LOO-CV training across 55 folds.

## Data Source

**U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates, 2022**
- Geographic level: Public Use Microdata Areas (PUMAs) within New York City
- 55 NYC neighborhoods (PUMAs)
- Tables: B27001, B17001, B18101, B01003, B11001

### Limitations

- ACS 5-Year estimates have margins of error (not displayed — available via Census API MOE variables)
- PUMA boundaries don't perfectly align with colloquial neighborhood boundaries
- Small n (55) limits statistical power — LOO-CV and nonparametric tests are appropriate responses
- Campaign ROI assumes $15/contact cost, $6,200 avg plan value, 10% conversion lift (simplified estimates)
