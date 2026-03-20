### Neighborhood Marketing Strategy for Health Insurance Plans

### Description
Use customer segmentation and socio-economic data to design a targeted marketing campaign aimed at increasing health insurance uptake in NYC neighborhoods with low enrollment.

### Dataset
Live data from the U.S. Census Bureau ACS 5-Year API (2022). Tables used:
- B27001: Health insurance coverage by age and sex
- B17001: Poverty status (18-64)
- B18101: Disability status by age and sex
- B01003: Total population
- B11001: Total households

Geographic level: 55 NYC Public Use Microdata Areas (PUMAs)

### Technical Deliverables

**Customer Segmentation**
- K-Means clustering on PyTorch autoencoder latent representations (7 features compressed to 3D)
- Silhouette-optimized k selection (k=2 through 6)
- Bootstrap stability analysis (20 resamples with 95% CI)
- Kruskal-Wallis test confirming clusters differ significantly

**Deep Learning (PyTorch)**
- Autoencoder for nonlinear dimensionality reduction of neighborhood features
- Binary classifier (3-layer neural net with dropout) for risk prediction
- Leave-One-Out Cross-Validation (55 folds) for honest out-of-sample evaluation
- Bootstrap confidence intervals (1000 resamples) on AUC
- Permutation feature importance (10 shuffles per feature)

**Statistical Analysis**
- Shapiro-Wilk normality tests driving method selection
- Spearman rank correlations with p-values and effect sizes
- Kruskal-Wallis H-tests for borough and cluster group comparisons
- Bootstrap confidence intervals on cluster mean rates

**Interactive Dashboard (Flask + Vega-Lite)**
- 5 pages: Dashboard, Neighborhood Data, Explore, Risk Predictor, Campaign Strategy
- Filterable/searchable data table with cluster badges and risk indicators
- Interactive scatter plots with configurable axes
- Age group distributions by borough (grouped bar charts)
- Feature distribution histograms
- Live PyTorch inference from form inputs
- Campaign ROI projections per cluster with phased recommendations

**Accessibility**
- AuADHD-friendly design: progressive disclosure, 48px touch targets, reduced motion support
- High contrast dark theme with consistent color language
- Responsive layout (sidebar collapses on mobile)

**Data Engineering**
- Live Census API pipeline (no static CSVs)
- 15+ engineered features from raw ACS table codes
- SQL-equivalent transformations documented in README

### Skills Showcased
- Customer segmentation (K-Means on autoencoder embeddings)
- Python (pandas, numpy, scipy, scikit-learn, PyTorch, Flask)
- Deep Learning (autoencoder, neural net classifier, LOO-CV)
- Statistical Analysis (Shapiro-Wilk, Spearman, Kruskal-Wallis, bootstrap CIs)
- Marketing Analytics (campaign ROI projections, phased rollout strategy)
- Visualization (Vega-Lite interactive charts, responsive dark theme)
- SQL (equivalent transformations documented)
- Data Engineering (live API pipeline, feature engineering)
- Web Development (Flask, Jinja2, Bootstrap 5)
- Accessibility (AuADHD-friendly UX design)

### Technologies and Tools
- Languages: Python
- Libraries: pandas, numpy, scipy, scikit-learn, PyTorch, Flask, requests
- Visualization: Vega-Lite (interactive, responsive charts)
- Frontend: Bootstrap 5, custom CSS design system
- Data Source: U.S. Census Bureau ACS API
- Validation: LOO-CV, bootstrap, nonparametric hypothesis tests
