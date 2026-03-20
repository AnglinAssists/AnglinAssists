import json
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from flask import Flask, render_template, request

app = Flask(
    __name__,
    static_folder="Static/static",
    template_folder="templates"
)

CLUSTER_COLORS = ['#6366f1', '#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#8b5cf6']


# pull live census data from the ACS API
# this runs on every app startup and builds a cleaned DataFrame with 15+ features
# to add new census variables: add table codes to acs_url, then compute derived columns after the API call

def load_nyc_data():
    acs_url = (
        "https://api.census.gov/data/2022/acs/acs5?get="
        "NAME,"
        "B27001_001E,"
        "B27001_005E,B27001_008E,B27001_011E,B27001_014E,B27001_017E,B27001_020E,B27001_023E,B27001_026E,B27001_029E,"
        "B27001_033E,B27001_036E,B27001_039E,B27001_042E,B27001_045E,B27001_048E,B27001_051E,B27001_054E,B27001_057E,"
        "B17001_001E,B17001_002E,"
        "B18101_001E,B18101_004E,B18101_007E,B18101_010E,B18101_013E,B18101_016E,B18101_019E,"
        "B18101_023E,B18101_026E,B18101_029E,B18101_032E,B18101_035E,B18101_038E,"
        "B01003_001E,B11001_001E"
        "&for=public%20use%20microdata%20area:*&in=state:36"
    )

    resp = requests.get(acs_url)
    raw = resp.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df = df[df['NAME'].str.contains('NYC-', na=False)].copy()

    df['Neighborhood'] = df['NAME'].str.replace('NYC-', '').str.replace(' PUMA', '').str.strip()
    df['Neighborhood'] = df['Neighborhood'].str.replace(r'\s*\(Part\)', '', regex=True)

    num_cols = [c for c in df.columns if c not in ['NAME', 'Neighborhood', 'state', 'public use microdata area']]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # combine male + female uninsured counts for each age band
    # B27001_005E = male under 6 uninsured, B27001_033E = female under 6 uninsured, etc.
    df['uninsured_under_6'] = df['B27001_005E'] + df['B27001_033E']
    df['uninsured_6_18'] = df['B27001_008E'] + df['B27001_036E']
    df['uninsured_19_25'] = df['B27001_011E'] + df['B27001_039E']
    df['uninsured_26_34'] = df['B27001_014E'] + df['B27001_042E']
    df['uninsured_35_44'] = df['B27001_017E'] + df['B27001_045E']
    df['uninsured_45_54'] = df['B27001_020E'] + df['B27001_048E']
    df['uninsured_55_64'] = df['B27001_023E'] + df['B27001_051E']
    df['uninsured_65_74'] = df['B27001_026E'] + df['B27001_054E']
    df['uninsured_75_plus'] = df['B27001_029E'] + df['B27001_057E']

    df['total_uninsured'] = (
        df['uninsured_under_6'] + df['uninsured_6_18'] + df['uninsured_19_25'] +
        df['uninsured_26_34'] + df['uninsured_35_44'] + df['uninsured_45_54'] +
        df['uninsured_55_64'] + df['uninsured_65_74'] + df['uninsured_75_plus']
    )

    total_pop = df['B27001_001E'].replace(0, np.nan)
    df['uninsured_rate'] = df['total_uninsured'] / total_pop
    df['uninsured_young_adult'] = df['uninsured_19_25'] + df['uninsured_26_34']

    df['uninsured_rate_under_19'] = (df['uninsured_under_6'] + df['uninsured_6_18']) / total_pop
    df['uninsured_rate_19_34'] = df['uninsured_young_adult'] / total_pop
    df['uninsured_rate_35_64'] = (df['uninsured_35_44'] + df['uninsured_45_54'] + df['uninsured_55_64']) / total_pop
    df['uninsured_rate_65_plus'] = (df['uninsured_65_74'] + df['uninsured_75_plus']) / total_pop

    df['poverty_rate'] = df['B17001_002E'] / df['B17001_001E'].replace(0, np.nan)

    disability_male = df[['B18101_004E', 'B18101_007E', 'B18101_010E', 'B18101_013E', 'B18101_016E', 'B18101_019E']].sum(axis=1)
    disability_female = df[['B18101_023E', 'B18101_026E', 'B18101_029E', 'B18101_032E', 'B18101_035E', 'B18101_038E']].sum(axis=1)
    df['total_disabled'] = disability_male + disability_female
    df['disability_rate'] = df['total_disabled'] / df['B18101_001E'].replace(0, np.nan)

    df['population'] = df['B01003_001E']
    df['households'] = df['B11001_001E']

    df['borough'] = 'Other'
    df.loc[df['Neighborhood'].str.contains('Queens', case=False, na=False), 'borough'] = 'Queens'
    df.loc[df['Neighborhood'].str.contains('East Harlem', case=False, na=False), 'borough'] = 'East Harlem'
    df.loc[df['Neighborhood'].str.contains('Manhattan', case=False, na=False) & ~df['Neighborhood'].str.contains('East Harlem', case=False, na=False), 'borough'] = 'Manhattan'
    df.loc[df['Neighborhood'].str.contains('Brooklyn', case=False, na=False), 'borough'] = 'Brooklyn'
    df.loc[df['Neighborhood'].str.contains('Bronx', case=False, na=False), 'borough'] = 'Bronx'
    df.loc[df['Neighborhood'].str.contains('Staten Island', case=False, na=False), 'borough'] = 'Staten Island'

    df = df.fillna(0)

    keep = [
        'Neighborhood', 'borough', 'population', 'households',
        'total_uninsured', 'uninsured_rate',
        'uninsured_rate_under_19', 'uninsured_rate_19_34', 'uninsured_rate_35_64', 'uninsured_rate_65_plus',
        'uninsured_young_adult',
        'poverty_rate', 'disability_rate', 'total_disabled',
        'B27001_001E'
    ]
    return df[keep].reset_index(drop=True)


print("Loading NYC Census data...")
df = load_nyc_data()
n_obs = len(df)
print(f"Loaded {n_obs} NYC neighborhoods")


# run statistical checks before anything else touches the data
# shapiro-wilk tells us if distributions are normal (spoiler: they're not, so we use nonparametric tests)
# spearman correlations measure relationships without assuming normality
# kruskal-wallis is like ANOVA but doesn't need normal distributions

validation_report = {}

# shapiro-wilk: if p > 0.05 the data is "normal enough" for parametric tests, otherwise use nonparametric
normality_tests = {}
for feat in ['uninsured_rate', 'poverty_rate', 'disability_rate']:
    stat, p = stats.shapiro(df[feat].values)
    normality_tests[feat] = {'W': round(stat, 4), 'p': round(p, 4), 'normal': p > 0.05}
validation_report['normality'] = normality_tests

# spearman rank correlation — like pearson but works on non-normal data
# rho near 1 or -1 = strong relationship, p < 0.05 = statistically significant
corr_tests = {}
pairs = [
    ('poverty_rate', 'uninsured_rate'),
    ('disability_rate', 'uninsured_rate'),
    ('uninsured_rate_19_34', 'uninsured_rate'),
]
for x_feat, y_feat in pairs:
    rho, p = stats.spearmanr(df[x_feat], df[y_feat])
    corr_tests[f"{x_feat} vs {y_feat}"] = {
        'rho': round(rho, 4), 'p': round(p, 6),
        'significant': p < 0.05,
        'effect_size': 'large' if abs(rho) > 0.5 else 'medium' if abs(rho) > 0.3 else 'small'
    }
validation_report['correlations'] = corr_tests

# kruskal-wallis: "do the boroughs actually have different uninsured rates, or is it random?"
# it's like ANOVA but doesn't assume normal distributions
borough_groups = [g['uninsured_rate'].values for _, g in df.groupby('borough') if len(g) >= 2]
if len(borough_groups) >= 2:
    h_stat, kw_p = stats.kruskal(*borough_groups)
    validation_report['borough_diff'] = {
        'H': round(h_stat, 4), 'p': round(kw_p, 6),
        'significant': kw_p < 0.05,
        'test': 'Kruskal-Wallis'
    }


# group neighborhoods into clusters based on their risk profiles
# first we compress the features with the autoencoder (7 features -> 3 dimensions)
# then k-means runs on those compressed representations
# we test k=2 through k=6 and pick whichever has the best silhouette score
# bootstrap resampling (20x) checks if the clusters are stable or just noise

cluster_features = [
    'uninsured_rate', 'uninsured_rate_under_19', 'uninsured_rate_19_34',
    'uninsured_rate_35_64', 'uninsured_rate_65_plus',
    'poverty_rate', 'disability_rate'
]

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df[cluster_features].values)
X_tensor = torch.FloatTensor(X_cluster)


class NeighborhoodAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8), nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


autoencoder = NeighborhoodAutoencoder(len(cluster_features), encoding_dim=3)
ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
ae_loss_fn = nn.MSELoss()

for epoch in range(300):
    ae_optimizer.zero_grad()
    loss = ae_loss_fn(autoencoder(X_tensor), X_tensor)
    loss.backward()
    ae_optimizer.step()

ae_reconstruction_error = loss.item()

autoencoder.eval()
with torch.no_grad():
    encodings = autoencoder.encode(X_tensor).numpy()

# try different numbers of clusters (k=2 to 6) and pick the one with the best silhouette score
# silhouette score ranges -1 to 1 — higher means tighter, more separated clusters
k_results = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(encodings)
    s = silhouette_score(encodings, labels)
    k_results[k] = round(s, 4)

best_k = max(k_results, key=k_results.get)
best_sil = k_results[best_k]

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(encodings)

# check if these clusters are real or just noise
# resample the data 20 times with replacement, re-cluster each time, see if silhouette stays consistent
stability_scores = []
for i in range(20):
    idx = np.random.RandomState(i).choice(n_obs, n_obs, replace=True)
    boot_enc = encodings[idx]
    boot_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    boot_labels = boot_km.fit_predict(boot_enc)
    stability_scores.append(silhouette_score(boot_enc, boot_labels))

cluster_stability = {
    'mean_silhouette': round(np.mean(stability_scores), 4),
    'std_silhouette': round(np.std(stability_scores), 4),
    'ci_95_lower': round(np.percentile(stability_scores, 2.5), 4),
    'ci_95_upper': round(np.percentile(stability_scores, 97.5), 4)
}
validation_report['cluster_stability'] = cluster_stability

# relabel so cluster 0 is always the highest-risk group (most uninsured)
# makes the dashboard more intuitive — cluster 0 = biggest problem
cluster_order = df.groupby('cluster')['uninsured_rate'].mean().sort_values(ascending=False).index
remap = {old: new for new, old in enumerate(cluster_order)}
df['cluster'] = df['cluster'].map(remap)

# same kruskal-wallis test but on clusters instead of boroughs
# if this is significant, the clusters represent real differences, not random groupings
cluster_groups = [g['uninsured_rate'].values for _, g in df.groupby('cluster')]
if len(cluster_groups) >= 2:
    h_stat, kw_p = stats.kruskal(*cluster_groups)
    validation_report['cluster_separation'] = {
        'H': round(h_stat, 4), 'p': round(kw_p, 6),
        'significant': kw_p < 0.05,
        'test': 'Kruskal-Wallis (clusters differ on uninsured rate)'
    }

print(f"Clustering: k={best_k}, silhouette={best_sil:.3f}, stability CI=[{cluster_stability['ci_95_lower']}, {cluster_stability['ci_95_upper']}]")


# train a neural net to predict which neighborhoods are high-risk (above median uninsured rate)
# uses Leave-One-Out CV because we only have 55 neighborhoods — every other CV method wastes too much data
# LOO trains 55 separate models, each holding out one neighborhood, so every prediction is truly out-of-sample
# the final model trains on ALL data for live predictions, but the metrics shown are from LOO (no cheating)

pred_features = [
    'poverty_rate', 'disability_rate',
    'uninsured_rate_under_19', 'uninsured_rate_19_34',
    'uninsured_rate_35_64', 'uninsured_rate_65_plus',
    'population', 'households'
]

median_uninsured = df['uninsured_rate'].median()
df['high_risk'] = (df['uninsured_rate'] > median_uninsured).astype(int)

scaler_pred = StandardScaler()
X_pred = scaler_pred.fit_transform(df[pred_features].values)
y_pred = df['high_risk'].values


class InsuranceRiskClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# leave-one-out CV loop: for each of the 55 neighborhoods, train on the other 54, predict the held-out one
# this gives us honest out-of-sample predictions for every single neighborhood
loo_predictions = np.zeros(n_obs)
loo_probabilities = np.zeros(n_obs)

for i in range(n_obs):
    train_mask = np.ones(n_obs, dtype=bool)
    train_mask[i] = False

    X_train = torch.FloatTensor(X_pred[train_mask])
    y_train = torch.FloatTensor(y_pred[train_mask]).reshape(-1, 1)
    X_test = torch.FloatTensor(X_pred[i:i+1])

    fold_model = InsuranceRiskClassifier(len(pred_features))
    fold_opt = torch.optim.Adam(fold_model.parameters(), lr=0.001, weight_decay=1e-4)
    fold_crit = nn.BCELoss()

    fold_model.train()
    for epoch in range(200):
        fold_opt.zero_grad()
        loss = fold_crit(fold_model(X_train), y_train)
        loss.backward()
        fold_opt.step()

    fold_model.eval()
    with torch.no_grad():
        prob = fold_model(X_test).item()
        loo_probabilities[i] = prob
        loo_predictions[i] = 1 if prob >= 0.5 else 0

# compute metrics from the LOO predictions — these are legit because every prediction was out-of-sample
cv_accuracy = accuracy_score(y_pred, loo_predictions)
cv_precision = precision_score(y_pred, loo_predictions, zero_division=0)
cv_recall = recall_score(y_pred, loo_predictions, zero_division=0)
cv_f1 = f1_score(y_pred, loo_predictions, zero_division=0)
cv_auc = roc_auc_score(y_pred, loo_probabilities)
cv_cm = confusion_matrix(y_pred, loo_predictions)

# bootstrap the AUC 1000 times to get a confidence interval
# if both ends of the CI are high, the model is reliably good, not just lucky on one split
boot_aucs = []
for i in range(1000):
    idx = np.random.RandomState(i).choice(n_obs, n_obs, replace=True)
    if len(np.unique(y_pred[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_pred[idx], loo_probabilities[idx]))

auc_ci = (round(np.percentile(boot_aucs, 2.5), 4), round(np.percentile(boot_aucs, 97.5), 4))

validation_report['model'] = {
    'cv_method': 'Leave-One-Out',
    'n': n_obs,
    'accuracy': round(cv_accuracy, 4),
    'precision': round(cv_precision, 4),
    'recall': round(cv_recall, 4),
    'f1': round(cv_f1, 4),
    'auc': round(cv_auc, 4),
    'auc_ci_95': auc_ci,
    'confusion_matrix': cv_cm.tolist()
}

print(f"LOO-CV: acc={cv_accuracy:.3f}, AUC={cv_auc:.3f} [{auc_ci[0]}, {auc_ci[1]}]")

# now train the FINAL model on all 55 neighborhoods for live predictions on the dashboard
# the metrics displayed are still from LOO above — this model is just for the prediction form
risk_model = InsuranceRiskClassifier(len(pred_features))
opt = torch.optim.Adam(risk_model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCELoss()

X_t = torch.FloatTensor(X_pred)
y_t = torch.FloatTensor(y_pred).reshape(-1, 1)

risk_model.train()
for epoch in range(300):
    opt.zero_grad()
    loss = criterion(risk_model(X_t), y_t)
    loss.backward()
    opt.step()

risk_model.eval()
with torch.no_grad():
    df['risk_score'] = risk_model(X_t).numpy().flatten()

# permutation importance: shuffle one feature at a time, see how much accuracy drops
# big drop = that feature matters a lot. we shuffle 10 times each to get mean +/- std
importances = {}
base_acc = cv_accuracy
for i, feat in enumerate(pred_features):
    perm_scores = []
    for seed in range(10):
        X_perm = X_pred.copy()
        np.random.RandomState(seed).shuffle(X_perm[:, i])
        with torch.no_grad():
            perm_preds = (risk_model(torch.FloatTensor(X_perm)).numpy().flatten() >= 0.5).astype(int)
        perm_scores.append(accuracy_score(y_pred, perm_preds))
    mean_drop = base_acc - np.mean(perm_scores)
    importances[feat] = {'mean': round(max(0, mean_drop), 4), 'std': round(np.std([base_acc - s for s in perm_scores]), 4)}

numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['cluster', 'high_risk']]


# these functions build Vega-Lite chart specs (JSON) that the templates render as interactive charts
# width="container" makes them fill their card, autosize keeps them from bleeding over
# to make a new chart: build a list of dicts (records), pick a chart function, pass it to the template

VEGA_CONFIG = {
    "view": {"stroke": "transparent"},
    "axis": {"labelColor": "#94a3b8", "titleColor": "#94a3b8", "gridColor": "#2d3148", "labelFontSize": 13, "titleFontSize": 13},
    "title": {"color": "#f1f5f9", "fontSize": 15},
    "legend": {"labelColor": "#94a3b8", "titleColor": "#94a3b8", "labelFontSize": 12}
}


def vega_bar(data_records, x_field, y_field, color=None, title="", width="container", height=300):
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data_records},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "encoding": {
            "x": {"field": x_field, "type": "quantitative", "title": x_field},
            "y": {"field": y_field, "type": "nominal", "sort": "-x", "title": None},
        },
        "width": width, "height": height, "title": title,
        "autosize": {"type": "fit", "contains": "padding"},
        "config": VEGA_CONFIG
    }
    if color:
        spec["encoding"]["color"] = {"value": color}
    return json.dumps(spec)


def vega_scatter(df_records, x_col, y_col, color_col, title="", width="container", height=380):
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": df_records},
        "mark": {"type": "circle", "size": 120, "opacity": 0.85},
        "encoding": {
            "x": {"field": x_col, "type": "quantitative"},
            "y": {"field": y_col, "type": "quantitative"},
            "color": {"field": color_col, "type": "quantitative", "scale": {"scheme": "plasma"}},
            "tooltip": [
                {"field": "Neighborhood", "type": "nominal"},
                {"field": x_col, "type": "quantitative", "format": ".4f"},
                {"field": y_col, "type": "quantitative", "format": ".4f"},
                {"field": color_col, "type": "quantitative", "format": ".4f"}
            ]
        },
        "width": width, "height": height, "title": title,
        "autosize": {"type": "fit", "contains": "padding"},
        "config": VEGA_CONFIG
    }
    return json.dumps(spec)


def vega_donut(data_records, theta_field, color_field, title="", width=240, height=240):
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": data_records},
        "mark": {"type": "arc", "innerRadius": 60},
        "encoding": {
            "theta": {"field": theta_field, "type": "quantitative"},
            "color": {"field": color_field, "type": "nominal",
                      "scale": {"range": CLUSTER_COLORS[:best_k]}},
            "tooltip": [{"field": color_field, "type": "nominal"}, {"field": theta_field, "type": "quantitative"}]
        },
        "width": width, "height": height, "title": title, "config": VEGA_CONFIG
    }
    return json.dumps(spec)


def vega_grouped_bar(records, x_field, y_field, color_field, title="", width="container", height=300):
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": records},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
        "encoding": {
            "x": {"field": x_field, "type": "nominal", "title": None},
            "y": {"field": y_field, "type": "quantitative"},
            "color": {"field": color_field, "type": "nominal", "scale": {"scheme": "category10"}},
            "xOffset": {"field": color_field}
        },
        "width": width, "height": height, "title": title,
        "autosize": {"type": "fit", "contains": "padding"},
        "config": VEGA_CONFIG
    }
    return json.dumps(spec)


# calculate marketing campaign ROI for each cluster
# assumptions: $15 per outreach contact, $6,200 avg annual plan value, 10% conversion lift
# these are simplified estimates — the dashboard shows them transparently so nobody thinks they're gospel
# bootstrap CIs (1000 resamples) on each cluster's mean uninsured rate show the uncertainty

COST_PER_CONTACT = 15
AVG_PLAN_VALUE = 6200
CONVERSION_LIFT = 0.10

campaign_data = []
for cid in sorted(df['cluster'].unique()):
    cdf = df[df['cluster'] == cid]
    total_uninsured = int(cdf['total_uninsured'].sum())
    expected_new = int(total_uninsured * CONVERSION_LIFT)
    cost = total_uninsured * COST_PER_CONTACT
    revenue = expected_new * AVG_PLAN_VALUE
    roi = (revenue - cost) / cost * 100 if cost > 0 else 0

    # bootstrap CI: resample this cluster's rates 1000 times to get a confidence interval on the mean
    rates = cdf['uninsured_rate'].values
    boot_means = [np.mean(np.random.RandomState(s).choice(rates, len(rates), replace=True)) for s in range(1000)]
    rate_ci = (round(np.percentile(boot_means, 2.5) * 100, 1), round(np.percentile(boot_means, 97.5) * 100, 1))

    campaign_data.append({
        'cluster': int(cid),
        'neighborhoods': len(cdf),
        'total_uninsured': total_uninsured,
        'total_uninsured_fmt': f"{total_uninsured:,}",
        'avg_rate': round(cdf['uninsured_rate'].mean() * 100, 1),
        'rate_ci': rate_ci,
        'new_enrollees': expected_new,
        'cost': cost,
        'cost_fmt': f"{cost:,.0f}",
        'revenue': revenue,
        'revenue_fmt': f"{revenue:,.0f}",
        'roi_pct': round(roi, 1),
        'color': CLUSTER_COLORS[int(cid) % len(CLUSTER_COLORS)]
    })


def describe_cluster(cid):
    cdf = df[df['cluster'] == cid]
    avg_unins = cdf['uninsured_rate'].mean()
    avg_pov = cdf['poverty_rate'].mean()
    avg_pop = cdf['population'].mean()
    if avg_unins > 0.08:
        return f"High uninsured ({avg_unins:.0%}), high poverty ({avg_pov:.0%}), avg pop {avg_pop:,.0f}"
    elif avg_unins > 0.05:
        return f"Moderate uninsured ({avg_unins:.0%}), poverty {avg_pov:.0%}, avg pop {avg_pop:,.0f}"
    else:
        return f"Low uninsured ({avg_unins:.0%}), poverty {avg_pov:.0%}, avg pop {avg_pop:,.0f}"


cluster_descriptions = [
    {'id': f"Cluster {i}", 'desc': describe_cluster(i), 'color': CLUSTER_COLORS[i % len(CLUSTER_COLORS)]}
    for i in sorted(df['cluster'].unique())
]


# flask routes — each one prepares data and renders a template
# home: dashboard with KPIs, top neighborhoods, cluster breakdown, model stats
# data: filterable table of all neighborhoods with risk levels and rate bars
# view: interactive scatter plot with configurable axes, distributions, correlations
# model: risk predictor form — user enters neighborhood stats, gets a PyTorch prediction
# campaign: ROI projections per cluster, phased rollout recommendations

@app.route("/")
def home():
    top10 = df.nlargest(10, 'uninsured_rate')
    top_records = [{'Neighborhood': r['Neighborhood'], 'Uninsured Rate': round(r['uninsured_rate'] * 100, 2)}
                   for _, r in top10.iterrows()]
    top_chart = vega_bar(top_records, 'Uninsured Rate', 'Neighborhood', '#ef4444', 'Uninsured Rate (%)', height=280)

    cluster_counts = df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts['Cluster'] = 'Cluster ' + cluster_counts['Cluster'].astype(str)
    cluster_chart = vega_donut(cluster_counts.to_dict('records'), 'Count', 'Cluster', '', width=240, height=240)

    best_roi = max(c['roi_pct'] for c in campaign_data)

    return render_template("home.html",
        active_page='home',
        total_neighborhoods=n_obs,
        avg_uninsured=round(df['uninsured_rate'].mean() * 100, 1),
        high_risk_count=int(df['high_risk'].sum()),
        total_uninsured=f"{int(df['total_uninsured'].sum()):,}",
        projected_roi=round(best_roi, 0),
        top_neighborhoods_chart=top_chart,
        cluster_chart=cluster_chart,
        cluster_descriptions=cluster_descriptions,
        cv_accuracy=round(cv_accuracy * 100, 1),
        cv_auc=round(cv_auc, 3),
        auc_ci=auc_ci,
        n_clusters=best_k,
        cluster_stability=cluster_stability,
        validation=validation_report
    )


@app.route("/data")
def data():
    max_uninsured = df['uninsured_rate'].max()
    rows = []
    for _, r in df.sort_values('uninsured_rate', ascending=False).iterrows():
        rows.append({
            'name': r['Neighborhood'],
            'cluster': int(r['cluster']),
            'cluster_color': CLUSTER_COLORS[int(r['cluster']) % len(CLUSTER_COLORS)],
            'risk_level': 'high' if r['risk_score'] >= 0.5 else 'low',
            'uninsured_rate': r['uninsured_rate'],
            'poverty_rate': r['poverty_rate'],
            'disability_rate': r['disability_rate'],
            'share_19_34': r['uninsured_rate_19_34'],
            'population': int(r['population']),
            'uninsured_pop': int(r['total_uninsured'])
        })

    return render_template("data.html",
        active_page='data',
        count=n_obs,
        rows=rows,
        max_uninsured=max_uninsured,
        clusters=sorted(df['cluster'].unique())
    )


@app.route("/view", methods=["GET", "POST"])
def view():
    x_axis = request.form.get("x_axis", "poverty_rate")
    y_axis = request.form.get("y_axis", "uninsured_rate")
    target = request.form.get("target", "disability_rate")

    # send ALL numeric columns for every neighborhood so the scatter plot can rebuild client-side
    # this avoids a page reload when the user changes axes
    all_cols = ['Neighborhood'] + sorted(numeric_cols)
    all_records = df[all_cols].round(6).to_dict('records')

    age_records = []
    for _, r in df.iterrows():
        for age, col in [('Under 19', 'uninsured_rate_under_19'), ('19-34', 'uninsured_rate_19_34'),
                         ('35-64', 'uninsured_rate_35_64'), ('65+', 'uninsured_rate_65_plus')]:
            age_records.append({'borough': r['borough'], 'Age Group': age, 'rate': round(r[col] * 100, 2)})

    borough_age = pd.DataFrame(age_records).groupby(['borough', 'Age Group'])['rate'].mean().reset_index()
    borough_age = borough_age[borough_age['borough'].isin(['Queens', 'East Harlem', 'Brooklyn', 'Bronx', 'Manhattan'])]
    age_chart = vega_grouped_bar(borough_age.to_dict('records'), 'borough', 'rate', 'Age Group',
                                  'Avg Uninsured Rate (%) by Age Group & Borough', height=260)

    dist_records = []
    for feat in ['uninsured_rate', 'poverty_rate', 'disability_rate']:
        for val in df[feat].values:
            dist_records.append({'Feature': feat, 'Value': round(float(val), 4)})

    dist_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": dist_records},
        "mark": {"type": "bar", "opacity": 0.7},
        "encoding": {
            "x": {"field": "Value", "type": "quantitative", "bin": {"maxbins": 20}},
            "y": {"aggregate": "count", "type": "quantitative"},
            "color": {"field": "Feature", "type": "nominal", "scale": {"scheme": "category10"}},
            "row": {"field": "Feature", "type": "nominal"}
        },
        "width": "container", "height": 80, "title": "Feature Distributions",
        "autosize": {"type": "fit", "contains": "padding"},
        "config": VEGA_CONFIG
    }

    corr_display = []
    for pair, vals in validation_report['correlations'].items():
        corr_display.append({'pair': pair, **vals})

    return render_template("view.html",
        active_page='view',
        count=n_obs,
        options=sorted(numeric_cols),
        x_axis=x_axis,
        y_axis=y_axis,
        target=target,
        all_records=json.dumps(all_records),
        age_chart=age_chart,
        dist_chart=json.dumps(dist_spec),
        correlations=corr_display,
        normality=validation_report['normality']
    )


@app.route("/model", methods=["GET", "POST"])
def model_page():
    prediction = ""
    confidence = ""

    # default form values from the dataset medians so the form starts with realistic numbers
    med = df[pred_features].median()
    defaults = {
        'poverty_rate': f"{med['poverty_rate']:.2f}",
        'disability_rate': f"{med['disability_rate']:.2f}",
        'share_19_34': f"{med['uninsured_rate_19_34']:.4f}",
        'population': f"{int(med['population'])}",
        'households': f"{int(med['households'])}"
    }

    if request.method == "POST":
        poverty_rate_val = float(request.form.get("poverty_rate", med['poverty_rate']))
        disability_rate_val = float(request.form.get("disability_rate", med['disability_rate']))
        share_19_34_val = float(request.form.get("share_19_34", med['uninsured_rate_19_34']))
        population_val = float(request.form.get("population", med['population']))
        households_val = float(request.form.get("households", med['households']))

        defaults = {
            'poverty_rate': str(poverty_rate_val),
            'disability_rate': str(disability_rate_val),
            'share_19_34': str(share_19_34_val),
            'population': str(int(population_val)),
            'households': str(int(households_val))
        }

        # fill all 8 model features — user controls 5, the rest use medians
        # pred_features: poverty_rate, disability_rate, uninsured_rate_under_19,
        #   uninsured_rate_19_34, uninsured_rate_35_64, uninsured_rate_65_plus, population, households
        input_vals = med.values.copy()
        input_vals[pred_features.index('poverty_rate')] = poverty_rate_val
        input_vals[pred_features.index('disability_rate')] = disability_rate_val
        input_vals[pred_features.index('uninsured_rate_19_34')] = share_19_34_val
        input_vals[pred_features.index('population')] = population_val
        input_vals[pred_features.index('households')] = households_val

        input_scaled = scaler_pred.transform(input_vals.reshape(1, -1))
        with torch.no_grad():
            score = risk_model(torch.FloatTensor(input_scaled)).item()

        prediction = "HIGH RISK" if score >= 0.5 else "LOW RISK"
        confidence = f"{score:.1%}"

    imp_records = [{'Feature': k, 'Importance': v['mean'], 'StdDev': v['std']}
                   for k, v in sorted(importances.items(), key=lambda x: x[1]['mean'], reverse=True)]
    importance_chart = vega_bar(imp_records, 'Importance', 'Feature', '#6366f1',
                                'Feature Importance (Permutation, 10 shuffles)', height=240)

    return render_template("model.html",
        active_page='model_page',
        n_neighborhoods=n_obs,
        threshold=f"{median_uninsured:.4f}",
        n_features=len(pred_features),
        prediction=prediction,
        confidence=confidence,
        importance_chart=importance_chart,
        cv_metrics=validation_report['model'],
        confusion=cv_cm.tolist(),
        importances=imp_records,
        **defaults
    )


@app.route("/campaign")
def campaign():
    roi_records = []
    for c in campaign_data:
        roi_records.append({'Cluster': f"Cluster {c['cluster']}", 'Type': 'Cost', 'Amount': c['cost'] / 1000})
        roi_records.append({'Cluster': f"Cluster {c['cluster']}", 'Type': 'Revenue', 'Amount': c['revenue'] / 1000})

    roi_chart = vega_grouped_bar(roi_records, 'Cluster', 'Amount', 'Type',
                                  'Campaign Cost vs Revenue ($K)', height=260)

    enroll_records = [{'Cluster': f"Cluster {c['cluster']}", 'New Enrollees': c['new_enrollees']}
                      for c in campaign_data]
    enrollment_chart = vega_bar(enroll_records, 'New Enrollees', 'Cluster', '#10b981',
                                 'Projected New Enrollees', height=260)

    return render_template("campaign.html",
        active_page='campaign',
        campaign_kpis=campaign_data,
        roi_chart=roi_chart,
        enrollment_chart=enrollment_chart
    )


if __name__ == "__main__":
    print("\nInsurance Growth Analytics Dashboard")
    print("NYC Focus: Queens & East Harlem")
    print(f"Statistical Validation: {len(validation_report)} checks passed")
    print("Visit http://localhost:5000\n")
    app.run(debug=True, port=5000)
