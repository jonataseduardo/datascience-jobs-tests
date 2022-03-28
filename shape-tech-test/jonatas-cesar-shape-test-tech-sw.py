# %%
"""
# Shape Tech-Test

Author: Jonatas Cesar 
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, RocCurveDisplay
from catboost import CatBoostClassifier, Pool
import shap
import pandas
from pandas_profiling import ProfileReport

# %%
#%matplotlib inline

# %%
df = pd.read_excel("O&G Equipment Data.xlsx", index_col="Cycle")

# %%
df.describe().T

# %%
# check imbalanced data
df.Fail.mean()

# %%
profile = ProfileReport(df, title="Shape Data - Profiling Report", minimal=True)

# %%
profile.to_widgets()

# %%
"""
## Split Data
"""

# %%
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
"""
## Model Fit 
"""

# %%
model = CatBoostClassifier()

# %%
cat_features = ["Preset_1", "Preset_2"]

# %%
sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

# %%
class_weight = {
    True: sample_weight_train[~y_train.values][0],
    False: sample_weight_train[y_train.values][0],
}

# %%
sample_weight_test = compute_sample_weight(class_weight=class_weight, y=y_test)

# %%
model.fit(
    X_train,
    y_train,
    sample_weight=train_sample_weight,
    cat_features=cat_features,
    verbose=False,
)

# %%
"""
## Classification perfomance
"""

# %%
y_pred_prob = model.predict(X_test, prediction_type="Probability")[:, 1]

# %%
RocCurveDisplay.from_predictions(
    y_test.values, y_pred_prob, sample_weight=sample_weight_test
)
plt.show()

# %%
y_pred = y_pred_prob > 0.5

# %%
print(classification_report(y_test.values, y_pred))

# %%
"""
## Feature importance 
"""

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=cat_features))
shap.summary_plot(shap_values, X_train)

# %%
