# %%
"""
# Shape Tech-Test

Author: Jonatas Cesar 
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from pandas_profiling import ProfileReport
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# %%
#%matplotlib inline

# %%
"""
## 0 - Data profiling
"""

# %%
df = pd.read_excel("O&G Equipment Data.xlsx", index_col="Cycle")
df[["Preset_1", "Preset_2"]] = df[["Preset_1", "Preset_2"]].astype("category")

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
## 1 - Calculating how many times the equiment has failed
"""

# %%
pd.DataFrame(df.Fail.value_counts())

# %%
"""
## 2 - Categorize equipment failures by setups configurations (preset 1 and preset 2)
"""

# %%


def eval_presetfail(df, preset_col):
    preset_fail = (
        pd.DataFrame(df.loc[:, [preset_col, "Fail"]].value_counts(), columns=["Counts"])
        .reset_index()
        .pivot(index=preset_col, columns="Fail", values="Counts")
    )

    preset_fail.columns = ["Not Fail Count", "Fail Count"]

    preset_fail["Fail rate"] = preset_fail.loc[:, "Fail Count"] / (
        preset_fail.loc[:, "Not Fail Count"] + preset_fail.loc[:, "Fail Count"]
    )
    return preset_fail


# %%
eval_presetfail(df, "Preset_1")


# %%
eval_presetfail(df, "Preset_2")


# %%
"""
## 3 – Categorize equipment failures by their nature/root cause according to parameter readings
"""


# %%
parameter_readings = [
    "Temperature",
    "Pressure",
    "VibrationX",
    "VibrationY",
    "VibrationZ",
    "Frequency",
]

# %%
sns.pairplot(df.loc[:, parameter_readings + ["Fail"]], hue="Fail")
plt.show()

# %%

mean_readings = df.groupby("Fail").mean().T
mean_readings.columns = ["Not Fail Avg", "Fail Avg"]
mean_readings["Avg Increase (%)"] = 100 * (
    mean_readings["Fail Avg"] / mean_readings["Not Fail Avg"] - 1
)
mean_readings

# %%
"""
## 4 - Create a model using the technique you think is most appropriate and measure its performance 
"""

# %%
"""
## 4.1 - Split Data
"""

# %%
X, y = df.iloc[:, :-1], df.iloc[:, -1]

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
"""
### 4.2 - Model Fit 
"""

# %%
model = CatBoostClassifier()

# %%
cat_features = ["Preset_1", "Preset_2"]

# %%
model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    verbose=False,
)

# %%
"""
## 4.3 - Classification perfomance
"""

# %%
y_pred_prob = model.predict(X_test, prediction_type="Probability")[:, 1]

# %%
RocCurveDisplay.from_predictions(y_test.values, y_pred_prob)
plt.show()

# %%
y_pred = y_pred_prob > 0.5

# %%
print(classification_report(y_test.values, y_pred))

# %%
"""
## 5 - Analyze variable importance
"""

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Pool(X_train, y_train, cat_features=cat_features))
shap.summary_plot(shap_values, X_train)

# %%
