
## Installing SHAP for Causal-SHAP and G-SHAP algorithms
# !pip install dowhy shap -q


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import gshap

# =========================================================
# [Step 0] Feature / Target 준비
# =========================================================
target_variable = 'target_RUL'


feature_cols = feature_list

X_train_model = train_df[feature_cols].dropna().copy()
y_train_model = train_df.loc[X_train_model.index, target_variable].copy()

X_test_model = test_df[feature_cols].dropna().copy()
y_test_model = test_df.loc[X_test_model.index, target_variable].copy()

print("X_train shape:", X_train_model.shape)
print("X_test shape :", X_test_model.shape)

# =========================================================
# [Step 1] XGBRegressor 학습
# =========================================================
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda'   # GPU 없으면 제거
)

xgb_model.fit(
    X_train_model,
    y_train_model,
    eval_set=[(X_test_model, y_test_model)],
    verbose=False
)

print("Model training completed.")

# =========================================================
# [Step 2] 예측 함수 정의
# =========================================================
def f(x):
    x_df = pd.DataFrame(x, columns=feature_cols)
    return xgb_model.predict(x_df)   # 1D vector

# =========================================================
# [Step 3] generalized function g 정의
# "낮은 predicted life ratio 비율"의 평균 -> scalar
# =========================================================
threshold = np.percentile(y_train_model, 25)
print("Low life-ratio threshold:", threshold)

def g(y_pred):
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_pred < threshold)   # scalar

# =========================================================
# [Step 4] Background / Evaluation data
# =========================================================
background = X_train_model.sample(min(200, len(X_train_model)), random_state=42)
X_eval = X_test_model.iloc[:50].copy()

# =========================================================
# [Step 5] G-SHAP 계산
# =========================================================
explainer = gshap.KernelExplainer(
    model=f,
    data=background.values,
    g=g
)

g_shap_values = explainer.gshap_values(
    X_eval.values,
    nsamples=100
)

print("g_shap_values shape:", np.shape(g_shap_values))

# 혹시 환경에 따라 (n_features, 1) 형태면 1D로 평탄화
g_shap_values = np.asarray(g_shap_values).squeeze()

print("after squeeze:", g_shap_values.shape)

# =========================================================
# [Step 6] 결과 정리
# =========================================================
gshap_df = pd.DataFrame({
    "feature": feature_cols,
    "g_shap_value": g_shap_values
}).sort_values("g_shap_value", key=np.abs, ascending=False)

print(gshap_df)

# =========================================================
# [Step 7] 시각화
# =========================================================
plt.figure(figsize=(10, 7))
plt.barh(gshap_df["feature"], gshap_df["g_shap_value"])
plt.gca().invert_yaxis()
plt.title("G-SHAP: Contribution to Low Predicted Life Ratio Rate")
plt.xlabel("G-SHAP value")
plt.show()



