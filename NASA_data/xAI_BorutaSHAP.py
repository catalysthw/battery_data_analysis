
## Installing BorutaSHAP
# !pip install BorutaShap -q


import scipy.stats
import BorutaShap as borutashap_module   # 클래스 말고 모듈 자체를 import

# SciPy binomtest wrapper
def fixed_binom_test(k, n, p=0.5, alternative='two-sided'):
    return scipy.stats.binomtest(
        k=int(round(k)),   # float -> int
        n=int(n),
        p=p,
        alternative=alternative
    ).pvalue

# 핵심: BorutaShap 모듈 내부 전역 함수명을 패치
borutashap_module.binom_test = fixed_binom_test

# 이제 클래스 사용
from BorutaShap import BorutaShap
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

model = XGBRegressor(
    n_estimators=100,
    max_depth=7,
    random_state=42,
    tree_method='hist',
    device='cuda',
    objective='reg:squarederror'
)

Feature_Selector = BorutaShap(
    model=model,
    importance_measure='shap',
    classification=False
)

train_df = full_df[full_df['battery_id'].isin(train_ids)]
test_df = full_df[full_df['battery_id'].isin(test_ids)]

X_train, y_train = train_df[feature_list], train_df['target_RUL']
X_test, y_test = test_df[feature_list], test_df['target_RUL']

Feature_Selector.fit(X=X_train, y=y_train, n_trials=50, sample=False, verbose=True)

plt.figure(figsize=(12, 8))
Feature_Selector.plot(which_features='all')


confirmed_features = Feature_Selector.accepted
tentative_features = Feature_Selector.tentative
rejected_features = Feature_Selector.rejected

print("\n" + "="*30)
print(f"✅ Confirmed Features ({len(confirmed_features)}): {confirmed_features}")
print(f"⚠️ Tentative Features ({len(tentative_features)}): {tentative_features}")
print(f"❌ Rejected Features ({len(rejected_features)}): {rejected_features}")
print("="*30)


