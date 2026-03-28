# !pip install dowhy shap -q

from sklearn.preprocessing import StandardScaler
import pandas as pd
from dowhy import CausalModel

candidate_treatments = ['v_jump', 'slope_20', 'slope_100', 'voltage_mean', 'voltage_std']
confounders = ['cycle_count', 'temp_mean', 'temp_std', 'current_mean', 'current_std']

results = []

for tr in candidate_treatments:
    cols = [tr] + confounders + ['target_life_ratio']
    df_tmp = train_df[cols].dropna().copy()

    # 표준화
    scaler = StandardScaler()
    df_tmp[[tr] + confounders] = scaler.fit_transform(df_tmp[[tr] + confounders])

    graph = f"""
    digraph {{
        cycle_count -> {tr};
        temp_mean -> {tr};
        temp_std -> {tr};
        current_mean -> {tr};
        current_std -> {tr};

        cycle_count -> target_life_ratio;
        temp_mean -> target_life_ratio;
        temp_std -> target_life_ratio;
        current_mean -> target_life_ratio;
        current_std -> target_life_ratio;

        {tr} -> target_life_ratio;
    }}
    """

    cm = CausalModel(
        data=df_tmp,
        treatment=tr,
        outcome='target_life_ratio',
        graph=graph
    )

    estimand = cm.identify_effect(proceed_when_unidentifiable=True)
    est = cm.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression"
    )

    results.append({
        'treatment': tr,
        'causal_effect_std': est.value
    })

causal_effect_std_df = pd.DataFrame(results)
causal_effect_std_df['abs_effect'] = causal_effect_std_df['causal_effect_std'].abs()
causal_effect_std_df = causal_effect_std_df.sort_values('abs_effect', ascending=False)

print(causal_effect_std_df)

