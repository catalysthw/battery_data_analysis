import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt


from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


import random






data_folder = "/kaggle/input/nasa-battery-dataset/cleaned_dataset"
meta_data = pd.read_csv(os.path.join(data_folder, "metadata.csv"))
print(meta_data)



def extract_features_from_file(file_path, battery_id, cycle_idx, total_cycles, meta_capacity, num_samples=50):
    if not os.path.exists(file_path): return []
    df = pd.read_csv(file_path)
    if len(df) < 110: return []

    # 1. 물리 변수 및 방전량 계산
    v_vals = df['Voltage_measured'].values
    t_vals = df['Time'].values
    dt = df['Time'].diff().fillna(0)
    
    # 누적 방전량 계산 (Ah)
    cum_discharged_ah = (df['Current_measured'].abs() * dt).cumsum() / 3600
    calc_total_cap = cum_discharged_ah.iloc[-1] 
    
    if calc_total_cap == 0: return []

    # V-jump 계산
    v_init = v_vals[0]
    try:
        v_drop_idx = df[df['Current_measured'].abs() > 0.1].index[0]
        v_jump = v_init - v_vals[v_drop_idx]
    except: 
        v_jump = 0

    # 2. 메타데이터 기반 지표 계산 (iloc 사용으로 인덱스 안전 확보)
    b_cap_max = meta_capacity.max()
    meta_cap_idx = meta_capacity.iloc[cycle_idx-1] 

    cap_error_abs = abs(meta_cap_idx - calc_total_cap)
    cap_error_rel = cap_error_abs / (meta_cap_idx + 1e-9)
    target_soh = meta_cap_idx / b_cap_max if b_cap_max > 0 else 0

    # 3. 샘플링 루프
    file_features = []
    indices = np.linspace(100, len(df)-11, num_samples, dtype=int)
    
    for idx in indices:
        # 통계량 계산
        past_df = df.iloc[:idx]
        aggs = past_df[['Voltage_measured', 'Current_measured', 'Temperature_measured']].agg(['mean', 'std'])
        
        current_discharged = cum_discharged_ah.iloc[idx]
        
        # 잔량(Remaining Capacity) 및 비율 계산
        rem_cap_calc = calc_total_cap - current_discharged
        rem_cap_ratio = rem_cap_calc / (calc_total_cap + 1e-9)
        
        rem_cap_meta = meta_cap_idx - current_discharged
        rem_cap_ratio_meta = max(0, rem_cap_meta) / (meta_cap_idx + 1e-9)

        def get_slope(win_size):
            return (v_vals[idx] - v_vals[idx-win_size]) / (t_vals[idx] - t_vals[idx-win_size] + 1e-9)

        # [들여쓰기 주의] row 딕셔너리 시작
        row = {
            'battery_id': battery_id,
            'cycle_count': cycle_idx,
            'voltage': v_vals[idx], 
            'current': df['Current_measured'].iloc[idx], 
            'temp': df['Temperature_measured'].iloc[idx],
            'v_jump': v_jump,
            'slope_20': get_slope(20), 
            'slope_100': get_slope(100),
            'rel_discharged_ratio': current_discharged / calc_total_cap,
            'cap_error_abs': cap_error_abs,
            'cap_error_rel': cap_error_rel,
                        
            # Target values
            'target_SOH': target_soh,            
            'target_RUL': t_vals[-1] - t_vals[idx],
            'target_remaining_cycles': total_cycles - cycle_idx,
            'target_life_ratio': (total_cycles - cycle_idx) / total_cycles,
            'target_CRA': rem_cap_calc,      
            'target_RAC': rem_cap_ratio,     
            'target_MCRA': rem_cap_meta,     
            'target_MRAC': rem_cap_ratio_meta,            

            # 통계 피처
            'voltage_mean': aggs.loc['mean', 'Voltage_measured'], 
            'voltage_std': aggs.loc['std', 'Voltage_measured'],
            'current_mean': aggs.loc['mean', 'Current_measured'], 
            'current_std': aggs.loc['std', 'Current_measured'],
            'temp_mean': aggs.loc['mean', 'Temperature_measured'], 
            'temp_std': aggs.loc['std', 'Temperature_measured']
        }
        file_features.append(row)
        
    return file_features



def prepare_battery_data(metadata, data_folder, num_samples=50, test_size=0.3, random_state=42):

    # Capacity 컬럼을 숫자로 변환
    metadata['Capacity'] = pd.to_numeric(metadata['Capacity'], errors='coerce')
    
    all_data = []
    all_ids = metadata['battery_id'].unique()

    
    # [수정 포인트] test_size가 0이면 분할하지 않고 전체를 train_ids로 설정
    if test_size == 0 or test_size is None:
        train_ids = all_ids
        test_ids = np.array([]) # 빈 배열
    else:
        train_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=random_state)
    
    # # ID 분리는 메인 로직 시작 전 수행
    # train_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=random_state)

    for b_id in all_ids:
        # 특정 배터리의 방전 파일만 필터링
        b_df = metadata[(metadata['battery_id'] == b_id) & (metadata['type'] == 'discharge')].sort_values('filename')

        # 2. b_df에서 바로 필요한 기준값들 추출
        b_total_cycles = len(b_df)              # 이 배터리가 수행한 총 방전 횟수
        b_meta_capacity = b_df['Capacity']      # 이 배터리의 방전 capacity 데이터
        
        for i, row_meta in enumerate(b_df.itertuples(), 1):
            file_path = os.path.join(data_folder, "data", row_meta.filename)
            
            # 특징 추출 함수 호출
            file_feats = extract_features_from_file(
                file_path=file_path,
                battery_id=b_id,
                cycle_idx=i,
                total_cycles=b_total_cycles,
                meta_capacity=b_meta_capacity,  # Series 데이터 전달
                num_samples=num_samples         # 기본값 50
            )
            all_data.extend(file_feats)

    final_df = pd.DataFrame(all_data)
    
    # 피처 리스트 자동 생성 (ID와 Target 제외)
    # target_으로 시작하는 모든 컬럼을 feature에서 제외
    target_cols = [c for c in final_df.columns if c.startswith('target')] + ['battery_id', 'cap_error_abs', 'cap_error_rel']
    feature_list = [c for c in final_df.columns if c not in target_cols]
    
    # 최종 분할
    train_data = final_df[final_df['battery_id'].isin(train_ids)]
    test_data = final_df[final_df['battery_id'].isin(test_ids)]
    
    return train_data, test_data, feature_list, train_ids

# 실행 예시

# 1. 전체 Battery ID 추출 및 분할
all_battery_ids = meta_data['battery_id'].unique()
# 고정된 결과를 원하시면 random_state를 설정하세요.
train_ids, test_ids = train_test_split(all_battery_ids, test_size=0.3, random_state=42)

print(f"Total Batteries: {len(all_battery_ids)}")
print(f"Train IDs ({len(train_ids)}): {train_ids}")
print(f"Test IDs  ({len(test_ids)}): {test_ids}")



# 1. 데이터 폴더 및 설정값 지정
DATA_PATH = data_folder            # NASA 데이터셋이 있는 폴더 경로로 수정하세요
NUM_SAMPLES_PER_FILE = 50          # 파일당 추출할 포인트 수 (질문자님 의견 반영)


# Full feature generation
print(">>> Generating features for ALL batteries (This may take a while)...")
# 기존에 정의한 prepare_battery_data에서 test_size=0으로 설정하여 모든 ID 포함
full_df, _, feature_list, _ = prepare_battery_data(
    metadata=meta_data, 
    data_folder=DATA_PATH,
    num_samples=NUM_SAMPLES_PER_FILE,
    test_size=0.0, # 전체 배터리를 하나의 DataFrame으로 수집
    random_state=42
)




def run_experiment(train_df, test_df, feature_list, iteration_idx, model_type='xgb'):
    """
    model_type: 'xgb', 'lgbm', 'rf' 중 선택 가능
    """
    # [Step 1] 데이터 준비 (현재 SOC 예측 기준)
    # X_train, y_train = train_df[feature_list], train_df['target_RUL']
    # X_test, y_test = test_df[feature_list], test_df['target_RUL']

    X_train, y_train = train_df[feature_list], train_df['target_life_ratio']
    X_test, y_test = test_df[feature_list], test_df['target_life_ratio']

    

    # [Step 2] 모델 설정
    if model_type == 'lgbm':
        model = LGBMRegressor(
            n_estimators=1000,          # 충분히 학습하도록 늘림
            learning_rate=0.03,         # 더 세밀하게 학습
            num_leaves=127,             # 복잡도를 XGB(depth 7) 수준으로 상향
            max_depth=-1,               # num_leaves에 의해 조절되도록 설정
            min_child_samples=20,       # 과적합 방지를 위해 조정  
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            importance_type='gain',
            # force_col_wise=True
        )
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'xgb':
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective='reg:squarederror'
        )
    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")

    # [Step 3] 학습 및 예측
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # [Step 4] 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {"Model": model_type, "Iteration": iteration_idx, "MAE": mae, "RMSE": rmse, "R2": r2}
    




all_results = []
num_iterations = 10

print(f">>> Starting {num_iterations} iterations of experiments...")
print("-" * 60)

# for i in range(1, num_iterations + 1):
for iteration_idx in range(1, num_iterations + 1):
    # 매번 새로운 random_state(또는 None)로 데이터 분할
    # (주의: 여기서는 매번 파일을 새로 읽지 않도록 이전에 만든 full_df를 재활용하는 것이 빠릅니다)

    # 0에서 9999 사이의 랜덤한 숫자 하나를 뽑습니다.
    current_seed = random.randint(0, 9999)
    
    # 1. ID 전체 목록에서 무작위 분할 (seed=None)
    all_ids = meta_data['battery_id'].unique()
    train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=current_seed)
    
    # 2. 미리 생성된 full_df에서 필터링 (가장 효율적인 방식)
    train_df_iter = full_df[full_df['battery_id'].isin(train_ids)]
    test_df_iter = full_df[full_df['battery_id'].isin(test_ids)]

    
    # 3. 실험 수행 및 결과 저장
    # result = run_experiment(train_df_iter, test_df_iter, feature_list, i)
    result = run_experiment(train_df_iter, test_df_iter, feature_list, iteration_idx, model_type='xgb')
    # result = run_experiment(train_df_iter, test_df_iter, feature_list, iteration_idx, model_type='rf')
    # result = run_experiment(train_df_iter, test_df_iter, feature_list, iteration_idx, model_type='lgbm')
    
    # 5. [중요] 결과 딕셔너리에 시드 정보 추가
    result['Seed'] = current_seed
    
    all_results.append(result)

    # 진행 상황 출력 (시드 포함)
    print(f"Iteration {iteration_idx:02d}/{num_iterations} | Seed: {current_seed}  \
    | MAE: {result['MAE']:.2f}s | RMSE: {result['RMSE']:.4f}s | R2: {result['R2']:.4f}")


# --- 결과 리포트 생성 ---
results_table = pd.DataFrame(all_results)

print("\n" + "="*60)
print(f"FINAL PERFORMANCE SUMMARY ({num_iterations} Iterations)")
print("="*60)
print(results_table.to_string(index=False))
print("-" * 60)

# 평균 및 표준편차 출력
summary = results_table.drop(['Model','Iteration'], axis=1, errors='ignore').agg(['mean', 'std'])
print(summary)
print("="*60)












