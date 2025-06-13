from initialization.BiLSTM.datamaker import make_dataset
from initialization.BiLSTM.transform_LSTM import train_main
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def RTPV_TRAIN(files_dir):
    tag = 'RTPV'

    pv_csv_path = files_dir / 'PV.csv'

    original_data = pd.read_csv(pv_csv_path)  # 400M2
    original_data.columns = original_data.columns.str.strip()

    # 选取 数值 型 变量
    input_features_list = ['IR_hori_kjphpm2', 'Incidence_Angle_Horizontal',
                           'Tout']  # Renamed 'input' to 'input_features_list' to avoid conflict
    original_data = original_data[
        ['IR_hori_kjphpm2', 'Incidence_Angle_Horizontal', 'Tout', 'RTPV_KJPH']]

    var_data = original_data[input_features_list]  # Features
    ylable_data = original_data[['RTPV_KJPH']]  # Labels

    # --- START OF MODIFICATION ---
    # 归一化处理
    # 使用标准化（z-score标准化）

    # 创建独立的特征归一化器
    feature_scaler = StandardScaler()
    var_data = feature_scaler.fit_transform(var_data)

    # 创建独立的标签归一化器
    label_scaler = StandardScaler()
    ylable_data = label_scaler.fit_transform(ylable_data)

    # 保存两个归一化器
    dump(feature_scaler, files_dir /f'feature_scaler_{tag}')  # Save feature scaler separately
    dump(label_scaler, files_dir /f'scaler_{tag}')  # Save label scaler (your original 'scaler')
    # --- END OF MODIFICATION ---

    # 定义滑动窗口大小
    window_size = 4
    # 制作数据集
    train_set, train_label, test_set, test_label = make_dataset(var_data, ylable_data, window_size)
    # 保存数据
    dump(train_set, files_dir /f'train_set_{tag}')
    dump(train_label, files_dir /f'train_label_{tag}')
    dump(test_set, files_dir /f'test_set_{tag}')
    dump(test_label, files_dir /f'test_label_{tag}')
    print('数据 形状：')
    print(train_set.size(), train_label.size())
    print(test_set.size(), test_label.size())

    input_dim = len(input_features_list)  # Use the correct list
    if input_dim < 2:
        num_heads = 1
        print(f"警告：input_dim ({input_dim}) 太小。多头注意力将退化为单头注意力。")
    elif input_dim % 2 == 0:
        num_heads = 2
    elif input_dim % 3 == 0:
        num_heads = 3
    else:
        num_heads_candidate = input_dim // 2
        while input_dim % num_heads_candidate != 0 and num_heads_candidate > 1:
            num_heads_candidate -= 1
        num_heads = num_heads_candidate if num_heads_candidate >= 1 else 1
        if num_heads == 1:
            print(f"警告：input_dim ({input_dim}) 无法被大于 1 的整数整除。num_heads 设置为 1。")
    output_dim = 1
    train_main(input_dim, num_heads, output_dim, tag,files_dir)



