import pandas as pd
import numpy as np
from datetime import datetime, timedelta



def equip_light_demand(file_path, num_steps, time_now, step_duration):
    """
    读取POWER.csv文件，处理功率数据，并根据输入的时间参数截取和选择数据。

    参数:
    file_path (str): POWER.csv文件的路径。
    num_steps (int): 需要截取的步数。
    time_now (float): 当前时间相对于年初的小时数。
    step_duration (float): 每个步长的时长，单位为小时。

    返回:
    tuple: 包含截取后的 'P_EQU_kW' 和 'P_LIG_kW' NumPy 数组。
           如果文件或列不存在，则返回 None。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None, None

    # 清除列名前后空格
    df.columns = df.columns.str.strip()

    # 检查所需的列是否存在
    required_cols = ['P_EQUkJph', 'P_LIGkJph']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误: CSV文件中缺少列 '{col}'。")
            return None, None

    # 将kJ/ph (千焦/每小时) 转换为 kW (千瓦)
    df['P_EQU_kW'] = df['P_EQUkJph'] * 0.2778
    df['P_LIG_kW'] = df['P_LIGkJph'] * 0.2778

    # 假设数据是每年8760小时，步长为0.5小时
    total_hours_in_data = 8760
    original_data_step_hours = 0.5  # 原始数据的步长
    total_data_points = int(total_hours_in_data / original_data_step_hours)

    if len(df) != total_data_points:
        print(f"警告: 数据点数量 ({len(df)}) 与预期 ({total_data_points}) 不符。")
        print("请确保 'POWER.csv' 包含8760小时的数据，步长为0.5小时。")

    # 根据时间截取数据
    # 计算 'time_now' 对应的原始数据中的起始索引
    start_index_original = int(time_now / original_data_step_hours)

    # 计算总的时间范围
    total_time_horizon = num_steps * step_duration

    # 计算需要截取多少个原始数据点才能覆盖总时间范围
    num_original_points_for_horizon = int(total_time_horizon / original_data_step_hours)
    end_index_original = start_index_original + num_original_points_for_horizon

    # 确保索引在有效范围内
    start_index_original = max(0, start_index_original)
    end_index_original = min(len(df), end_index_original)

    if start_index_original >= len(df) or start_index_original == end_index_original:
        print(f"警告: 'time_now' ({time_now}) 超出数据范围或没有足够的未来数据点。")
        print("请检查输入时间是否在8760小时数据的有效时间范围内。")
        return np.array([]), np.array([])  # 返回空 NumPy 数组

    # 截取原始数据并转换为 NumPy 数组
    P_EQU_final = df['P_EQU_kW'].iloc[start_index_original:end_index_original].to_numpy()
    P_LIG_final = df['P_LIG_kW'].iloc[start_index_original:end_index_original].to_numpy()

    return P_EQU_final, P_LIG_final


if __name__ == "__main__":
    # 示例用法
    file_path = 'POWER.csv' # 请确保您的项目中有此文件

    # 请根据您的需求修改以下参数
    num_steps_input = 10  # 步长数，例如10个步长
    step_duration_input = 0.5 # 每个步长的时长，例如0.5小时

    # 当前时间相对于年初的小时数，例如 0.5 表示年初的0.5小时
    # 假设一年有 8760 小时
    time_now_input_hours = 2000

    P_EQU_data, P_LIG_data = equip_light_demand(file_path, num_steps_input, time_now_input_hours, step_duration_input)

    if P_EQU_data is not None and P_LIG_data is not None:
        print("\n截取后的 P_EQU (kW) 数据 (NumPy 数组):")
        print(P_EQU_data)
        print("\n截取后的 P_LIG (kW) 数据 (NumPy 数组):")
        print(P_LIG_data)

        print(f"\n截取的数据点数量 (P_EQU): {len(P_EQU_data)}")
        print(f"截取的数据点数量 (P_LIG): {len(P_LIG_data)}")
        print(f"P_EQU_data 的类型: {type(P_EQU_data)}")
        print(f"P_LIG_data 的类型: {type(P_LIG_data)}")
    else:
        print("\n数据处理失败。")
