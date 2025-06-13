import os

import param_manage as pm
import pandas as pd
import numpy as np # 确保 numpy 被导入，因为 pandas 内部会用到它

file_path = 'MPC_INPUT.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, 'MPC_INPUT.csv')
try:
    input_df = pd.read_csv(file_path)
    input_df.columns = input_df.columns.str.strip() # 清理列名中的空白字符
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保文件存在并与脚本在同一目录下。")
    exit() # 如果文件不存在，直接退出程序

def data_clip(time_now: float, horizon: int):
    """
    根据当前时间 (time_now) 和预测步长 (horizon) 裁剪输入数据，
    并提取仿真所需的各项参数。

    参数:
        time_now (float): 当前的仿真时间点。
        horizon (int): 需要裁剪的未来步长数量（行数）。

    返回:
        tuple: 一个包含以下元素的元组：
            - T_amb_list (np.array): 未来环境温度列表。
            - Q_in (np.array): 未来内部热增益列表（已转换为 kW）。
            - vent_flow (np.array): 未来通风流量列表（已转换为 kg/s）。
            - measured_temp (np.array): 测量到的区域空气温度列表。
            - Tin_t (float): 初始区域空气温度。
            - measure_total_power (np.array): 测量到的总功率列表（已转换为 kW）。
            - Twall_t_dict (dict): 初始墙体温度字典。
            - Tsoil_t_list (np.array): 未来土壤温度列表。
            - tank_df (pd.DataFrame): 包含初始时刻水箱温度的 DataFrame。
    """
    # 查找与 time_now 最接近的 TIME 值对应的行索引
    # .abs().idxmin() 找到绝对差值最小的那个索引
    start_index = (input_df['TIME'] - time_now).abs().idxmin()
    print(f"\nTime_now {time_now} 对应的实际开始时间为: {input_df.loc[start_index, 'TIME']}")

    # 计算裁剪的结束索引 (不包含此索引的行)
    # 使用 min() 确保裁剪不会超出 DataFrame 的最大范围
    end_index_for_slice = start_index + horizon
    df_clip = input_df.iloc[start_index: min(end_index_for_slice, len(input_df))].copy()

    # 提取和转换数据
    # 请根据您的 CSV 文件中的实际列名进行调整
    T_amb_list = df_clip['Tout'].values
    Q_in_list = df_clip['Qin_kJph'].values * 0.2778  # 假设原始单位是 kJ/h，转换为 W (1 kJ/h = 0.2778 W = 0.0002778 kW)
    vent_flow = df_clip['Mrate_kgph'].values / 3600  # 假设原始单位是 kg/h，转换为 kg/s
    measured_temp = df_clip['TAIR_Zone1'].values # 假设这是您的区域空气温度列

    # 获取初始区域空气温度 (裁剪数据的第一行)
    Tin_t = measured_temp[0]

    measure_total_power = df_clip['measure_total_power'].values * 0.2778 # 假设原始单位是 J/h，转换为 W

    # 初始化墙体温度字典 (假设初始墙体温度与区域空气温度相同)
    Twall_t_dict = {}
    # 确保 pm.wall_temp_columns 在 param_manage.py 中有定义
    if hasattr(pm, 'wall_temp_columns'):
        for wall in pm.wall_temp_columns:
            Twall_t_dict[wall] = measured_temp[0]
    else:
        print("警告: param_manage.py 中未找到 'wall_temp_columns'。Twall_t_dict 将为空。")


    Tsoil_t_list = df_clip['T_Soil'].values # 假设这是您的土壤温度列

    # 定义需要提取的水箱列名列表
    columns_to_extract_tank = ['TIME'] + [f'TankAC{i}' for i in range(1, 6)] + [f'TankSC{i}' for i in range(1, 6)]

    # 提取初始时刻 (start_index 对应行) 的水箱温度数据
    # 使用 .loc[idx:idx] 确保结果是一个单行的 DataFrame
    tank_df = input_df.loc[start_index:start_index, columns_to_extract_tank].copy()


    P_pan_kjh = input_df.loc[start_index, 'P_Fan_kjph']

    return (T_amb_list, Q_in_list, vent_flow, measured_temp, Tin_t,
            measure_total_power, Twall_t_dict, Tsoil_t_list, tank_df, P_pan_kjh)


import pandas as pd
import numpy as np
import torch
from joblib import load
from pathlib import Path


# --- 预测器类定义（保持不变，或根据实际情况从其他文件导入） ---
# --- 全局模型和归一化器加载（只执行一次） ---
# 配置参数
BASE_PATH = pm.model_dir
MODEL_PATH = BASE_PATH / 'transformer_bilstm_RTPV.pt'
FEATURE_SCALER_PATH = BASE_PATH / 'feature_scaler_RTPV'
LABEL_SCALER_PATH = BASE_PATH / 'scaler_RTPV'
WEATHER_CSV_PATH_GLOBAL = 'weather.csv'  # 初始加载天气数据时使用
class Predictor:
    def __init__(self, model_path: str, feature_scaler_path: str, label_scaler_path: str, window_size: int = 4):
        self.window_size = window_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.feature_scaler = self._load_scaler(feature_scaler_path, "特征归一化器")
        self.label_scaler = self._load_scaler(label_scaler_path, "标签归一化器")
        self.feature_columns = ['IR_hori_kjphpm2', 'Incidence_Angle_Horizontal', 'Tout']

    def _load_model(self, path: str):
        try:
            model = torch.load(path, map_location=self.device, weights_only=False)
            model.eval()
            print(f"模型 '{Path(path).name}' 成功加载。")
            return model
        except Exception as e:
            print(f"错误: 加载模型 '{path}' 失败。")
            raise e

    def _load_scaler(self, path: str, name: str):
        try:
            scaler = load(path)
            print(f"{name} '{Path(path).name}' 成功加载。")
            return scaler
        except Exception as e:
            print(f"错误: 加载归一化器 '{path}' 失败。")
            raise e

    def _prepare_input_data(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("输入数据 DataFrame 为空，无法进行归一化和窗口处理。")
        scaled_features = self.feature_scaler.transform(data[self.feature_columns])
        shape = (scaled_features.shape[0] - self.window_size + 1, self.window_size, scaled_features.shape[1])
        strides = (scaled_features.strides[0], scaled_features.strides[0], scaled_features.strides[1])
        windowed_data = np.lib.stride_tricks.as_strided(scaled_features, shape=shape, strides=strides)
        return torch.from_numpy(windowed_data).float().to(self.device)

    # Predict 方法：恢复为宽松过滤，确保不提前起始点，只求多不求少
    def predict(self, start_time: float, duration_hours: float, weather_data_slice: pd.DataFrame):
        if weather_data_slice is None or weather_data_slice.empty:
            return None
        missing_cols = [col for col in self.feature_columns if col not in weather_data_slice.columns]
        if missing_cols:
            print(f"错误: 传入的 weather_data_slice 缺少必要的特征列: {missing_cols}")
            return None
        if 'TIME' not in weather_data_slice.columns:
            print("错误: 传入的 weather_data_slice 缺少 'TIME' 列。")
            return None
        if len(weather_data_slice) < self.window_size:
            # 数据不足以形成任何窗口
            return None

        # original_step_hours = 0.5 # 这个变量在此方法中不再直接用于计算长度，但其值是模型固有的

        try:
            input_tensor = self._prepare_input_data(weather_data_slice)
        except ValueError as e:
            print(f"错误: 准备输入数据时发生错误: {e}")
            return None

        # 检查是否能够生成任何滑动窗口
        if input_tensor.shape[0] == 0:
            return None

        with torch.no_grad():
            scaled_predictions = self.model(input_tensor).cpu().numpy()

        unscaled_predictions = self.label_scaler.inverse_transform(scaled_predictions)

        # 生成 weather_data_slice 能支持的所有预测点的时间序列
        all_possible_prediction_times = weather_data_slice['TIME'].iloc[self.window_size - 1:].values

        if len(unscaled_predictions) != len(all_possible_prediction_times):
            print(f"错误: 模型预测结果数量 ({len(unscaled_predictions)}) 与预期时间点数量 ({len(all_possible_prediction_times)}) 不匹配。")
            return None

        # --- 核心修改：恢复为宽松过滤，确保不提前起始点，只求多不求少 ---

        end_time_requested = start_time + duration_hours

        # 筛选条件：
        # 1. 预测点必须 >= start_time (不允许起始点变小，使用 np.isclose 确保对齐)
        # 2. 预测点必须 <= end_time_requested (允许包含结束点，即使多一个)
        # 使用 np.isclose 避免浮点数比较误差
        valid_indices = np.where(
            (all_possible_prediction_times >= start_time - 1e-9) & # 保证不让起始点变小
            (all_possible_prediction_times <= end_time_requested + 1e-9) # 尽量多给，包含所有在范围内的点
        )[0]

        if len(valid_indices) == 0:
            # 如果在这个宽松过滤下都没有点，那就是真的没数据
            return None

        time_points_for_output = all_possible_prediction_times[valid_indices]
        predictions_for_output = unscaled_predictions[valid_indices].flatten()

        # --- 修改到此结束 ---

        prediction_df = pd.DataFrame({
            'TIME': time_points_for_output,
            'Predicted_Value': predictions_for_output
        })

        # 返回的 DataFrame 可能会比期望多一个点，或者在极端情况下可能少了点（如果数据源不足）
        # 精确的长度控制将完全由上层 RTPV_Predict 负责。
        return prediction_df




# 全局 Predictor 实例，只加载一次模型和归一化器
GLOBAL_PREDICTOR = None
try:
    GLOBAL_PREDICTOR = Predictor(
        model_path=str(MODEL_PATH),
        feature_scaler_path=str(FEATURE_SCALER_PATH),
        label_scaler_path=str(LABEL_SCALER_PATH)
    )
except Exception as e:
    print(f"严重错误: 全局预测器初始化失败。请检查模型和归一化器路径。错误: {e}")
    GLOBAL_PREDICTOR = None

# --- 全局天气数据加载和切片函数 ---
FULL_WEATHER_DF = pd.DataFrame()  # 初始化为空，以防加载失败
try:
    FULL_WEATHER_DF = pd.read_csv(WEATHER_CSV_PATH_GLOBAL)
    FULL_WEATHER_DF.columns = FULL_WEATHER_DF.columns.str.strip()
    FULL_WEATHER_DF['TIME'] = pd.to_numeric(FULL_WEATHER_DF['TIME'])
    # print(f"全局天气数据 '{WEATHER_CSV_PATH_GLOBAL}' 加载成功。") # 减少模块加载时的打印
except Exception as e:
    print(f"严重错误: 全局天气数据加载失败。请检查 '{WEATHER_CSV_PATH_GLOBAL}' 文件。错误: {e}")
    FULL_WEATHER_DF = pd.DataFrame()

# 定义原始模型步长
ORIGINAL_MODEL_STEP = 0.5


def get_weather_slice_for_prediction(start_time: float, num_prediction_steps: int, window_size: int) -> pd.DataFrame:
    """
    从全局天气数据中切片出预测所需的特定时间范围数据。
    这里提供的数据范围会**远超理论所需**，以确保 Predictor.predict 总是能获得足够的数据。

    Args:
        start_time (float): 预测的开始时间。
        num_prediction_steps (int): 期望最终返回的原始步长下的预测点数。
        window_size (int): 模型所需的滑动窗口大小。

    Returns:
        pd.DataFrame: 包含所需天气数据的 DataFrame 切片。如果全局数据未加载或切片为空则返回空的DataFrame。
    """
    if FULL_WEATHER_DF.empty:
        return pd.DataFrame()

    # 计算切片起始时间：从 start_time 往前多回溯一些，确保 Predictor 有足够的历史数据来形成第一个窗口
    # 额外多回溯两个窗口的长度，增加鲁棒性
    slice_start_time_val = start_time - window_size * ORIGINAL_MODEL_STEP * 2
    slice_start_time_val = max(0.0, slice_start_time_val)

    # 计算切片结束时间：覆盖到期望的最后一个预测点，再往后多几个窗口长度
    # 期望的最后一个原始预测点的时间
    expected_last_prediction_time = start_time + (num_prediction_steps - 1) * ORIGINAL_MODEL_STEP

    # 切片结束时间：从 expected_last_prediction_time 往后多延长几个窗口长度
    # 确保 Predictor 有足够的未来数据来生成所有期望的预测点
    slice_end_time_val = expected_last_prediction_time + window_size * ORIGINAL_MODEL_STEP * 2  # 往后多延长 2 个窗口长度

    # 确保时间范围合理
    slice_end_time_val = min(FULL_WEATHER_DF['TIME'].max(), slice_end_time_val)

    # 使用 np.searchsorted 进行整数索引查找，这是最可靠的方式
    start_idx_in_full_df = np.searchsorted(FULL_WEATHER_DF['TIME'].values, slice_start_time_val - 1e-9)
    end_idx_in_full_df = np.searchsorted(FULL_WEATHER_DF['TIME'].values, slice_end_time_val + 1e-9)

    start_idx_in_full_df = max(0, start_idx_in_full_df)
    end_idx_in_full_df = min(len(FULL_WEATHER_DF), end_idx_in_full_df)

    # 通过整数索引进行切片
    time_range_data = FULL_WEATHER_DF.iloc[start_idx_in_full_df:end_idx_in_full_df].reset_index(drop=True)

    return time_range_data


def main():
    """
    主函数，用于测试 data_clip 功能。
    您可以修改 time_now 和 horizon 的值进行不同的测试。
    """
    print("--- 启动 main() 测试 ---")

    # 示例用法：设置当前仿真时间和预测步长
    current_sim_time = 50.0  # 例如，从时间 50.0 开始裁剪
    prediction_horizon = 5  # 例如，裁剪未来 48 个步长 (如果步长是0.5，则为 24 小时的数据)

    print(f"\n正在为 time_now = {current_sim_time} 和 horizon = {prediction_horizon} 个步长裁剪数据。")

    # 调用 data_clip 函数获取所有参数
    (T_amb_list, Q_in, vent_flow, measured_temp, Tin_t,
     measure_total_power, Twall_t_dict, Tsoil_t_list, tank_df) = data_clip(current_sim_time, prediction_horizon)

    # 打印一些结果进行验证
    print("\n--- 裁剪数据概览 ---")
    print(f"环境温度 (T_amb_list) 长度: {len(T_amb_list)} (前5个值): {T_amb_list[:5]}")
    print(f"内部热增益 (Q_in) 长度: {len(Q_in)} (前5个值): {Q_in[:5]} kW")
    print(f"通风流量 (vent_flow) 长度: {len(vent_flow)} (前5个值): {vent_flow[:5]} kg/s")
    print(f"测量温度 (measured_temp) 长度: {len(measured_temp)} (前5个值): {measured_temp[:5]}")
    print(f"初始区域温度 (Tin_t): {Tin_t}")
    print(f"总测量功率 (measure_total_power) 长度: {len(measure_total_power)} (前5个值): {measure_total_power[:5]} kW")
    print(f"初始墙体温度 (Twall_t_dict): {Twall_t_dict}")
    print(f"土壤温度 (Tsoil_t_list) 长度: {len(Tsoil_t_list)} (前5个值): {Tsoil_t_list[:5]}")

    print("\n--- 提取的水箱数据 (tank_df) ---")
    print(tank_df)
    if not tank_df.empty:
        print(f"水箱数据中 TIME 的值: {tank_df['TIME'].iloc[0]}")

    print("\n--- 测试完成 ---")

# 这确保只有当脚本作为主程序运行时才执行 main() 函数
if __name__ == "__main__":
    main()