
import numpy as np
import pandas as pd  # 导入 pandas 以便在必要时进行类型提示

# 从 input_manage 模块导入所有必要的全局资源和辅助函数
# 导入这些会触发 input_manage.py 中的全局初始化代码（模型加载、天气数据加载）
from input_manage import GLOBAL_PREDICTOR, FULL_WEATHER_DF, ORIGINAL_MODEL_STEP, get_weather_slice_for_prediction



def RTPV_Predict(start_hour: float, num_steps: int, output_step_hours: float = 0.5) -> np.ndarray:
    """
    光伏发电量预测的核心接口函数。

    Args:
        start_hour (float): 预测的开始时间，以年度小时数表示 (e.g., 4000.0)。
        num_steps (int): 从开始时间算起，需要预测的步数 (e.g., 5 步)。
                         注意：这是期望在 0.5h 步长下得到的最终预测点数。
        output_step_hours (float): 期望的输出预测步长（小时），默认为 0.5 小时。
                                   保证是 0.5 的整数倍。

    Returns:
        np.ndarray: 包含重采样后的预测值的 NumPy 一维数组。
                    如果无法进行预测或结果为空，则返回一个空的 NumPy 数组 `np.array([])`。
    """
    final_predictions_array = np.array([])

    if GLOBAL_PREDICTOR is None or FULL_WEATHER_DF.empty:
        print("错误: 全局预测器或天气数据未成功初始化，无法进行预测。请检查 input_manage.py 的初始化日志。")
        return final_predictions_array

    if num_steps <= 0:
        print("警告: 预测步数 (num_steps) 必须是正整数。")
        return final_predictions_array

    # 计算传递给 Predictor.predict 的总时长
    # 这个总时长应能覆盖从 start_hour 开始的 num_steps 个原始点
    # (num_steps - 1) * ORIGINAL_MODEL_STEP 是从第一个点到最后一个点的时长
    # 加上一个 ORIGINAL_MODEL_STEP 是为了确保 Predictor.predict 至少能返回 num_steps 个点
    # (即覆盖到 start_hour + num_steps * ORIGINAL_MODEL_STEP 这个时间点)
    duration_for_predictor = num_steps * ORIGINAL_MODEL_STEP * 2  # 两倍缓冲

    try:
        # 1. 从全局天气数据中切片出预测所需的原始天气数据
        # 这里传递的是一个稍微宽松的时间范围，以确保 raw_predictions_df 足够长
        weather_data_slice = get_weather_slice_for_prediction(
            start_time=start_hour,  # 参数名 'start_time' 匹配
            num_prediction_steps=num_steps,  # 参数名 'num_prediction_steps' 匹配
            window_size=GLOBAL_PREDICTOR.window_size  # 参数名 'window_size' 匹配
        )

        if weather_data_slice.empty:
            print(f"⚠️ 警告: 无法为时间范围 [{start_hour}, 期望 {num_steps} 步] 获取足够的天气数据切片。")
            return final_predictions_array

        # 2. 获取原始步长（0.5小时）的预测结果 DataFrame
        # 这里的 raw_predictions_df 可能包含比 num_steps 更多的点
        raw_predictions_df = GLOBAL_PREDICTOR.predict(
            start_time=start_hour,
            duration_hours=duration_for_predictor,  # 将计算后的时长传递给 Predictor
            weather_data_slice=weather_data_slice
        )

        if raw_predictions_df is None or raw_predictions_df.empty:
            print("警告: 没有生成原始预测数据或原始预测数据为空。无法继续重采样。")
            return final_predictions_array

        # --- 核心修正点：在重采样之前，精确截取 raw_predictions_df 的长度 ---
        # 确保 raw_predictions_df 严格只包含 num_steps 个点
        # raw_predictions_df 的第一个点就是 start_hour 对应的预测值（索引为0）

        if len(raw_predictions_df) < num_steps:
            print(f"警告: Predictor 返回的原始预测点数量 ({len(raw_predictions_df)}) 少于期望 ({num_steps})。")
            return final_predictions_array

        # 严格截取 num_steps 个点，不多不少
        trimmed_predictions_df = raw_predictions_df.iloc[:num_steps].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
        # 默认是截取后的原始预测结果的 Predicted_Value 数组
        final_predictions_array = trimmed_predictions_df['Predicted_Value'].values

        output_step_multiplier = int(round(output_step_hours / ORIGINAL_MODEL_STEP))

        if output_step_multiplier == 1:  # 输出步长就是原始步长
            # 已经从 trimmed_predictions_df 获取，无需额外操作
            pass
        else:
            original_values_to_process = trimmed_predictions_df['Predicted_Value'].values
            # 计算可以被 output_step_multiplier 整除的最大长度，确保能正确重塑
            # 任何末尾无法构成完整一组的数据点都会被舍弃
            effective_length = (len(original_values_to_process) // output_step_multiplier) * output_step_multiplier
            if effective_length == 0:
                # 如果没有足够的数据进行分组平均，则结果为空数组
                final_predictions_array = np.array([])
            else:
                # 截取数据以匹配有效长度
                truncated_values = original_values_to_process[:effective_length]
                final_predictions_array = truncated_values.reshape(-1, output_step_multiplier).mean(axis=1)
            # --- 修改结束 ---

        # 最终返回
        if final_predictions_array.size > 0:
            return final_predictions_array
        else:
            return np.array([])

    except Exception as e:
        print(f"\n主程序执行过程中发生严重错误: {e}")
        return np.array([])


if __name__ == '__main__':
    print("--- 应用程序启动 ---")

    # 示例用法：
    # 在循环中多次调用 PV_gen.RTPV_Predict
    print("\n--- 在循环中多次调用 PV_gen.RTPV_Predict ---")

    current_start_hour = 4000.0
    total_prediction_duration = 10.0  # 假设总共要预测 10 小时
    step_duration_per_call = 2.5  # 每次调用预测 2.5 小时（5步 * 0.5小时/步）

    num_calls = int(
        total_prediction_duration / ORIGINAL_MODEL_STEP / (step_duration_per_call / ORIGINAL_MODEL_STEP))
    # 确保 num_calls 的计算是正确的，基于步数
    # num_steps_per_call = int(step_duration_per_call / PV_gen.ORIGINAL_MODEL_STEP)
    # total_num_steps = int(total_prediction_duration / PV_gen.ORIGINAL_MODEL_STEP)
    # num_calls = total_num_steps // num_steps_per_call # 整除

    for i in range(num_calls):
        start_time_for_this_call = current_start_hour + i * step_duration_per_call
        num_steps_for_this_call = int(step_duration_per_call / ORIGINAL_MODEL_STEP)

        print(f"\n--- 循环调用 {i + 1}/{num_calls}: 预测从 {start_time_for_this_call} 小时开始 ---")
        # 调用 PV_gen 模块中的 RTPV_Predict 函数
        predicted_values = RTPV_Predict(start_time_for_this_call, num_steps_for_this_call,
                                               1.0)  # 示例：输出步长为 1.0 小时

        if predicted_values.size > 0:
            print(f"获取到的预测结果 ({predicted_values.size} 个值):")
            print(predicted_values)
        else:
            print("本次调用未获取到预测结果。")

    print("\n--- 循环调用示例结束 ---")

    # 你仍然可以使用之前的单个示例调用
    print("\n--- 单次调用示例：预测从第 4000 小时开始预测 10 步，输出步长 0.5 小时（默认） ---")
    result1 = RTPV_Predict(4000, 10, 0.5)
    print("返回的结果类型:", type(result1))
    print("返回的数组内容:", result1)

    # 尝试预测没有足够数据的时间范围
    print("\n--- 单次调用示例：尝试预测没有足够数据的时间范围 ---")
    result5 = RTPV_Predict(8000, 2, 0.5)
    print("返回的结果类型:", type(result5))
    print("结果是否为空:", result5.size == 0)
    print("返回的数组内容:", result5)

    print("\n--- 应用程序结束 ---")