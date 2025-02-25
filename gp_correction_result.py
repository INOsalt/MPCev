from thermal_model_nostar import ThermalModel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from standard_class import StandardizedGP
from sklearn.metrics import mean_squared_error


def load_real_data(file_path):
    """
    从实际数据集加载数据并进行单位转换。

    输入:
    - file_path: 数据文件路径 (CSV)

    输出:
    - time: 时间序列 (小时)
    - external_temp: 室外温度 (°C)
    - Q_heat: 空间热负荷 (W)
    - vent_temp: 通风供气温度 (°C)
    - vent_flow: 通风质量流量 (kg/s)
    - measured_temp: 实测室内温度 (°C)
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 确认时间间隔为 0.5 小时，生成时间序列 (小时)
    time = df['TIME'].values * 0.5  # 原始数据以 0.5 为步长

    # 室外温度 (°C)
    external_temp = df['Tout'].values

    # 墙温
    # wall_temp_columns = ['TSI_S4', 'TSI_S6',#roof
    #                  'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10',#window
    #                  'TSI_S11', 'TSI_S12', 'TSI_S13', 'TSI_S14']#'TSI_S1', 'TSI_S2', 'TSI_S3',  'TSI_S5',# ext wall
    # Twall_t_dict = {}
    # for wall in wall_temp_columns:
    #     Twall_t_dict[wall] = df[wall].values

    # 空间热负荷 (kJ/hr -> W)
    # 注意: 1 kJ/hr = 0.2778 W
    Q_heat = df['QHEAT_Zone1'].values * 0.2778
    Q_cool = df['QCOOL_Zone1'].values * 0.2778
    Q_space = Q_heat - Q_cool
    # Q_in = np.zeros(len(df['QHEAT_Zone1']))
    Q_in = df['Qin_kJph'].values * 0.2778

    # 通风供气温度 (°C)
    vent_temp = df['TAIR_fresh'].values

    # 通风流量 (kg/hr -> kg/s)
    # 注意: 1 kg/hr = 1/3600 kg/s
    vent_flow = df['Mrate_kgph'].values / 3600

    # 实测室内温度 (°C)
    measured_temp = df['TAIR_Zone1'].values

    return time, external_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, measured_temp, Q_space


def gaussian_process_correction(time, external_temp, measured, predicted, iteration,
                                initial_downsample_rate=1, max_retries=10):
    """
    使用 StandardizedGP 类实现高斯过程回归校正灰箱模型的误差，并处理可能的收敛问题。

    参数:
    - time: 时间序列 (小时, 一维数组)
    - external_temp: 室外温度 (°C, 一维数组)
    - measured_temp: 实测室内温度 (°C, 一维数组)
    - predicted_temp: 灰箱模型预测的室内温度 (°C, 一维数组)
    - initial_downsample_rate: 初始降采样率，用于减少训练数据规模 (默认: 1, 不降采样)
    - max_retries: 最大重试次数，用于调整参数解决收敛问题

    返回:
    - corrected_temp: 校正后的室内温度 (°C, 一维数组)
    - gp_model: 训练好的 StandardizedGP 模型对象
    """
    residual = measured - predicted  # 计算残差（目标值）
    X = np.column_stack((time, external_temp))  # 构建输入特征

    downsample_rate = initial_downsample_rate  # 设置初始降采样率
    retry_count = 0  # 当前重试次数

    while retry_count <= max_retries:
        try:
            # 初始化 StandardizedGP 模型，设置当前降采样率
            gp_model = StandardizedGP(
                kernel=C(1.0, (1e-2, 1e5)) * RBF(length_scale=10.0, length_scale_bounds=(1e-3, 1e3)),
                n_restarts_optimizer=50,
                alpha=1e-2,
                downsample_rate=downsample_rate
            )

            # 训练模型
            gp_model.fit(X, residual)

            # 训练成功，打印核参数
            print(f"高斯过程收敛成功！降采样率: {downsample_rate}, 核参数: {gp_model.gp.kernel_}")

            # 使用模型预测校正值
            correction = gp_model.predict(X)
            corrected = predicted + correction

            # Save the trained Gaussian Process model
            gp_model.save(f"gp_model_{iteration}.pkl")
            print(f"Model saved to gp_model_{iteration}.pkl")
            return corrected, gp_model

        except Exception as e:
            # 捕获收敛失败的异常
            print(f"高斯过程收敛失败，重试中 (尝试次数: {retry_count + 1}/{max_retries})")
            print(f"异常信息: {e}")

            # 增大降采样率以减少训练数据规模
            downsample_rate *= 2
            print(f"调整降采样率至: {downsample_rate}")

            # 调整模型参数：增大核函数的长度尺度
            if retry_count == max_retries - 1:  # 最后一次重试
                print("尝试更改高斯过程模型参数...")
                gp_model.kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=20.0, length_scale_bounds=(1e-1, 1e3))

            retry_count += 1

    # 如果最终仍未收敛，返回未校正值
    print("高斯过程校正失败，返回未校正的预测值。")
    return predicted, None


def gaussian_process_apply(time, external_temp, predicted, iteration):
    """
    使用预训练的高斯过程模型校正灰箱模型的预测结果。

    参数:
    - time: 时间序列 (小时, 一维数组)
    - external_temp: 室外温度 (°C, 一维数组)
    - predicted: 灰箱模型预测的室内温度 (°C, 一维数组)
    - iteration: 模型迭代编号，用于加载对应保存的模型文件

    返回:
    - corrected_temp: 校正后的室内温度 (°C, 一维数组)
    - gp_model: 加载的预训练 StandardizedGP 模型对象（如果加载成功）
    """
    X = np.column_stack((time, external_temp))  # 构建输入特征
    print(f"X shape: {X.shape}")  # 应为 (n,2)

    try:
        # 从文件加载预训练模型
        gp_model = StandardizedGP.load(f"gp_model_{iteration}.pkl")
        print(f"成功加载预训练高斯过程模型: gp_model_{iteration}.pkl")
        print(gp_model.gp.kernel_)  # 应输出训练后的核参数表达式

        # 使用模型预测校正值
        correction = gp_model.predict(X)
        corrected = predicted + correction
        return corrected

    except Exception as e:
        # 处理模型加载失败的情况
        print(f"模型加载失败，返回未校正的预测值。错误信息: {e}")
        return predicted


def visualize_before_correction(time, measured_temp, predicted_temp, iteration):
    """
    在高斯过程校正之前可视化灰箱模型预测值与实测值的对比。

    参数:
    - time: 时间序列 (小时)
    - measured_temp: 实测室内温度 (°C)
    - predicted_temp: 灰箱模型预测的室内温度 (°C)
    """
    predicted_temp[0] = measured_temp[0]
    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_temp, label="Measured", linewidth=2)
    plt.plot(time, predicted_temp, label="Predicted(Grey-box Model)", linestyle='--', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Indoor Temperature (°C)")
    plt.legend()
    plt.title("Grey-box Model Prediction vs Measured Temperature (Before Correction)")
    plt.grid(True)
    # plt.savefig(f"before_correction_{iteration}.png", dpi=300)
    plt.show()


def visualize(time, measured_Q, predicted_Q, corrected_Q, iteration):
    """
    Visualize heat load results.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, measured_Q, label="Measured", linewidth=2)
    plt.plot(time, predicted_Q, label="Predicted", linestyle='--', linewidth=2, alpha=0.6)
    plt.plot(time, corrected_Q, label="Corrected", linestyle='-.', linewidth=2, alpha=0.6)
    plt.xlabel("Time (hours)")
    plt.ylabel("Heat Load (W)")
    plt.legend()
    plt.title("Heat Load Prediction and Correction")
    plt.grid(True)
    # plt.savefig(f"Q_results_{iteration}.png", dpi=300)
    plt.show()


def train_gp_correction_for_Q(thermal_model, time_horizon, Tamb_t_list, Tin_t_list, Qin_t_list, step_pre, vent_flow,
                              Q_space_measured, iteration, mode="train"):
    """
    使用 predict_period 进行预测并训练高斯过程校正模型

    参数:
    - thermal_model: ThermalModel 实例
    - time_horizon: 预测时间长度
    - Tamb_t_list: 环境温度列表
    - Tin_t_list: 室内温度列表
    - Qin_t_list: 内部热负荷列表
    - step_pre: 时间步长
    - vent_flow: 通风流量
    - Q_space_measured: 实测的空间热负荷

    返回:
    - corrected_Q: 校正后的热负荷预测值
    - gp_model: 训练好的高斯过程模型
    """
    # 使用 predict_period 进行预测
    Tin_list, Twall_dict_list, Qzone_list, Qahu_list, Qspace_list, _ = thermal_model.predict_peiod(
        time_horizon,
        Tamb_t_list,
        Tin_t_list[0],
        Qin_t_list,
        step_pre,
        vent_flow
    )

    # 构建时间序列
    time_hour = np.arange(len(Qspace_list)) * step_pre / 3600
    line_len = min(len(time_hour), len(Q_space_measured), len(Qspace_list), int(1e3))  # 输入输出预测的长度，用一个很大的表示全序列
    # print(len(Q_space_measured))
    # print(len(Qspace_list))

    visualize_before_correction(time_hour[:line_len], Q_space_measured[:line_len], Qspace_list[:line_len], iteration)
    visualize_before_correction(time_hour[:line_len], measured_temp[:line_len], Tin_list[:line_len], iteration)

    if mode == "train":

        # 进行高斯过程校正
        corrected_Q, gp_model = gaussian_process_correction(
            time_hour[:line_len],
            Tamb_t_list[:line_len],  # 确保长度匹配
            Q_space_measured[:line_len],  # 实测热负荷
            np.array(Qspace_list[:line_len]),  # 预测热负荷
            iteration,
            initial_downsample_rate=1,
            max_retries=10
        )
        visualize(time_hour[:line_len], Q_space_measured[:line_len], Qspace_list[:line_len], corrected_Q[:line_len],
                  iteration)
        return corrected_Q, gp_model, Qspace_list

    else:
        corrected_Q = gaussian_process_apply(
            time_hour[:line_len],
            Tamb_t_list[:line_len],  # 确保长度匹配
            np.array(Qspace_list[:line_len]),  # 预测热负荷
            iteration
        )

        # 可视化结果
        visualize(time_hour[:line_len], Q_space_measured[:line_len], Qspace_list[:line_len], corrected_Q[:line_len],
                  iteration)

        return corrected_Q, None, Qspace_list


# 在主函数或需要的地方添加以下代码
if __name__ == "__main__":
    # 加载数据
    file_path = 'RC.csv'
    time, external_temp, Q_heat, Q_cool, Q_in, vent_temp, vent_flow, measured_temp, Q_space = load_real_data(file_path)

    # 初始化模型参数
    # params = [0.0001, 0.001999, 0.00062838, 40723395.97479104, 200671880.13560498] #self.Rstar_win, self.Rstar_wall, self.Rair, self.Cstar, self.C_air
    params = [0.0028, 0.054, 190679918.65329826]
    # [0.002436990358271271, 0.019999999999999997, 0.00010728150932535978, 218600129.0216549, 16773920.364684984]#28432390.466448814
    wall_RC_params = pd.read_csv('rc_params_curvefit.csv', index_col=0)

    gp_model_name = "None"
    # 初始化模型
    thermal_model = ThermalModel(params, wall_RC_params, gp_model_name)

    # 设置预测参数
    time_horizon = len(time)  # 预测时间长度
    step_pre = 0.5 * 3600  # 时间步长（小时）

    # 准备输入数据
    Tamb_t_list = external_temp
    Tin_t_list = [measured_temp[0]]  # 只需要初始温度
    Qin_t_list = Q_in
    vent_flow_value = np.mean(vent_flow)  # 使用平均通风流量，或者可以传入完整的 vent_flow 列表

    for iteration in range(9, 10):#用来表示调用哪个模型
        # 训练高斯过程校正模型
        corrected_Q, gp_model, predicted_Q = train_gp_correction_for_Q(
            thermal_model,
            time_horizon,
            Tamb_t_list,
            Tin_t_list,
            Qin_t_list,
            step_pre,
            vent_flow_value,
            Q_space,
            iteration,
            mode="apply" #train是训练模式
        )

        # 计算和打印误差指标
        mse_before = mean_squared_error(Q_space[:len(predicted_Q)], predicted_Q)
        mse_after = mean_squared_error(Q_space[:len(corrected_Q)], corrected_Q)

        # Calculate and print error metrics
        print(f"Iteration: {iteration}")
        print(f"MSE before correction: {mse_before:.2f}")
        print(f"MSE after correction: {mse_after:.2f}")
        print(f"Improvement: {((mse_before - mse_after) / mse_before * 100):.2f}%")
        # 写入到文本文件
        file_path = "gp_correction_results.txt"  # 你可以根据需要修改文件路径
        with open(file_path, "a") as f:  # "a" 模式确保文件不存在时会创建，且内容追加到文件末尾
            f.write(f"Iteration: {iteration}\n")
            f.write(f"MSE before correction: {mse_before:.2f}\n")
            f.write(f"MSE after correction: {mse_after:.2f}\n")
            f.write(f"Improvement: {((mse_before - mse_after) / mse_before * 100):.2f}%\n")



