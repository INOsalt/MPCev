import pandas as pd
import numpy as np

from input_manage import data_clip
from thermal_model import ThermalModel
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score
from HVAC_power import HVACSimulator
import param_manage as pm  # 使用统一参数
from people_behave import equip_light_demand  # 导入设备用电行为模块


class power_demand:
    def __init__(self, current_time, step, step_pre, horizon):
        """
        热力能源系统仿真器
        :param rc_params_path: 墙体RC参数文件路径
        :param model_params: 核心模型参数列表
        :param gp_model_name: 高斯过程模型名称
        """
        self.current_time = current_time
        self.step = step
        self.step_pre = step_pre
        self.horizon = horizon #整数 代表多少个时间步长

        # 结果缓存区
        self.prediction_results = None
        self.ac_sign_list = []
        self.sc_sign_list = []
        self.ac_cop_list = []
        self.sc_cop_list = []
        self.ac_power_list = []
        self.sc_power_list = []
        self.total_power_list_pre = []
        self.sh_power_list = None
        self.pan_power_list = None

    def chiller_power(self, Tin_t, Tamb_t_list, Qin_t_list, vent_flow,
                Tsoil_t_list, Twall_t_dict_0, tanktem_data, floor_types=None):
        # 如果 floor_types 没有被传入，或者传入的是 None，就使用默认值
        if floor_types is None:
            floor_types = ['top', 'middle', 'bottom']

        """
        执行完整仿真流程
        :param prediction_hours: 预测时长（小时）
        :return: 四元组 (总功率需求, AC功率, SC功率, 设备功率)
        """
        wall_params_map = {
            'top': pm.wall_params_top,
            'middle': pm.wall_params_middle,
            'bottom': pm.wall_params_bottom
        }

        gp_model_map = {
            'top': pm.gp_model_name_top,
            'middle': pm.gp_model_name_middle,
            'bottom': pm.gp_model_name_bottom
        }

        SC_Demand_series_ori = None  # 用于累加 Qspace_t_list_corrected
        AC_Demand_series = None  # 用于累加 Qahu_t_list

        for current_floor_type in floor_types:
            # Select the correct wall parameters for the current floor type
            selected_wall_params = wall_params_map.get(current_floor_type)
            selected_gp_params = gp_model_map.get(current_floor_type)

            if selected_wall_params is None:
                print(
                    f"Warning: No specific wall parameters found for floor type: {current_floor_type}. Defaulting to 'middle' wall parameters.")
                selected_wall_params = pm.wall_params_middle  # <--- HERE: Default to middle
                selected_gp_params = pm.gp_model_name_middle
                # Optionally, you might want to consider if this floor type should still be processed
                # For now, we'll proceed with 'middle' parameters.



            self.thermal_model = ThermalModel(
                pm.model_params,  # This remains constant (general model params)
                selected_wall_params,  # Pass the specific or defaulted wall parameters for this floor
                selected_gp_params,  # This remains constant
                current_floor_type  # Pass the current floor type (e.g., 'top', 'middle')
            )  # thermal_model对象

            # 执行预测并缓存结果
            Tin_t_list, Twall_t_dict_list, Qzone_t_list, Qahu_t_list, Qspace_t_list, Qspace_t_list_corrected \
                = self.thermal_model.predict_peiod(
                self.horizon,
                Tamb_t_list,
                Tin_t,
                Qin_t_list,
                self.step_pre,
                vent_flow,
                self.current_time,
                Tsoil_t_list,
                Twall_t_dict_0
            )

            # --- 对 Qspace_t_list_corrected 进行累加 ---
            # 确保 Qspace_t_list_corrected 是一个 NumPy 数组
            Qspace_t_list_corrected = np.array(Qspace_t_list_corrected)
            if SC_Demand_series_ori is None:
                SC_Demand_series_ori = Qspace_t_list_corrected
            else:
                SC_Demand_series_ori = SC_Demand_series_ori + Qspace_t_list_corrected

            SH_Demand_series = np.where(SC_Demand_series_ori > 0, SC_Demand_series_ori, 0)
            self.sh_power_list = SH_Demand_series
            SC_Demand_series = np.where(SC_Demand_series_ori < 0, -SC_Demand_series_ori, 0)
            # --- 对 Qahu_t_list 进行累加 ---
            # 确保 Qahu_t_list 是一个 NumPy 数组
            Qahu_t_list = np.array(Qahu_t_list)
            if AC_Demand_series is None:
                AC_Demand_series = Qahu_t_list
            else:
                AC_Demand_series = AC_Demand_series + Qahu_t_list


        # === 初始化模拟器 ===
        simulator = HVACSimulator(
            dt_step=self.step_pre *3600 ,#秒 预测步长
            coef_ac_path=pm.AC_BQ_Path,
            coef_sc_path=pm.SC_BQ_Path,
        )

        # 初始温度与状态

        AC_T_top = tanktem_data['TankAC1'].item()
        AC_T_middle = tanktem_data['TankAC3'].item()
        AC_T_bottom = tanktem_data['TankAC5'].item()
        AC_state = 0  # 状态通常是整数，不是从数据中读取的

        SC_T_top = tanktem_data['TankSC1'].item()
        SC_T_middle = tanktem_data['TankSC3'].item()
        SC_T_bottom = tanktem_data['TankSC5'].item()
        SC_state = 0

        # === 结果存储 ===
        AC_T_top_all, ACsign_list = [], []

        # === 模拟循环 ===
        for t in range(self.horizon):
            Qe_tp_AC = AC_Demand_series[t]
            Qe_tp_SC = SC_Demand_series[t]
            Tout_t = Tamb_t_list[t]

            ACpower, SCpower, _, AC_T_top, AC_T_middle, AC_T_bottom, AC_state = simulator.calculate_hvac_power(
                t, AC_T_top, AC_T_middle,AC_T_bottom, AC_state,
                SC_T_top, SC_T_middle, SC_T_bottom, SC_state,
                Qe_tp_AC, Qe_tp_SC, Tout_t
            )

            COP_pred_AC = simulator.predict_cop(Qe_tp_AC, Tout_t, simulator.coef_ac_path)
            COP_pred_SC = simulator.predict_cop(Qe_tp_SC, Tout_t, simulator.coef_sc_path)

            # 存储结果
            AC_T_top_all.append(AC_T_top)
            self.ac_sign_list.append(AC_state)
            self.sc_sign_list.append(SC_state)
            self.ac_cop_list.append(COP_pred_AC)
            self.sc_cop_list.append(COP_pred_SC)
            self.ac_power_list.append(ACpower)
            self.sc_power_list.append(SCpower)


    def generate_total_demand(self,Tin_t, Tamb_t_list, Qin_t_list, vent_flow,
                Tsoil_t_list, Twall_t_dict_0, tanktem_data, P_pan_kjh, floor_types=None):
        """生成总功率需求"""
        self.chiller_power(Tin_t, Tamb_t_list, Qin_t_list, vent_flow,
                Tsoil_t_list, Twall_t_dict_0, tanktem_data, floor_types)


        # 获取行为设备功耗
        file_path = 'POWER.csv'
        equip_list,light_list = equip_light_demand(file_path, self.horizon, self.current_time, self.step_pre)  # 假设equip返回对象有get_power_profile方法
        self.equip_power_list = equip_list + light_list

        P_pan_w = P_pan_kjh * 0.2778  # 正确的换算
        self.pan_power_list = np.full_like(self.equip_power_list, P_pan_w)

        # 计算总功耗
        self.total_power_list_pre = (self.pan_power_list + self.equip_power_list) * 3 + self.ac_power_list + self.sc_power_list

        # --- 重新采样到 self.step ---
        # 计算新步长与原始步长的比率
        if self.step_pre == 0:
            raise ValueError("self.step_pre 不能为零。")

        resample_ratio = int(self.step / self.step_pre)

        # 确保数组的长度是 resample_ratio 的倍数
        # 如果不是，您可能需要决定如何处理剩余数据
        # 为简单起见，我们将截断或填充，如果不能完美整除。

        # 计算新数组的长度
        new_length = len(self.total_power_list_pre) // resample_ratio

        # 重塑数组并沿新轴取平均值
        self.total_power_list = self.total_power_list_pre[:new_length * resample_ratio].reshape(new_length,
                                                                                                resample_ratio).mean(
            axis=1)

        return self.total_power_list  # 或者您需要返回的任何内容

    @property
    def result_summary(self):
        """生成结果摘要"""
        return {
            'total_demand_pre_resample': self.total_power_list_pre,  # 为清晰起见重命名
            'total_demand_resampled': self.total_power_list,  # 添加重新采样后的列表
            'ac_power': self.ac_power_list,
            'sc_power': self.sc_power_list,
            'equipment_power': self.equip_power_list,
            'pan_power': self.pan_power_list,  # 添加盘管功率
            'sh_power': self.sh_power_list,  # 添加SH功率
            'ac_cop': self.ac_cop_list,
            'sc_cop': self.sc_cop_list
        }

def load_test_input_data():
    """
    从实际数据集加载数据并进行单位转换。
    """
    file_path = 'RC.csv'
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    T_amb_list = df['Tout'].values
    Q_in = df['Qin_kJph'].values * 0.2778
    vent_flow = df['Mrate_kgph'].values / 3600
    measured_temp = df['TAIR_Zone1'].values
    Tin_t = measured_temp[0]

    # 重新导入 measure_total_power
    measure_total_power = df['measure_total_power'].values * 0.2778

    Twall_t_dict = {}
    wall_temp_columns = ['TSI_S4', 'TSI_S6', 'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10', 'TSI_S11', 'TSI_S12',
                         'TSI_S13', 'TSI_S14']
    for wall in wall_temp_columns:
        Twall_t_dict[wall] = measured_temp[0]

    Tsoil_t_list = df['T_Soil'].values

    tanktem_file_path = 'tanktem.csv'  # 确保这个文件存在
    tanktem_df = pd.read_csv(tanktem_file_path)
    tanktem_df.columns = tanktem_df.columns.str.strip()  # 清理列名，去除空格

    tanktem_data = tanktem_df.iloc[0:1]

    # 返回 measure_total_power
    return T_amb_list, Q_in, vent_flow, Tin_t, Tsoil_t_list, Twall_t_dict, tanktem_data, measure_total_power

# --- 主测试函数 ---
if __name__ == "__main__":
    matplotlib.rc("font", family='Microsoft YaHei')
    print("开始功率需求仿真测试...")

    # --- 1. 定义仿真参数 ---
    current_time_start = 0
    step_pre = 0.5
    step = 1
    horizon_hours = 8760
    horizon_steps = int(horizon_hours / step_pre)

    # --- 2. 加载测试输入数据 ---
    try:
        (Tamb_t_list_full, Qin_t_list_full, vent_flow_list_full, measured_temp, Tin_t_initial, measure_total_power_full,
         Twall_t_dict_0_initial, Tsoil_t_list_full, tanktem_data_initial, P_pan_kjh) \
            = data_clip(0, 17520)

        Tamb_t_list = Tamb_t_list_full[:horizon_steps]
        Qin_t_list = Qin_t_list_full[:horizon_steps]
        vent_flow_list = vent_flow_list_full[:horizon_steps]
        Tsoil_t_list = Tsoil_t_list_full[:horizon_steps]
        # 切片 measure_total_power 以匹配仿真时长
        measure_total_power = measure_total_power_full[:horizon_steps]

        Tin_t = Tin_t_initial
        Twall_t_dict_0 = Twall_t_dict_0_initial
        tanktem_data = tanktem_data_initial


    except FileNotFoundError as e:
        print(f"加载测试数据时出错: {e}。请确保 'RC.csv'、'POWER.csv' 和 COP 系数文件在正确的目录中。")
        print("退出测试。")
        exit()
    except Exception as e:
        print(f"数据加载过程中发生未知错误: {e}")
        print("退出测试。")
        exit()

    # --- 3. 初始化 power_demand 对象 ---
    power_sim = power_demand(
        current_time=current_time_start,
        step=step,
        step_pre=step_pre,
        horizon=horizon_steps
    )

    # --- 4. 生成总需求 ---
    print(f"\n正在生成 {horizon_hours} 小时的总需求，基本步长为 {step_pre} 小时...")
    total_power_resampled = power_sim.generate_total_demand(
        Tin_t,
        Tamb_t_list,
        Qin_t_list,
        vent_flow_list,
        Tsoil_t_list,
        Twall_t_dict_0,
        tanktem_data,
        P_pan_kjh,
        floor_types=['top', 'middle', 'bottom']
    )

    # --- 5. 显示结果 ---
    print("\n--- 仿真结果 ---")
    summary = power_sim.result_summary

    print(f"原始总功率列表 (step_pre={power_sim.step_pre}h) 长度: {len(summary['total_demand_pre_resample'])}")
    print(f"重新采样后的总功率列表 (step={power_sim.step}h) 长度: {len(summary['total_demand_resampled'])}")

    # 确保实测数据与重新采样后的仿真数据长度一致，以便进行对比
    # 如果 horizon_steps / resample_ratio 与 measure_total_power 的长度不匹配，需要重新采样 measure_total_power
    resample_ratio_for_measured = int(step / step_pre)
    measured_resampled_length = len(measure_total_power) // resample_ratio_for_measured
    # 对实测数据也进行与仿真结果相同的重新采样（平均值）
    measured_total_power_resampled = measure_total_power[
                                     :measured_resampled_length * resample_ratio_for_measured].reshape(
        measured_resampled_length, resample_ratio_for_measured).mean(axis=1)

    min_len_comparison = min(len(total_power_resampled), len(measured_total_power_resampled))

    print("\n--- 仿真结果与实测数据对比 ---")
    print(f"实测总功率（原始步长）长度: {len(measure_total_power)}")
    print(f"实测总功率（重新采样到 {step}h 步长）长度: {len(measured_total_power_resampled)}")

    # 计算评估指标
    # 确保用于计算的数组长度一致
    mse = mean_squared_error(measured_total_power_resampled[:min_len_comparison],
                             total_power_resampled[:min_len_comparison])
    rmse = np.sqrt(mse)
    r2 = r2_score(measured_total_power_resampled[:min_len_comparison], total_power_resampled[:min_len_comparison])

    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"R 方 (R2_score): {r2:.2f}")

    # 打印前5个值进行直观对比
    print("\n重新采样后总功率前5个值 (W):")
    print(total_power_resampled[:5])
    print("\n重新采样后实测总功率前5个值 (W):")
    print(measured_total_power_resampled[:5])

    # --- 6. 绘图 ---
    print("\n生成图表...")
    plt.figure(figsize=(12, 6))

    time_resampled = np.arange(0, min_len_comparison * step, step)  # 使用最短长度绘制

    plt.plot(time_resampled, total_power_resampled[:min_len_comparison], 'o-', label=f'仿真总功率 (步长={step}h)',
             markersize=4)
    plt.plot(time_resampled, measured_total_power_resampled[:min_len_comparison], 'x-',
             label=f'实测总功率 (步长={step}h)', markersize=4)

    plt.xlabel('时间 (小时)')
    plt.ylabel('功率 (W)')
    plt.title('仿真总功率需求与实测数据对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n测试完成。")