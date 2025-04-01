import pandas as pd
import numpy as np
from thermal_model import ThermalModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from HVAC_power import HVACSimulator
import param_manage as pm  # 使用统一参数
from data_management import demand
from people_behave import equip  # 导入设备用电行为模块


class power_demand:
    def __init__(self,current_time, step, horizon):
        """
        热力能源系统仿真器
        :param rc_params_path: 墙体RC参数文件路径
        :param model_params: 核心模型参数列表
        :param gp_model_name: 高斯过程模型名称
        """
        self.current_time = current_time
        self.step = step
        self.horizon = horizon

        # 结果缓存区
        self.prediction_results = None
        self.ac_cop_list = []
        self.sc_cop_list = []
        self.ac_power_list = []
        self.sc_power_list = []
        self.total_power_list = []

    def chiller_power(self):
        """
        执行完整仿真流程
        :param prediction_hours: 预测时长（小时）
        :return: 四元组 (总功率需求, AC功率, SC功率, 设备功率)
        """
        (time_horzion, Tamb_t_list, Tin_t, Qin_t_list,
         step_pre, vent_flow, time_now) = demand() #数据传递

        self.thermal_model = ThermalModel(
            pm.model_params,
            pm.wall_params,
            pm.gp_model_name
        ) #thermal_model对象

        # 执行预测并缓存结果
        Tin_t_list, Twall_t_dict_list, Qzone_t_list, Qahu_t_list, Qspace_t_list, Qspace_t_list_corrected \
            = self.thermal_model.predict_peiod(
            self.horizon,
            Tamb_t_list,
            Tin_t,
            Qin_t_list,
            step_pre,
            vent_flow,
            self.current_time
        )

        # === 数据读取与预处理 ===
        cop_data = pd.read_csv('COP.csv')
        tank_data = pd.read_csv('tank.csv')
        tanktem_data = pd.read_csv('tanktem.csv')

        for df in [cop_data, tank_data, tanktem_data]:
            df.columns = df.columns.str.strip()

        merged_data = pd.concat([
            cop_data.set_index('TIME'),
            tank_data.set_index('TIME')
        ], axis=1)

        AC_Demand_series = merged_data['ACdemandKJPH'].values
        SC_Demand_series = merged_data['SCdemandKJPH'].values
        Tout_series = merged_data['Tout'].values
        T_top_actual = merged_data['TAC_P2AC_C'].values
        ACsign_actual = merged_data['AHUsign'].values
        ACCOP = merged_data['AHUCOP'].values
        SCsign_actual = merged_data['SCsign'].values
        SCCOP = merged_data['SCSHCOP'].values
        actual_ACpower = merged_data['ACPowerKHPH'].values
        actual_SCpower = merged_data['SCPowerKJPH'].values

        # === 初始化模拟器 ===
        simulator = HVACSimulator(
            dt_step=pm.DT_MICRO,
            coef_ac_path='ACdemandKJPH_BQ_Model_Coefficients.csv',
            coef_sc_path='SCdemandKJPH_BQ_Model_Coefficients.csv',
            tanktem_data=tanktem_data
        )

        # 初始温度与状态
        AC_T_top = AC_T_middle = AC_T_bottom = 24.0
        AC_state = 0

        # === 结果存储 ===
        AC_T_top_all, ACsign_list = [], []
        AC_COP_preds, SC_COP_preds = [], []
        ACPowers, SCPowers = [], []

        # === 模拟循环 ===
        for t in range(len(merged_data) - 1):
            Qe_tp_AC = AC_Demand_series[t]
            Qe_tp_SC = SC_Demand_series[t]
            Tout_t = Tout_series[t]

            ACpower, SCpower, _, AC_T_top, AC_T_middle, AC_T_bottom, AC_state = simulator.calculate_hvac_power(
                t, AC_T_top, AC_T_middle, AC_T_bottom, AC_state,
                Qe_tp_AC, Qe_tp_SC,
                AC_Demand_series, Tout_series
            )

            COP_pred_AC = simulator.predict_cop(Qe_tp_AC, Tout_t, simulator.coef_ac_path)
            COP_pred_SC = simulator.predict_cop(Qe_tp_SC, Tout_t, simulator.coef_sc_path)

            # 存储结果
            AC_T_top_all.append(AC_T_top)
            ACsign_list.append(AC_state)
            AC_COP_preds.append(COP_pred_AC)
            SC_COP_preds.append(COP_pred_SC)
            ACPowers.append(ACpower)
            SCPowers.append(SCpower)

        # 生成总需求
        self._generate_total_demand()

        return (
            self.total_power_list,
            self.ac_power_list,
            self.sc_power_list,
            self.equip_power_list
        )


    def _generate_total_demand(self):
        """生成总功率需求"""
        # 获取行为设备功耗
        time_points = len(self.ac_power_list)
        self.equip_power_list = equip().get_power_profile(time_points)  # 假设equip返回对象有get_power_profile方法

        # 计算总功耗
        self.total_power_list = []#补充

    @property
    def result_summary(self):
        """生成结果摘要"""
        return {
            'total_demand': self.total_power_list,
            'ac_power': self.ac_power_list,
            'sc_power': self.sc_power_list,
            'equipment_power': self.equip_power_list,
            'ac_cop': self.ac_cop_list,
            'sc_cop': self.sc_cop_list
        }
