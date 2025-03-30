import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import param_manage as pm


class HVACSimulator:
    def __init__(self, dt_step, coef_ac_path, coef_sc_path, tanktem_data=None):
        self.dt_step = dt_step
        self.dt_micro = pm.DT_MICRO
        self.coef_ac_path = coef_ac_path
        self.coef_sc_path = coef_sc_path
        self.tanktem_data = tanktem_data
        self._precompute_constants()

    def _precompute_constants(self):
        self.M_total = pm.V_TANK * pm.RHO
        self.n_nodes = 3
        self.M_node = self.M_total / self.n_nodes
        self.c_node = self.M_node * pm.CP
        self.D_tank = np.sqrt((4 * pm.V_TANK) / (np.pi * pm.H_TANK))
        self.A = np.pi * (self.D_tank / 2) ** 2
        self.A_cond = self.A

    def simulate_tank_at_t(self, t, T_top, T_middle, T_bottom, Demand_series, T_amb_series, state_prev):
        max_micro_steps = self.dt_step // self.dt_micro
        Demand = Demand_series[t]
        T_amb = T_amb_series[t]
        current_state = state_prev

        m_dot_bottom = 11.2 * 3600 / (pm.CP * 4.99) / 3600 if current_state == 1 else 0
        m_dot_top = Demand / (pm.CP * 5) / 3600
        delta_TAC = (Demand / 1920 / pm.CP if Demand > 0 else 5) if current_state == 1 else 0
        delta_User = 5 if Demand > 0 else 0

        if self.tanktem_data is not None and t > 0 and t <= len(self.tanktem_data):
            T_top = self.tanktem_data['TankAC1'].iloc[t - 1]
            T_middle = self.tanktem_data['TankAC3'].iloc[t - 1]
            T_bottom = self.tanktem_data['TankAC5'].iloc[t - 1]

        for _ in range(max_micro_steps):
            dT_top = (m_dot_top * pm.CP * (T_bottom + delta_User - T_top) +
                      m_dot_bottom * pm.CP * (T_middle - T_top) +
                      pm.K * self.A_cond * (T_middle - T_top) / pm.L -
                      pm.U * self.A * (T_top - T_amb)) / self.c_node
            dT_top = (m_dot_top * pm.CP * (T_bottom + delta_User - T_top) +
                      m_dot_bottom * pm.CP * (T_middle - T_top) +
                      pm.K * self.A_cond * (T_middle - T_top) / pm.L -
                      pm.U * self.A * (T_top - T_amb)) / self.c_node

            dT_middle = ((m_dot_top * pm.CP * (T_top - T_middle) +
                          m_dot_bottom * pm.CP * (T_bottom - T_middle)) +
                         pm.K * self.A_cond * (T_top - T_middle + T_bottom - T_middle) / pm.L) / self.c_node

            dT_bottom = (m_dot_top * pm.CP * (T_middle - T_bottom) +
                         m_dot_bottom * pm.CP * ((T_top - delta_TAC) - T_bottom) +
                         pm.K * self.A_cond * (T_middle - T_bottom) / pm.L -
                         pm.U * self.A * (T_bottom - T_amb)) / self.c_node

            T_top += dT_top * self.dt_micro
            T_middle += dT_middle * self.dt_micro
            T_bottom += dT_bottom * self.dt_micro

            # 控制状态更新
            if current_state == 1 and (T_top - pm.T_THRESHOLD_OFF) < 0:
                current_state = 0
            elif current_state == 0 and (T_top - pm.T_THRESHOLD_OFF) >= (pm.T_THRESHOLD_ON - pm.T_THRESHOLD_OFF):
                current_state = 1

        return T_top, T_middle, T_bottom, current_state

    def predict_cop(self, Qe, Tout, coef_path):
        if Qe == 0:
            return np.nan
        coef_df = pd.read_csv(coef_path)
        coef_dict = dict(zip(coef_df['Feature'], coef_df['Coefficient']))
        features = {
            '1/Qe': 1 / Qe,
            'Qe': Qe,
            'Tcwi/Qe': Tout / Qe,
            'Tcwi^2/Qe': (Tout ** 2) / Qe,
            'Tcwi': Tout,
            'Qe*Tcwi': Qe * Tout,
            'Tcwi^2': Tout ** 2,
            'Qe*Tcwi^2': Qe * Tout ** 2
        }
        y_inv = coef_dict.get('Intercept', 0.0)
        for key, value in features.items():
            y_inv += coef_dict.get(key, 0.0) * value
        return 1 / y_inv if y_inv != 0 else np.nan

    def calculate_hvac_power(self, t, T_top, T_middle, T_bottom, AC_state,
                              Qe_tp_AC, Qe_tp_SC, Demand_series, Tout_series):
        Tout_t = Tout_series[t]
        T_top, T_middle, T_bottom, AC_state = self.simulate_tank_at_t(
            t, T_top, T_middle, T_bottom, Demand_series, Tout_series, AC_state
        )

        COP_AC = self.predict_cop(Qe_tp_AC, Tout_t, self.coef_ac_path)
        COP_SC = self.predict_cop(Qe_tp_SC, Tout_t, self.coef_sc_path)

        ACpower = AC_state * Qe_tp_AC / COP_AC if COP_AC and not np.isnan(COP_AC) else 0
        SCpower = Qe_tp_SC / COP_SC if COP_SC and not np.isnan(COP_SC) else 0
        HVAC_power = ACpower + SCpower

        return ACpower, SCpower, HVAC_power, T_top, T_middle, T_bottom, AC_state


def main():
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

    # === 结果合并 ===
    results_df = pd.DataFrame({
        'T_top': AC_T_top_all,
        'T_top_actual': T_top_actual[:len(AC_T_top_all)],
        'ACsign': ACsign_list,
        'ACsign_actual': ACsign_actual[:len(ACsign_list)],
        'Predicted_ACCOP': AC_COP_preds,
        'Predicted_SCCOP': SC_COP_preds,
        'actual_ACCOP': ACCOP[:len(AC_COP_preds)],
        'actual_SCCOP': SCCOP[:len(SC_COP_preds)],
        'Predicted_ACpower': ACPowers,
        'Predicted_SCpower': SCPowers,
        'actual_ACpower': actual_ACpower[:len(ACsign_list)],
        'actual_SCpower': actual_SCpower[:len(ACsign_list)],
    })

    # === 可视化与误差评估 ===
    for label, col in [('AC', 'ACpower'), ('SC', 'SCpower')]:
        results_df[f'Predicted_{col}'] = results_df[f'Predicted_{col}'].fillna(0)
        results_df[f'actual_{col}'] = results_df[f'actual_{col}'].fillna(0)

        mse = mean_squared_error(results_df[f'actual_{col}'], results_df[f'Predicted_{col}'])
        r2 = r2_score(results_df[f'actual_{col}'], results_df[f'Predicted_{col}'])
        print(f'Error between {label}power: MSE = {mse:.4f}, R2 = {r2:.4f}')

        plt.figure(figsize=(12, 5))
        plt.plot(results_df[f'Predicted_{col}'], label=f'Predicted_{col}', color='blue')
        plt.plot(results_df[f'actual_{col}'], label=f'actual_{col}', color='red', linestyle='--')
        plt.xlabel('Time Step (0.5h)')
        plt.ylabel('POWER')
        plt.title(f'{label} Power Prediction')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    # COP 可视化
    plt.figure(figsize=(12, 5))
    plt.plot(results_df['Predicted_ACCOP'], label='Predicted AC COP (BQ)', color='blue')
    plt.plot(results_df['actual_ACCOP'], label='Actual AC COP', color='red', linestyle='--')
    plt.title('Annual AC COP Prediction')
    plt.xlabel('Time Step (0.5h)')
    plt.ylabel('COP')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(results_df['Predicted_SCCOP'], label='Predicted SC COP (BQ)', color='blue')
    plt.plot(results_df['actual_SCCOP'], label='Actual SC COP', color='red', linestyle='--')
    plt.title('Annual SC COP Prediction')
    plt.xlabel('Time Step (0.5h)')
    plt.ylabel('COP')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存（如需要）
    # results_df.to_csv('annual_COP_prediction.csv', index=False)

# === 只在该文件被执行时运行主程序 ===
if __name__ == "__main__":
    main()