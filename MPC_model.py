# -*- coding: utf-8 -*-
import warnings
import traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime
import os
import csv
from power_demand import power_demand
from PV_gen import RTPV_Predict
from docplex.mp.model import Model
import param_manage as pm
from input_manage import data_clip
import sys
from RL_model import RLAgent

# 全局变量，用于存储从 measure.csv 读取的实际测量数据
_global_actual_data_df = None


def load_actual_measurements_from_csv(file_path="measure.csv"):
    """
    从 CSV 文件加载一年的真实测量数据并进行单位换算。
    """
    global _global_actual_data_df
    try:
        df = pd.read_csv(file_path)
        df['TIME'] = df['TIME'].astype(float)

        # 将 kJ/h 转换为 kW (1 kW = 3600 kJ/h, 1/3600 ≈ 0.0002778)
        # 注意：之前使用的是 0.2778，这里修正为正确的 1/3600
        conversion_factor = 3600.0
        df['demand_measure'] = df['demand_measure'] / conversion_factor
        df['pv_measure'] = df['pv_measure'] / conversion_factor
        print(f"单位换算: 已将 'demand_measure' 和 'pv_measure' 从 kJ/h 转换为 kW (除以 {conversion_factor})。")

        _global_actual_data_df = df
        print(f"成功从 {file_path} 加载真实测量数据。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请确保 'measure.csv' 存在。")
        # --- 新增这部分来打印完整的Traceback ---
        print("--- 以下是完整的错误追溯信息 ---")
        traceback.print_exc()
        print("---------------------------------")
        # -----------------------------------------
        sys.exit(1)
    except KeyError as e:
        print(f"错误: 'measure.csv' 缺少必需的列: {e}。")
        # --- 新增这部分来打印完整的Traceback ---
        print("--- 以下是完整的错误追溯信息 ---")
        traceback.print_exc()
        print("---------------------------------")
        # -----------------------------------------
        sys.exit(1)
    except Exception as e:
        print(f"加载 'measure.csv' 时发生错误: {e}")
        # --- 新增这部分来打印完整的Traceback ---
        print("--- 以下是完整的错误追溯信息 ---")
        traceback.print_exc()
        print("---------------------------------")
        # -----------------------------------------
        sys.exit(1)


def get_actual_measurements(current_time_h: float):
    """
    根据当前时间点（小时数）从全局实际数据中获取真实的需求和光伏值。
    """
    global _global_actual_data_df
    if _global_actual_data_df is None:
        raise RuntimeError("实际测量数据未加载。请先调用 load_actual_measurements_from_csv()。")

    try:
        time_diff = (_global_actual_data_df['TIME'] - current_time_h).abs()
        closest_row_index = time_diff.idxmin()
        actual_data = _global_actual_data_df.loc[closest_row_index]
        return actual_data['demand_measure'], actual_data['pv_measure']
    except Exception as e:
        print(f"获取 {current_time_h}h 的实际测量值时发生错误: {e}")
        # --- 新增这部分来打印完整的Traceback ---
        print("--- 以下是完整的错误追溯信息 ---")
        traceback.print_exc()
        print("---------------------------------")
        # -----------------------------------------
        raise


def get_scalar_from_possibly_array(value):
    """
    安全地从可能为列表、NumPy 数组或标量本身的值中提取标量。
    """
    if isinstance(value, (list, np.ndarray)):
        return float(value[0]) if len(value) > 0 else None
    try:
        return float(value)
    except (ValueError, TypeError):
        # --- 新增这部分来打印完整的Traceback ---
        print("--- 以下是完整的错误追溯信息 ---")
        traceback.print_exc()
        print("---------------------------------")
        # -----------------------------------------
        return value


class MPC_solver:
    def __init__(self, battery_capacity, current_time_dt: datetime.datetime, battery_rated_C=0.25,
                 battery_efficiency=0.95, time_pred: float = 0.5):
        self.time_pred = time_pred
        reference_start_of_year = datetime.datetime(current_time_dt.year, 1, 1, 0, 0, 0)
        time_difference: datetime.timedelta = current_time_dt - reference_start_of_year
        total_hours_since_ref = time_difference.total_seconds() / 3600.0
        self.time_stamp = current_time_dt
        self.current_time = round(total_hours_since_ref / 0.5) * 0.5
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        self.battery_rated_power_kW = battery_capacity * battery_rated_C
        self.BIG_M = 1e8

    def _get_tou_prices(self, daily_tou_prices_24h: list, mpc_step_h: float, mpc_horizon_num: int):
        if not (len(daily_tou_prices_24h) == 24):
            raise ValueError("daily_tou_prices_24h 必须是包含24个价格的列表。")
        current_hour_of_day_float = self.time_stamp.hour + self.time_stamp.minute / 60.0
        predicted_prices = []
        for i in range(mpc_horizon_num):
            absolute_start_hour_in_forecast = current_hour_of_day_float + (i * mpc_step_h)
            hour_index_in_24h_cycle = int(absolute_start_hour_in_forecast) % 24
            predicted_prices.append(daily_tou_prices_24h[hour_index_in_24h_cycle])
        return predicted_prices

    def solve_upper_mpc(self, initial_energy, step_upper: int = 24, horizon_upper: int = 7):
        horizion_pred = int(horizon_upper * step_upper / self.time_pred)
        try:
            (T_amb_list, Q_in_list, vent_flow_list, measured_temp, Tin_t, measure_total_power,
             Twall_t_dict_0, Tsoil_t_list, tank_df, P_pan_kjh) = data_clip(self.current_time, horizion_pred)
        except Exception as e:
            print(f"调用 data_clip 失败: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------
            return None

        power_sim = power_demand(current_time=self.current_time, step=step_upper, step_pre=self.time_pred,
                                 horizon=horizion_pred)
        try:
            total_power_forecast = power_sim.generate_total_demand(
                Tin_t, T_amb_list, Q_in_list, vent_flow_list, Tsoil_t_list,
                Twall_t_dict_0, tank_df, P_pan_kjh, floor_types=['top', 'middle', 'bottom'])
            PV_predicted_values = RTPV_Predict(self.current_time, horizion_pred, step_upper)
        except Exception as e:
            print(f"生成预测失败: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------
            return None

        total_power_forecast = total_power_forecast[:horizon_upper]
        PV_predicted_values = PV_predicted_values[:horizon_upper]

        mdl = Model("Upper_MPC")
        E = mdl.continuous_var_list(horizon_upper + 1, name='E', lb=0, ub=self.battery_capacity)
        curtailment = mdl.continuous_var_list(horizon_upper, name='curtailment', lb=0)
        buy_from_grid = mdl.continuous_var_list(horizon_upper, name='buy_from_grid', lb=0)
        sell_to_grid = mdl.continuous_var_list(horizon_upper, name='sell_to_grid', lb=0)
        mdl.minimize(sum(curtailment[k] for k in range(horizon_upper)))
        mdl.add_constraint(E[0] == initial_energy, ctname="initial_energy")
        for k in range(horizon_upper):
            mdl.add_constraint(
                E[k + 1] - E[k] == PV_predicted_values[k] - total_power_forecast[k] - curtailment[k] + buy_from_grid[
                    k] - sell_to_grid[k],
                ctname=f"energy_balance_{k}")
            mdl.add_constraint(curtailment[k] <= PV_predicted_values[k], ctname=f"curtailment_limit_{k}")

        try:
            sol = mdl.solve(log_output=False)
            if sol:
                return [sol.get_value(E[k]) for k in range(horizon_upper + 1)]
            else:
                print("上层 MPC 求解失败。")
                return None
        except Exception as e:
            print(f"求解上层 MPC 时发生错误: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------
            return None

    def solve_lower_mpc(self, E_ini_bat, E_target_bat: float, step_lower: int = 1, horizon_lower: int = 24,
                        forecast_demand_adjusted: list = None, forecast_pv_adjusted: list = None,
                        last_total_power_forecast_raw: list = None, last_pv_predicted_values_raw: list = None,
                        actual_demand_this_hour: float = None, actual_pv_this_hour: float = None,
                        current_env_data_from_clip: dict = None):
        total_power_forecast_for_mpc = forecast_demand_adjusted
        PV_predicted_values_for_mpc = forecast_pv_adjusted
        Price_from_grid = self._get_tou_prices(pm.C_buy, step_lower, horizon_lower)
        Price_to_grid = self._get_tou_prices(pm.C_sell, step_lower, horizon_lower)
        comparison_metrics = {}

        if last_total_power_forecast_raw and last_pv_predicted_values_raw and actual_demand_this_hour is not None and actual_pv_this_hour is not None:
            predicted_demand_prev_mpc = get_scalar_from_possibly_array(last_total_power_forecast_raw)
            predicted_pv_prev_mpc = get_scalar_from_possibly_array(last_pv_predicted_values_raw)
            demand_error = actual_demand_this_hour - predicted_demand_prev_mpc if predicted_demand_prev_mpc is not None else None
            pv_error = actual_pv_this_hour - predicted_pv_prev_mpc if predicted_pv_prev_mpc is not None else None
            comparison_metrics = {'demand_error_raw': demand_error, 'pv_error_raw': pv_error}

        mdl = Model("Lower_MPC")
        E_bat = mdl.continuous_var_list(horizon_lower + 1, name='E_bat', lb=0, ub=self.battery_capacity)
        P_grid_buy = mdl.continuous_var_list(horizon_lower, name='P_grid_buy', lb=0)
        P_grid_sell = mdl.continuous_var_list(horizon_lower, name='P_grid_sell', lb=0)
        P_charge = mdl.continuous_var_list(horizon_lower, name='P_charge', lb=0, ub=self.battery_rated_power_kW)
        P_discharge = mdl.continuous_var_list(horizon_lower, name='P_discharge', lb=0, ub=self.battery_rated_power_kW)
        binary_charge = mdl.binary_var_list(horizon_lower, name='binary_charge')
        binary_discharge = mdl.binary_var_list(horizon_lower, name='binary_discharge')

        cost_objective = sum(
            P_grid_buy[t] * Price_from_grid[t] * step_lower - P_grid_sell[t] * Price_to_grid[t] * step_lower for t in
            range(horizon_lower))
        target_energy_objective = 1000 * (E_bat[horizon_lower] - E_target_bat) ** 2
        mdl.minimize(cost_objective + target_energy_objective)
        mdl.add_constraint(E_bat[0] == E_ini_bat, ctname="initial_battery_energy")

        for t in range(horizon_lower):
            mdl.add_constraint(
                PV_predicted_values_for_mpc[t] + P_grid_buy[t] + P_discharge[t] * self.battery_efficiency >=
                total_power_forecast_for_mpc[t] + P_charge[t] / self.battery_efficiency + P_grid_sell[t],
                ctname=f"energy_balance_lower_{t}")
            mdl.add_constraint(
                E_bat[t + 1] == E_bat[t] + (P_charge[t] * self.battery_efficiency - P_discharge[
                    t] / self.battery_efficiency) * step_lower,
                ctname=f"battery_state_{t}")
            mdl.add_constraint(binary_charge[t] + binary_discharge[t] <= 1, ctname=f"mutual_exclusion_battery_{t}")
            mdl.add_constraint(P_charge[t] <= binary_charge[t] * self.battery_rated_power_kW,
                               ctname=f"charge_limit_M_{t}")
            mdl.add_constraint(P_discharge[t] <= binary_discharge[t] * self.battery_rated_power_kW,
                               ctname=f"discharge_limit_M_{t}")

        try:
            sol = mdl.solve(log_output=False)
            if sol:
                first_step_P_buy = sol.get_value(P_grid_buy[0])
                first_step_P_sell = sol.get_value(P_grid_sell[0])
                hourly_economic_cost = (first_step_P_buy * Price_from_grid[0] - first_step_P_sell * Price_to_grid[
                    0]) * step_lower
                return sol.get_value(E_bat[1]), comparison_metrics, hourly_economic_cost
            else:
                print(f"下层 MPC 在 {self.time_stamp} 求解失败。")
                return None, None, None
        except Exception as e:
            print(f"求解下层 MPC 时发生错误: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------
            return None, None, None


def run_mpc_simulation(
        start_time: datetime.datetime,
        simulation_duration_days: int,
        battery_capacity: float,
        initial_battery_energy: float,
        PV_max: float,
        demand_max: float,
        rl_agent: RLAgent,
        mode: str,
        model_dir: str,
        battery_rated_C: float = 0.25,
        battery_efficiency: float = 0.95,
        actual_data_file="measure.csv",
        rl_data_output_file="rl_training_data.csv"
):
    print(f"\n--- MPC 仿真开始 (模式: {mode.upper()}) ---")
    print(f"仿真开始时间: {start_time.strftime('%Y-%m-%d')}, 持续天数: {simulation_duration_days} 天")

    # 常量
    LOWER_MPC_STEP_HOURS = 1
    LOWER_MPC_HORIZON_HOURS = 24

    load_actual_measurements_from_csv(actual_data_file)
    best_daily_reward = -float('inf')
    os.makedirs(model_dir, exist_ok=True)

    # 初始化仿真状态变量
    current_sim_time = start_time
    current_battery_energy = initial_battery_energy
    last_demand_error_raw = None
    last_pv_error_raw = None
    last_total_power_forecast_series_raw = None
    last_pv_predicted_values_series_raw = None

    # 初始化用于前向填充的变量
    last_known_tank_temp = 20.0
    last_known_amb_temp = 15.0
    # ... 为其他可能缺失的环境数据添加 last_known 变量

    for day_offset in range(simulation_duration_days):
        print(f"\n--- 仿真第 {day_offset + 1}/{simulation_duration_days} 天 ---")
        current_day_cumulative_reward = 0.0

        upper_mpc = MPC_solver(battery_capacity, current_sim_time)
        planned_daily_storage = upper_mpc.solve_upper_mpc(initial_energy=current_battery_energy)
        if planned_daily_storage is None:
            print("上层 MPC 失败，终止仿真。")
            break
        current_day_target_energy_raw = planned_daily_storage[1] if len(
            planned_daily_storage) > 1 else current_battery_energy

        for hour_of_day in range(24):
            mpc_solver_hourly = MPC_solver(battery_capacity, current_sim_time)
            current_time_h = mpc_solver_hourly.current_time

            try:
                actual_demand_now, actual_pv_now = get_actual_measurements(current_time_h)
                horizon_pred_lower = int(LOWER_MPC_HORIZON_HOURS * LOWER_MPC_STEP_HOURS / mpc_solver_hourly.time_pred)
                raw_env_data_tuple = data_clip(current_time_h, horizon_pred_lower)
                raw_env_data_clip = {
                    'T_amb_list': raw_env_data_tuple[0], 'Q_in_list': raw_env_data_tuple[1],
                    'vent_flow_list': raw_env_data_tuple[2], 'measured_temp': raw_env_data_tuple[3],
                    'Tin_t': raw_env_data_tuple[4], 'measure_total_power': raw_env_data_tuple[5],
                    'Twall_t_dict_0': raw_env_data_tuple[6], 'Tsoil_t_list': raw_env_data_tuple[7],
                    'tank_df': raw_env_data_tuple[8], 'P_pan_kjh': raw_env_data_tuple[9]
                }
            except Exception as e:
                print(f"在 {current_sim_time} 获取数据失败: {e}")
                # --- 新增这部分来打印完整的Traceback ---
                print("--- 以下是完整的错误追溯信息 ---")
                traceback.print_exc()
                print("---------------------------------")
                # -----------------------------------------
                break

            power_sim = power_demand(current_time=current_time_h, step=LOWER_MPC_STEP_HOURS,
                                     step_pre=mpc_solver_hourly.time_pred, horizon=horizon_pred_lower)
            raw_demand_forecast = power_sim.generate_total_demand(
                raw_env_data_clip['Tin_t'], raw_env_data_clip['T_amb_list'], raw_env_data_clip['Q_in_list'],
                raw_env_data_clip['vent_flow_list'], raw_env_data_clip['Tsoil_t_list'],
                raw_env_data_clip['Twall_t_dict_0'], raw_env_data_clip['tank_df'], raw_env_data_clip['P_pan_kjh'],
                floor_types=['top', 'middle', 'bottom'])
            raw_pv_forecast = RTPV_Predict(current_time_h, horizon_pred_lower, LOWER_MPC_STEP_HOURS)

            s_t_dict = {
                'current_hour_of_day': current_sim_time.hour,
                'current_day_of_week': current_sim_time.weekday(),
                'current_battery_energy': current_battery_energy,
                'raw_upper_mpc_target_energy': current_day_target_energy_raw,
                'raw_demand_forecast_this_hour': get_scalar_from_possibly_array(raw_demand_forecast),
                'raw_pv_forecast_this_hour': get_scalar_from_possibly_array(raw_pv_forecast),
                'last_demand_error_raw': last_demand_error_raw,
                'last_pv_error_raw': last_pv_error_raw,
                'current_price_from_grid': get_scalar_from_possibly_array(
                    mpc_solver_hourly._get_tou_prices(pm.C_buy, LOWER_MPC_STEP_HOURS, 1)),
                'T_amb_current': get_scalar_from_possibly_array(raw_env_data_clip.get('T_amb_list')),
                'Q_in_current': get_scalar_from_possibly_array(raw_env_data_clip.get('Q_in_list')),
                'vent_flow_current': get_scalar_from_possibly_array(raw_env_data_clip.get('vent_flow_list')),
                'measured_temp_current': get_scalar_from_possibly_array(raw_env_data_clip.get('measured_temp')),
                'Tin_t_current': get_scalar_from_possibly_array(raw_env_data_clip.get('Tin_t')),
                'measure_total_power_current': get_scalar_from_possibly_array(
                    raw_env_data_clip.get('measure_total_power')),
                'Tsoil_t_current': get_scalar_from_possibly_array(raw_env_data_clip.get('Tsoil_t_list')),
                'P_pan_kjh_current': get_scalar_from_possibly_array(raw_env_data_clip.get('P_pan_kjh')),
            }
            tank_df = raw_env_data_clip.get('tank_df')
            if isinstance(tank_df, pd.DataFrame) and not tank_df.empty and 'TankTemp' in tank_df.columns:
                s_t_dict['TankTemp_current'] = get_scalar_from_possibly_array(tank_df.iloc[0]['TankTemp'])
            else:
                s_t_dict['TankTemp_current'] = None

            # --- 前向填充缺失值 ---
            if s_t_dict['TankTemp_current'] is not None:
                last_known_tank_temp = s_t_dict['TankTemp_current']
            else:
                s_t_dict['TankTemp_current'] = last_known_tank_temp

            if s_t_dict['T_amb_current'] is not None:
                last_known_amb_temp = s_t_dict['T_amb_current']
            else:
                s_t_dict['T_amb_current'] = last_known_amb_temp

            rl_action = rl_agent.choose_action(s_t_dict, deterministic=(mode == 'eval'))

            adj_demand_forecast = [f * (1 + rl_action['demand_correction_factor']) for f in raw_demand_forecast]
            adj_pv_forecast = [f * (1 + rl_action['pv_correction_factor']) for f in raw_pv_forecast]
            adj_demand_forecast = np.clip(adj_demand_forecast, 0, demand_max).tolist()
            adj_pv_forecast = np.clip(adj_pv_forecast, 0, PV_max).tolist()
            adj_E_target_bat = max(0.0, min(battery_capacity * 0.95,
                                            current_day_target_energy_raw + rl_action['battery_target_adjustment_kWh']))

            next_hour_battery_energy, comparison_metrics, hourly_cost = mpc_solver_hourly.solve_lower_mpc(
                E_ini_bat=current_battery_energy, E_target_bat=adj_E_target_bat,
                forecast_demand_adjusted=adj_demand_forecast,
                forecast_pv_adjusted=adj_pv_forecast,
                last_total_power_forecast_raw=last_total_power_forecast_series_raw,
                last_pv_predicted_values_raw=last_pv_predicted_values_series_raw,
                actual_demand_this_hour=actual_demand_now,
                actual_pv_this_hour=actual_pv_now,
                current_env_data_from_clip=raw_env_data_clip
            )

            if hourly_cost is None:
                print(f"下层 MPC 在 {current_sim_time} 求解失败，跳过当天剩余时间。")
                break

            if mode == 'train':
                reward = -hourly_cost
                current_day_cumulative_reward += reward

                next_s_t_dict = s_t_dict.copy()
                next_s_t_dict['current_battery_energy'] = next_hour_battery_energy
                if comparison_metrics:
                    next_s_t_dict['last_demand_error_raw'] = comparison_metrics.get('demand_error_raw')
                    next_s_t_dict['last_pv_error_raw'] = comparison_metrics.get('pv_error_raw')

                done = (hour_of_day == 23)
                rl_agent.add_to_replay_buffer(s_t_dict, rl_action, reward, next_s_t_dict, done)
                rl_agent.train()

            current_battery_energy = next_hour_battery_energy
            current_sim_time += datetime.timedelta(hours=LOWER_MPC_STEP_HOURS)
            last_total_power_forecast_series_raw = raw_demand_forecast
            last_pv_predicted_values_series_raw = raw_pv_forecast
            if comparison_metrics:
                last_demand_error_raw = comparison_metrics.get('demand_error_raw')
                last_pv_error_raw = comparison_metrics.get('pv_error_raw')

        if mode == 'train':
            print(f"--- 第 {day_offset + 1} 天结束。当天累积奖励: {current_day_cumulative_reward:.2f} ---")
            latest_path = os.path.join(model_dir, "latest")
            rl_agent.save_model(latest_path)
            print(f"已将最新模型保存到 {latest_path}_*.pth")

            if current_day_cumulative_reward > best_daily_reward:
                best_daily_reward = current_day_cumulative_reward
                print(f"*** 新的最佳奖励纪录: {best_daily_reward:.2f}! 正在保存最佳模型。 ***")
                best_path = os.path.join(model_dir, "best")
                rl_agent.save_model(best_path)

    print(f"\n--- MPC 仿真结束 ---")


if __name__ == "__main__":
    args = {}
    while True:
        print("\n请选择运行模式:")
        print("  1: 训练 (Train)")
        print("  2: 评估 (Eval)")
        mode_choice = input("请输入选项 (1 或 2): ")
        if mode_choice in ['1', '2']:
            args['mode'] = 'train' if mode_choice == '1' else 'eval'
            break
        else:
            print("输入无效，请重新输入。")

    args['load_path'] = None
    if args['mode'] == 'eval':
        print("\n评估模式必须加载一个已训练的模型。")
        should_load = 'y'
    else:
        should_load = input("是否要加载一个已有模型继续训练? (y/n): ").lower()

    if should_load == 'y':
        while True:
            print("\n请选择要加载的模型:")
            print("  1: 加载表现最好的模型 (best)")
            print("  2: 加载上次训练的模型 (latest)")
            print("  3: 输入自定义路径前缀")
            load_choice = input("请输入选项 (1, 2, 或 3): ")
            model_dir_default = './rl_models_save'
            if load_choice == '1':
                args['load_path'] = os.path.join(model_dir_default, "best")
                break
            elif load_choice == '2':
                args['load_path'] = os.path.join(model_dir_default, "latest")
                break
            elif load_choice == '3':
                args['load_path'] = input("请输入模型的路径前缀 (例如 './rl_models_save/ep_10'): ")
                break
            else:
                print("输入无效，请重新输入。")

    while True:
        try:
            args['sim_days'] = int(input("\n请输入仿真的总天数 (例如 30): "))
            break
        except ValueError:
            print("输入无效，请输入一个整数。")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------

    args['model_dir'] = './rl_models_save'

    print("\n--- 配置确认 ---")
    print(f"  运行模式: {args['mode']}")
    print(f"  加载模型: {args['load_path'] or '不加载'}")
    print(f"  仿真天数: {args['sim_days']}")
    print("--------------------")
    input("按 Enter 键开始运行...")

    start_sim_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
    battery_capacity_kWh = 50.0
    initial_battery_kWh = 25.0

    state_features = [
        'current_hour_of_day', 'current_day_of_week',
        'current_battery_energy', 'raw_upper_mpc_target_energy',
        'raw_demand_forecast_this_hour', 'raw_pv_forecast_this_hour',
        'last_demand_error_raw', 'last_pv_error_raw',
        'current_price_from_grid',
        'T_amb_current', 'Q_in_current', 'vent_flow_current',
        'measured_temp_current', 'Tin_t_current', 'measure_total_power_current',
        'Tsoil_t_current', 'P_pan_kjh_current', 'TankTemp_current'
    ]
    rl_action_space_limits = {
        'demand_correction_factor': (-0.2, 0.2),
        'pv_correction_factor': (-0.2, 0.2),
        'battery_target_adjustment_kWh': (-5.0, 5.0)
    }

    rl_agent = RLAgent(
        action_space_limits=rl_action_space_limits,
        state_feature_names=state_features,
        model_path=args['model_dir']
    )

    if args['load_path']:
        print(f"\n正在从 '{args['load_path']}' 加载模型...")
        if not rl_agent.load_model(args['load_path']) and args['mode'] == 'eval':
            print("评估模式下模型加载失败，程序终止。")
            sys.exit(1)

    run_mpc_simulation(
        start_time=start_sim_time,
        simulation_duration_days=args['sim_days'],
        battery_capacity=battery_capacity_kWh,
        initial_battery_energy=initial_battery_kWh,
        PV_max=129942.0851 * 0.2278,
        demand_max=140620.9898 * 0.2278,
        rl_agent=rl_agent,
        mode=args['mode'],
        model_dir=args['model_dir'],
        actual_data_file="measure.csv",
        rl_data_output_file=f"rl_{args['mode']}_data.csv"
    )

    print("\n--- 程序执行完毕 ---")
    input("按 Enter 键退出。")