#!/usr/bin/env python
# -*- coding: utf-8 -*-

from power_demand import total_power_demand
from power_generation import total_power_generation
from EV_demand import total_EV
from docplex.mp.model import Model


def solve_upper_mpc(current_time, step, horizon, initial_energy, battery_capacity):
    """
    上层 MPC：目标为经济（例如最小化电池能量变化的二次成本），决策变量为每步的电池储能量。
    输入：
      - current_time: 当前时刻（可为时间戳或 datetime 对象）
      - step: 时间步长
      - horizon: 预测时域步数
      - initial_energy: 电池初始储能
      - battery_capacity: 电池最大容量
    输出：
      - planned_storage: 长度 horizon+1 的数组，表示每步的规划储能量（包括初始时刻）
    """
    # 获取预测数据（数组长度为 horizon）
    demand_forecast = total_power_demand(current_time, step, horizon).get_array()
    generation_forecast = total_power_generation(current_time, step, horizon).get_array()
    ev_forecast = total_EV(current_time, step, horizon).get_array()

    mdl = Model("Upper_MPC")

    # 决策变量：电池储能量 E[0...horizon]
    E = [mdl.continuous_var(lb=0, ub=battery_capacity, name=f"E_{t}") for t in range(horizon + 1)]

    # 初始状态
    mdl.add_constraint(E[0] == initial_energy, "Initial_Battery")

    # 这里构造一个简单的成本函数（储能变化的二次项），实际中可结合电价、损耗等构建更合理的目标函数
    cost_expr = mdl.sum((E[t + 1] - E[t]) ** 2 for t in range(horizon))
    mdl.minimize(cost_expr)

    solution = mdl.solve()
    if solution is None:
        raise Exception("上层 MPC 无可行解！")

    planned_storage = [solution.get_value(E[t]) for t in range(horizon + 1)]
    return planned_storage


def solve_lower_mpc(current_time, step, horizon, initial_energy, battery_capacity, planned_final_storage,
                    max_EV_charge):
    """
    下层 MPC：目标为经济（例如最小化电池储能变化的二次成本），决策变量为每步的电池储能量和 EV 的充电量。
    要求最后的电池储能量与上层规划一致。

    输入：
      - current_time: 当前时刻
      - step: 时间步长
      - horizon: 时域步数
      - initial_energy: 电池初始储能
      - battery_capacity: 电池最大容量
      - planned_final_storage: 上层规划的最终储能量（下层结束时电池能量应达到该值）
      - max_EV_charge: 每个时刻 EV 充电上限（单位：kWh），可与 EV 数量耦合
    输出：
      - battery_storage_solution: 每步的电池储能量数组（长度 horizon+1）
      - EV_charge_solution: 每步 EV 充电量数组（长度 horizon）
    """
    # 获取预测数据
    demand_forecast = total_power_demand(current_time, step, horizon).get_array()
    generation_forecast = total_power_generation(current_time, step, horizon).get_array()
    ev_forecast = total_EV(current_time, step, horizon).get_array()

    mdl = Model("Lower_MPC")

    # 决策变量：电池储能 E[0...horizon] 和 EV 每步充电量 EV_charge[0...horizon-1]
    E = [mdl.continuous_var(lb=0, ub=battery_capacity, name=f"E_{t}") for t in range(horizon + 1)]
    EV_charge = [mdl.continuous_var(lb=0, ub=max_EV_charge, name=f"EV_charge_{t}") for t in range(horizon)]

    # 初始状态
    mdl.add_constraint(E[0] == initial_energy, "Initial_Battery")

    # 电池动态约束：简单的能量平衡模型
    # E[t+1] = E[t] + (发电量 - 负荷) - EV充电量
    for t in range(horizon):
        net_energy = generation_forecast[t] - demand_forecast[t]
        mdl.add_constraint(E[t + 1] == E[t] + net_energy - EV_charge[t], f"Battery_Dynamics_{t}")

    # EV 充电量与 EV 数量耦合：假设每个 EV 每步最多可充电 10 kWh（可调整）
    max_charge_per_EV = 10
    for t in range(horizon):
        mdl.add_constraint(EV_charge[t] <= ev_forecast[t] * max_charge_per_EV, f"EV_Charge_Limit_{t}")

    # 下层结束时电池储能量必须与上层规划结果一致
    mdl.add_constraint(E[horizon] == planned_final_storage, "Final_Battery")

    # 同样构造简单的成本函数（储能变化的二次成本）
    cost_expr = mdl.sum((E[t + 1] - E[t]) ** 2 for t in range(horizon))
    mdl.minimize(cost_expr)

    solution = mdl.solve()
    if solution is None:
        raise Exception("下层 MPC 无可行解！")

    battery_storage_solution = [solution.get_value(E[t]) for t in range(horizon + 1)]
    EV_charge_solution = [solution.get_value(EV_charge[t]) for t in range(horizon)]
    return battery_storage_solution, EV_charge_solution


def main():
    # 参数设置（单位、数值可根据实际情况调整）
    current_time = 0  # 当前时刻（例如：时间戳或 datetime 对象）
    step = 1  # 时间步长（例如：1 小时）
    horizon = 10  # 时域步数
    initial_energy = 50  # 电池初始储能（例如：50 kWh）
    battery_capacity = 100  # 电池容量（例如：100 kWh）
    max_EV_charge = 20  # EV 每步充电上限（单位：kWh）

    # 上层 MPC：得到每步规划的电池储能量（上层输出）
    planned_storage = solve_upper_mpc(current_time, step, horizon, initial_energy, battery_capacity)
    print("上层 MPC 规划的电池储能量：")
    for t, energy in enumerate(planned_storage):
        print(f"步长 {t}: {energy:.2f}")

    # 下层 MPC：要求最终电池储能量达到上层规划值
    planned_final_storage = planned_storage[-1]
    battery_solution, EV_charge_solution = solve_lower_mpc(current_time, step, horizon, initial_energy,
                                                           battery_capacity, planned_final_storage, max_EV_charge)

    print("\n下层 MPC 计算得到的电池储能量：")
    for t, energy in enumerate(battery_solution):
        print(f"步长 {t}: {energy:.2f}")

    print("\n下层 MPC 计算得到的 EV 充电量：")
    for t, charge in enumerate(EV_charge_solution):
        print(f"步长 {t}: {charge:.2f}")


if __name__ == "__main__":
    main()
