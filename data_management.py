# data_management.py
import pandas as pd


# 热动态模型
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
thermal_df = pd.read_csv('RC.csv')
# 确认时间间隔为 0.5 小时，生成时间序列 (小时)
time = thermal_df['TIME'].values  # 原始数据以 0.5 为步长
# 室外温度 (°C)
external_temp = thermal_df['Tout'].values
# 空间热负荷 (kJ/hr -> W)
# 注意: 1 kJ/hr = 0.2778 W
Q_heat = thermal_df['QHEAT_Zone1'].values * 0.2778
Q_cool = thermal_df['QCOOL_Zone1'].values * 0.2778
Q_space = Q_heat - Q_cool
Q_in = thermal_df['Qin_kJph'].values * 0.2778
# 通风供气温度 (°C)
vent_temp = thermal_df['TAIR_fresh'].values
# 通风流量 (kg/hr -> kg/s)
# 注意: 1 kg/hr = 1/3600 kg/s
vent_flow = thermal_df['Mrate_kgph'].values / 3600
# 实测室内温度 (°C)
measured_temp = thermal_df['TAIR_Zone1'].values


