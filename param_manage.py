# param_manage.py
import pandas as pd

# 水箱参数
CP = 4.2          # 比热容 kJ/(kg·°C)
RHO = 1000        # 水的密度 kg/m³
V_TANK = 1.5      # 水箱体积 m³
H_TANK = 1.2      # 水箱高度 m
U = 1.08 / 3600   # 热损系数
K = 0.58 / 1000   # 导热系数
L = 0.5           # 节间距离

#Chiller参数
AC_Cap = 11.2 * 3600 #KJPH
delta_TAC_nom = 4.99 #度
delta_TAC_USER_nom = 5
SC_Cap = 26.3 * 3600
delta_TSC_nom = 2 #度
delta_TSC_USER_nom = 2

# 控制策略参数
T_THRESHOLD_AC_OFF = 5.1
T_THRESHOLD_AC_ON = 7.0
T_THRESHOLD_SC_OFF = 13.1
T_THRESHOLD_SC_ON = 15

# 计算步长
DT_MICRO = 450

#热模型
rc_params_path = 'rc_params_curvefit.csv',
model_params = [0.0028, 0.054, 190679918.65329826],
gp_model_name = "gp_model_9"
wall_params = pd.read_csv('rc_params_curvefit.csv', index_col=0)