# param_manage.py
import os
from pathlib import Path
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
wall_temp_columns = ['TSI_S4', 'TSI_S6', 'TSI_S7', 'TSI_S8', 'TSI_S9', 'TSI_S10', 'TSI_S11', 'TSI_S12',
                         'TSI_S13', 'TSI_S14']

model_params = [0.0028, 0.054, 190679918.65329826] # Removed trailing comma
# 你的相对路径定义
init_model_path = 'initialization/files'
model_dir = Path(init_model_path)
# 确保文件夹及其所有父文件夹都存在
model_dir.mkdir(parents=True, exist_ok=True)

# Combine base path with GP model filenames
gp_model_name_top = os.path.join(init_model_path, "gp_model_best_top.pkl")
gp_model_name_middle = os.path.join(init_model_path, "gp_model_best_middle.pkl")
gp_model_name_bottom = os.path.join(init_model_path, "gp_model_best_bottom.pkl")

train_set_RTPV = os.path.join(init_model_path, "train_set_RTPV")
train_label_RTPV = os.path.join(init_model_path, "train_label_RTPV")
test_set_RTPV = os.path.join(init_model_path, "test_set_RTPV")
test_label_RTPV = os.path.join(init_model_path, "test_label_RTPV")

# Combine base path with wall parameters CSV filenames
wall_params_top = pd.read_csv(os.path.join(init_model_path, 'rc_params_curvefit_top.csv'), index_col=0)
wall_params_middle = pd.read_csv(os.path.join(init_model_path, 'rc_params_curvefit_middle.csv'), index_col=0)
wall_params_bottom = pd.read_csv(os.path.join(init_model_path, 'rc_params_curvefit_bottom.csv'), index_col=0)

# SC COP
AC_BQ_Path = os.path.join(init_model_path, 'ACdemandKJPH_BQ_Model_Coefficients.csv')
SC_BQ_Path = os.path.join(init_model_path, 'SCdemandKJPH_BQ_Model_Coefficients.csv')


#主网电价
C_buy = [0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205,
         0.5802, 0.5802, 0.9863, 0.9863, 0.5802, 0.5802, 0.9863, 0.9863,
         0.9863, 0.9863, 0.9863, 0.5802, 0.5802, 0.5802, 0.5802, 0.5802]
C_sell = [0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
          0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
          0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453]
C_buy_business = [
    0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867, 0.2867,
    0.7093, 0.7093,
    1.1864, 1.1864,
    0.7093, 0.7093,
    1.1864, 1.1864, 1.1864, 1.1864, 1.1864,
    0.7093, 0.7093, 0.7093, 0.7093, 0.7093
]
C_tran = 0.1834