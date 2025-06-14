# Python program to compare BQ Model and MP Model for AC and SC chillers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures





def BQ_model(data, Qe_col, Tcwi_col, Tout_col,save_dir):
    # 移除 Qe=0 的数据点，防止除零错误
    data = data[(data[Qe_col] != 0) & (data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP'] > 0)].copy()


    # 取相关变量
    Qe = data[Qe_col]
    Tcwi = data[Tcwi_col]
    Tout = data[Tout_col]

    # 计算特征，避免 1/Qe 产生 inf
    X = pd.DataFrame({
        '1/Qe': 1 / Qe.replace(0, np.nan),  # 预防性替换 0 -> NaN
        'Qe': Qe,
        'Tcwi/Qe': Tout / Qe.replace(0, np.nan),
        'Tcwi^2/Qe': (Tout ** 2) / Qe.replace(0, np.nan),
        'Tcwi': Tout,
        'Qe*Tcwi': Qe * Tout,
        'Tcwi^2': Tout ** 2,
        'Qe*Tcwi^2': Qe * Tout ** 2
    })

    # 目标变量
    y = 1 / data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP']

    # 删除所有 NaN 行，确保 X 和 y 里没有 NaN/inf
    X = X.dropna()
    y = y.loc[X.index]  # 确保 y 和 X 的行数匹配

    # 训练模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # 计算误差
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # 保存模型系数
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    intercept = pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [model.intercept_]})
    result = pd.concat([coefficients, intercept])
    result.to_csv(save_dir/f'{Qe_col}_BQ_Model_Coefficients.csv', index=False)

    return model, mse, r2, y_pred


def MP_model(data, Qe_col, Tcwi_col, Tcwo_col, Tout_col,save_dir):
    data = data[(data[Qe_col] != 0) & (data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP'] > 0)].copy()
    Qe = data[Qe_col]
    Tcwi = data[Tcwi_col]
    Tcwo = data[Tcwo_col]
    Tout = data[Tout_col]

    X = pd.DataFrame({
        'Qe': Qe,
        'Tchwi': Tcwi,
        'Tcwi': Tout,
        'Qe^2': Qe ** 2,
        'Tchwi^2': Tcwi ** 2,
        'Qe*Tchwi': Qe * Tcwi,
        'Qe*Tcwi': Qe * Tout,
        'Qe*Tchwi*Tcwi': Qe * Tcwi * Tout,
        'Tchwi*Tcwi': Tcwi * Tout,
        'Tcwi^2': Tout ** 2
    })

    y = data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    intercept = pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [model.intercept_]})
    result = pd.concat([coefficients, intercept])
    result.to_csv(save_dir/f'{Qe_col}_MP_Model_Coefficients.csv', index=False)

    return model, mse, r2, y_pred


# # Load the datasets
# cop_data = pd.read_csv('COP.csv')
# cop_data.columns = cop_data.columns.str.strip()
#
# tank_data = pd.read_csv('tank.csv')
# tank_data.columns = tank_data.columns.str.strip()
#
# # Merge datasets on index instead of timestamp
# merged_data = pd.concat([cop_data, tank_data], axis=1)
#
# # Filter data for active periods only (when control signals are 1)
# AC_data = merged_data[merged_data['AHUsign'] == 1]
# SC_data = merged_data[merged_data['SCsign'] == 1]
#
# # Rated capacities (in kJ/hr)
# Cap_ACc = 11.2 * 3600  # AC chiller
# Cap_SCc = 26.3 * 3600  # SC chiller

# Constants
c = 4.2  # Specific heat capacity of water in kJ/(kg°C)
def COP_FITTING(merged_data,save_dir):
    AC_data = merged_data[merged_data['AHUsign'] == 1]
    SC_data = merged_data[merged_data['SCsign'] == 1]
    for chiller, data, Qe_col, Tcwi_col, Tcwo_col, Tout_col in [
        ('AC', AC_data, 'ACdemandKJPH', 'TAC_P2AC_C', 'TAC_AC2T_C', 'Tout'),
        ('SC', SC_data, 'SCdemandKJPH', 'TSC_P2SC_C', 'TSC_SC2T_C', 'Tout')
    ]:
    # for chiller, data, Qe_col, Tcwi_col, Tcwo_col, Tout_col in [
    #     ('AC', AC_data, 'Qe_AC', 'TAC_P2AC_C', 'TAC_AC2T_C', 'Tout'),
    #     ('SC', SC_data, 'Qe_SC', 'TSC_P2SC_C', 'TSC_SC2T_C', 'Tout')
    # ]:

        print(f'--- {chiller} Chiller ---')

        bq_model, bq_mse, bq_r2, bq_pred = BQ_model(data, Qe_col, Tcwi_col, Tout_col,save_dir)
        mp_model, mp_mse, mp_r2, mp_pred = MP_model(data, Qe_col, Tcwi_col, Tcwo_col, Tout_col,save_dir)

        print(f'BQ Model - MSE: {bq_mse:.4f}, R2: {bq_r2:.4f}')
        print(f'MP Model - MSE: {mp_mse:.4f}, R2: {mp_r2:.4f}')

        # Remove zero values from Qe and COP columns in the data
        filtered_data = data[(data[Qe_col] != 0) & (data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP'] > 0)].copy()

        # 创建第一张图 - BQ模型预测
        plt.figure(figsize=(12, 5))
        plt.plot(filtered_data.index, 1 / bq_pred, label='BQ Model Prediction', color='blue')
        plt.plot(filtered_data.index, filtered_data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP'], label='Actual', color='orange', linestyle='dashed')
        plt.title(f'{chiller} Chiller: Actual vs. BQ Model Prediction')
        plt.xlabel('Index')
        plt.ylabel('COP')
        plt.legend()
        plt.show()

        # 创建第二张图 - MP模型预测
        plt.figure(figsize=(12, 5))
        plt.plot(filtered_data.index, mp_pred, label='MP Model Prediction (Inverted)', color='red')
        plt.plot(filtered_data.index, filtered_data['AHUCOP' if Qe_col == 'ACdemandKJPH' else 'SCSHCOP'], label='Actual', color='orange', linestyle='dashed')
        plt.title(f'{chiller} Chiller: Actual vs. MP Model Prediction')
        plt.xlabel('Index')
        plt.ylabel('COP')
        plt.legend()
        plt.show()
