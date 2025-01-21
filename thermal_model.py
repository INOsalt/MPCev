import numpy as np


class ThermalModel:
    def __init__(self, params, gp_model):
        """
        初始化 ThermalModel 类

        参数:
        - params: 预训练的 2R2C 模型参数 [R_ext_wall, R_zone_wall, C_wall, C_zone]
        - gp_model: 高斯过程模型，用于室温预测误差校正
        """
        self.params = params
        self.gp_model = gp_model
        self.c_air = 1005  # 空气比热容 (J/kg·K)
        self.dt = None  # 时间步长将在后续设置

    def predict(self, Tamb_t, Tin_t, Tsp_t1, Qin_t, step_pre, vent_flow):
        """
        预测下一时刻的热负荷和温度

        参数:
        - Tamb_t: 当前环境温度 (°C)
        - Tin_t: 当前室内温度 (°C)
        - Qin_t: 当前内部热负荷 (W)
        - step_pre: 当前预测的时间步长 (小时)
        - vent_flow: 通风质量流量 (kg/s)

        返回:
        - Tin_t1: 下一时刻室内温度 (校正后的) (°C)
        - Twall_t1: 下一时刻墙体温度 (°C)
        - Q_zone: 室内热平衡负荷 (W)
        - Q_ahu: AHU 负荷 (W)
        - Q_space_heat: 空间加热负荷 (W)
        - Q_space_cool: 空间制冷负荷 (W)
        """
        self.dt = step_pre  # 设置时间步长
        R_ext_wall, R_zone_wall, C_wall, C_zone = self.params

        # **计算墙体温度**
        Twall_t = self._compute_Twall(Tamb_t, Tin_t, R_ext_wall, R_zone_wall)
        Tin_t1 = Tsp_t1
        dTin = Tin_t1 - Tin_t
        # **计算下一时刻室内温度**
        Q_zone = dTin * C_zone

        # **计算墙体热流**
        Q_ext_wall = (Tamb_t - Twall_t) / R_ext_wall  # 外墙传热
        Q_zone_wall = (Twall_t - Tin_t) / R_zone_wall  # 墙体向室内传热

        # **计算通风热流 (AHU 负荷)**
        Tsp_vent = self._compute_Tsp_vent(Tin_t)
        T_vent = Tsp_vent + 0.5
        temp_diff = T_vent - Tin_t
        Q_ahu = vent_flow * self.c_air * temp_diff

        # **热平衡
        Q_space = Q_zone - Q_ahu - Q_zone_wall - Qin_t

        # **计算空间加热和制冷负荷**
        Q_space_cool = 0
        Q_space_heat = 0
        if Q_space > 0:  # 加热
            Q_space_heat = Q_space
        elif Q_space < 0:  # 制冷
            Q_space_cool = Q_space

        # **计算下一时刻墙体温度**
        dTwall = (Q_ext_wall - Q_zone_wall) / C_wall
        Twall_t1 = Twall_t + dTwall * self.dt


        # **高斯过程校正室内温度**
        if self.gp_model:
            X = np.array([[Tamb_t, Tin_t, Qin_t]])  # 特征向量
            correction = self.gp_model.predict(X)  # 校正值
            Tin_t1 += correction

        return Tin_t1, Twall_t1, Q_zone, Q_ahu, Q_space_heat, Q_space_cool

    def _compute_Tsp_vent(self, Tin_t):
        """
        根据规则计算 Tsp_vent

        参数:
        - Tin_t: 当前室内温度 (°C)

        返回:
        - Tsp_vent: 通风供气温度设定点 (°C)
        """
        if Tin_t <= 21:
            return 21.0
        elif Tin_t >= 24:
            return 17.0
        else:
            return -1.333 * Tin_t + 49 - 0.5

    def _compute_Twall(self, Tamb_t, Tin_t, R_ext_wall, R_zone_wall):
        """
        计算墙体温度 Twall

        参数:
        - Tamb_t: 环境温度 (°C)
        - Tin_t: 室内温度 (°C)
        - R_ext_wall: 墙体外侧热阻
        - R_zone_wall: 墙体内侧热阻

        返回:
        - Twall: 墙体温度 (°C)
        """
        return (Tamb_t / R_ext_wall + Tin_t / R_zone_wall) / (1 / R_ext_wall + 1 / R_zone_wall)


# 主函数，仅在直接运行脚本时执行
if __name__ == "__main__":
    from joblib import load

    # 模拟加载高斯过程模型
    try:
        gp_model = load("gp_model.pkl")
        print("高斯过程模型已加载。")
    except FileNotFoundError:
        gp_model = None
        print("未找到高斯过程模型，继续使用未校正模型。")

    # 预训练的 2R2C 模型参数
    params = [6176805.99725779, 9987082.48439311, 3452420.56892008, 1671112.44674421]

    # 初始化 ThermalModel 类
    thermal_model = ThermalModel(params=params, gp_model=gp_model)

    # 输入参数
    Tamb_t = 15  # 当前环境温度
    Tin_t = 22  # 当前室内温度
    Qin_t = 100  # 内部热负荷
    vent_flow = 0.2  # 通风质量流量
    step_pre = 0.5  # 时间步长

    # 执行预测
    Tin_t1, Twall_t1, Q_zone, Q_ahu, Q_space_heat, Q_space_cool = thermal_model.predict(
        Tamb_t=Tamb_t,
        Tin_t=Tin_t,
        Qin_t=Qin_t,
        step_pre=step_pre,
        vent_flow=vent_flow
    )

    # 输出预测结果
    print(f"下一时刻室温 Tin_t+1: {Tin_t1:.2f}°C")
    print(f"下一时刻墙体温度 Twall_t+1: {Twall_t1:.2f}°C")
    print(f"室内热平衡负荷 Q_zone: {Q_zone:.2f} W")
    print(f"AHU 负荷 Q_ahu: {Q_ahu:.2f} W")
    print(f"空间加热负荷 Q_space_heat: {Q_space_heat:.2f} W")
    print(f"空间制冷负荷 Q_space_cool: {Q_space_cool:.2f} W")
