import datetime
import sys

from MPC_model import run_mpc_simulation
from RL_model import RLAgent

if __name__ == "__main__":
    # --- 1. 为HPC环境硬编码配置 (Hardcoded Configuration for HPC) ---
    #
    # 移除了所有交互式 input() 提示，直接定义运行参数。
    # 这使得脚本可以在非交互式环境中（如HPC作业调度系统）自动运行。
    #
    args = {
        'mode': 'train',                  # 运行模式设置为 'train'
        'sim_days': 365,                  # 仿真天数设置为 365
        'load_path': None,                # 设置为 None，表示从头开始训练，不加载现有模型
                                          # 如需继续训练，可改为 './rl_models_save/latest'
        'model_dir': './rl_models_save'   # 模型保存的目录
    }

    # 打印配置信息到日志，方便在HPC的输出文件中查看
    print("\n--- HPC Configuration ---")
    print(f"  Run Mode: {args['mode']}")
    print(f"  Load Model: {args['load_path'] or 'Not loading (starting new training)'}")
    print(f"  Simulation Days: {args['sim_days']}")
    print("--------------------")
    print("Starting the simulation...")

    # --- 2. 基于配置，初始化并运行程序 (The rest of the logic remains the same) ---
    start_sim_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
    battery_capacity_Wh = 3 * 30.0 * 1000
    initial_battery_Wh = 0

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

    # 如果指定了加载路径，则加载模型 (此逻辑保持不变)
    # 在当前配置下 (load_path=None), 这段代码将被跳过
    if args['load_path']:
        print(f"\nLoading model from '{args['load_path']}'...")
        if not rl_agent.load_model(args['load_path']) and args['mode'] == 'eval':
            print("Model loading failed in eval mode, terminating.")
            sys.exit(1)

    # 运行模拟
    run_mpc_simulation(
        start_time=start_sim_time,
        simulation_duration_days=args['sim_days'],
        battery_capacity=battery_capacity_Wh,
        initial_battery_energy=initial_battery_Wh,
        PV_max=129942.0851 * 0.2278,
        demand_max=140620.9898 * 0.2278,
        rl_agent=rl_agent,
        mode=args['mode'],
        model_dir=args['model_dir'],
        actual_data_file="measure.csv",
        rl_data_output_file=f"rl_{args['mode']}_data.csv"
    )

    print("\n--- Program execution finished ---")