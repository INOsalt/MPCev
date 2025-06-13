import datetime
import os
import sys

from MPC_model import run_mpc_simulation
from RL_model import RLAgent
from initialization_model import run_initializations

# --- Helper functions for user input ---

def get_bool_input(prompt_text):
    """Gets a 'y' or 'n' input from the user and returns a boolean."""
    while True:
        response = input(f"{prompt_text} (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("输入无效，请输入 'y' 或 'n'。")

def get_float_input(prompt_text, default_value):
    """Gets a float input from the user, with a default value."""
    while True:
        try:
            response = input(f"{prompt_text} [默认: {default_value}]: ")
            if response == "":
                return default_value
            return float(response)
        except ValueError:
            print("输入无效，请输入一个数字。")

def get_datetime_input(prompt_text, default_dt):
    """Gets a datetime input from the user, with a default value."""
    print(prompt_text)
    while True:
        try:
            dt_str = input(f"请输入开始日期和时间 (YYYY-MM-DD HH) [默认: {default_dt.strftime('%Y-%m-%d %H')}]: ")
            if not dt_str:
                return default_dt
            # Pad with zeros for minute and second
            return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H")
        except ValueError:
            print("格式无效，请使用 'YYYY-MM-DD HH' 格式。")


if __name__ == "__main__":
    # --- Step 1: Initialization Settings ---
    print("--- 初始化设置 ---")
    run_pv_init = get_bool_input("是否需要运行 PV 初始化?")
    run_cop_init = get_bool_input("是否需要运行 COP 初始化?")
    run_rc_init = get_bool_input("是否需要运行 RC 初始化?")
    run_initializations(run_pv=run_pv_init, run_cop=run_cop_init, run_rc=run_rc_init)

    # --- Step 2: Simulation Core Parameters ---
    print("\n--- 仿真核心参数设置 ---")
    start_sim_time = get_datetime_input(
        "请输入仿真开始时间。",
        datetime.datetime(2023, 1, 1, 0, 0, 0)
    )
    battery_capacity_Wh = get_float_input("请输入电池总容量 (Wh)", 50000.0)
    initial_battery_Wh = get_float_input("请输入电池初始电量 (Wh)", 25000.0)
    pv_max_val = get_float_input("请输入 PV 最大功率", 129942.0851 * 0.2278)
    demand_max_val = get_float_input("请输入需求最大功率", 140620.9898 * 0.2278)
    actual_data_file_path = input("请输入实际数据文件名 [默认: measure.csv]: ") or "measure.csv"


    args = {}
    # --- Step 3: RL Agent Mode and Model Loading ---
    while True:
        print("\n--- 运行模式选择 ---")
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
            model_dir_default = './rl_models_save'
            print(f"  (模型默认根目录: {model_dir_default})")
            print("  1: 加载表现最好的模型 (best)")
            print("  2: 加载上次训练的模型 (latest)")
            print("  3: 输入自定义路径前缀")
            load_choice = input("请输入选项 (1, 2, 或 3): ")

            if load_choice == '1':
                args['load_path'] = os.path.join(model_dir_default, "best")
                break
            elif load_choice == '2':
                args['load_path'] = os.path.join(model_dir_default, "latest")
                break
            elif load_choice == '3':
                custom_path = input("请输入模型的路径前缀 (例如 './rl_models_save/ep_10'): ")
                args['load_path'] = custom_path
                break
            else:
                print("输入无效，请重新输入。")

    # --- Step 4: Simulation Duration and Final Confirmation ---
    while True:
        try:
            args['sim_days'] = int(input("\n请输入仿真的总天数 (例如 30): "))
            break
        except ValueError:
            print("输入无效，请输入一个整数。")

    args['model_dir'] = './rl_models_save'
    rl_output_file = f"rl_{args['mode']}_data.csv"

    print("\n--- 最终配置确认 ---")
    print(f"  仿真开始时间: {start_sim_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  仿真总天数: {args['sim_days']}")
    print(f"  电池容量: {battery_capacity_Wh} Wh")
    print(f"  电池初始电量: {initial_battery_Wh} Wh")
    print(f"  PV 最大功率: {pv_max_val:.2f}")
    print(f"  需求最大功率: {demand_max_val:.2f}")
    print(f"  实际数据文件: {actual_data_file_path}")
    print("-------------------------")
    print(f"  运行模式: {args['mode']}")
    print(f"  加载模型路径: {args['load_path'] or '不加载'}")
    print(f"  模型保存目录: {args['model_dir']}")
    print(f"  RL输出数据文件: {rl_output_file}")
    print("-------------------------")
    input("按 Enter 键开始运行...")

    # --- Model and Simulation Setup ---
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

    # --- Run Simulation ---
    run_mpc_simulation(
        start_time=start_sim_time,
        simulation_duration_days=args['sim_days'],
        battery_capacity=battery_capacity_Wh,
        initial_battery_energy=initial_battery_Wh,
        PV_max=pv_max_val,
        demand_max=demand_max_val,
        rl_agent=rl_agent,
        mode=args['mode'],
        model_dir=args['model_dir'],
        actual_data_file=actual_data_file_path,
        rl_data_output_file=rl_output_file
    )

    print("\n--- 程序执行完毕 ---")
    input("按 Enter 键退出。")