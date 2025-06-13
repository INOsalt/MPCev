import traceback

import param_manage as pm
import input_manage as inp

# 导入您的初始化模块
from initialization.BiLSTM.RTPV_data import RTPV_TRAIN
from initialization.COP.COP_correction import COP_FITTING
from initialization.RCmodel.gp_correction_nostar import GP_correction
from initialization.RCmodel.wallRC_Tin_nostar import run_RC_analysis


def run_initializations(run_pv: bool = False, run_cop: bool = False, run_rc: bool = False):
    """
    执行选定的建筑模型初始化过程。

    参数:
        run_pv (bool): 如果为 True，则运行光伏 (PV) 模型生成。默认为 False。
        run_cop (bool): 如果为 True，则运行性能系数 (COP) 估计。默认为 False。
        run_rc (bool): 如果为 True，则运行 RC 模型分析和 GP 校正。默认为 False。
    """

    save_dir = pm.model_dir
    input_path = inp.input_file_path  # 尽管定义了 input_path，但在此处未直接使用，而是使用了 input_df。

    # --- 1. PV 模型生成 ---
    if run_pv:
        print("\n--- 正在运行光伏模型生成 ---")
        try:
            RTPV_TRAIN(save_dir)
            print("光伏模型生成成功完成。")
        except Exception as e:
            print(f"光伏模型生成过程中出错: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------

    # --- 2. COP 估计 ---
    if run_cop:
        print("\n--- 正在运行 COP 估计 ---")
        try:
            merged_data = inp.input_df
            COP_FITTING(merged_data, save_dir)
            print("COP 估计成功完成。")
        except Exception as e:
            print(f"COP 估计过程中出错: {e}")
            # --- 新增这部分来打印完整的Traceback ---
            print("--- 以下是完整的错误追溯信息 ---")
            traceback.print_exc()
            print("---------------------------------")
            # -----------------------------------------

    # --- 3. RC 模型分析和 GP 校正 ---
    if run_rc:
        print("\n--- 正在运行 RC 模型分析和 GP 校正 ---")
        wall_params_map = {
            'top': pm.wall_params_top,
            'middle': pm.wall_params_middle,
            'bottom': pm.wall_params_bottom
        }
        for floor_type in ['top', 'middle', 'bottom']:
            print(f"\n  -- 正在处理 {floor_type.capitalize()} 楼层的 RC 模型 --")
            try:
                run_RC_analysis(floor_type)  # 调用之前定义的 RC 分析函数
                selected_wall_params = wall_params_map.get(floor_type)
                if selected_wall_params is None:
                    print(f"    警告: 未找到 {floor_type} 楼层的墙体参数。跳过 GP 校正。")
                else:
                    GP_correction(floor_type, selected_wall_params, save_dir)
                print(f"  {floor_type.capitalize()} 楼层的 RC 模型分析和 GP 校正完成。")
            except Exception as e:
                print(f"  {floor_type.capitalize()} 楼层的 RC 模型分析过程中出错: {e}")
                # --- 新增这部分来打印完整的Traceback ---
                print("--- 以下是完整的错误追溯信息 ---")
                traceback.print_exc()
                print("---------------------------------")
                # -----------------------------------------
        print("\nRC 模型分析和 GP 校正完成。")

    print("\n初始化过程结束。")


# --- 如何使用该函数 ---
if __name__ == "__main__":
    print("正在启动初始化脚本...")

    # 示例 1: 运行所有初始化
    # print("\n--- 正在运行所有初始化 ---")
    # run_initializations(run_pv=True, run_cop=True, run_rc=True)

    # 示例 2: 仅运行 COP 估计和 RC 模型分析
    print("\n--- 正在运行 COP 估计和 RC 模型分析 ---")
    run_initializations(run_cop=True, run_rc=True)

    # 示例 3: 仅运行 PV 模型生成
    # print("\n--- 仅运行光伏模型生成 ---")
    # run_initializations(run_pv=True)

    # 示例 4: 不运行任何初始化 (演示)
    # print("\n--- 不运行任何初始化 (演示) ---")
    # run_initializations()