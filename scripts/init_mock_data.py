"""脚本：生成 50 条贴近真实场景的模拟工单，用于演示与检索测试。

用法：
    python scripts/init_mock_data.py            # 若已存在则拒绝覆盖
    python scripts/init_mock_data.py --force    # 覆盖
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import get_settings  # noqa: E402

MODELS = ["H1", "H1-2", "G1", "Go2", "B2", "R1"]
ENGINEERS = ["张工", "李工", "王工", "陈工", "刘工"]

FAULTS: dict[str, dict[str, str]] = {
    "关节异响": {
        "symptom": "{model} 在站立相向摆动相切换时，{joint}关节发出“咯噔”声，负载越大越明显，伴随约 0.5° 的位置偏差。",
        "diagnosis": "对比目标角度与编码器反馈，发现输出端存在回差；拆检确认减速器输出法兰螺栓预紧力不足，螺纹胶已失效。",
        "solution": "按扭矩规范重新紧固输出法兰螺栓，涂覆螺纹紧固胶，完成后重新执行关节零位标定，回差恢复正常。",
        "parts": "法兰螺栓、螺纹紧固胶",
    },
    "电机过热": {
        "symptom": "{model} 连续爬坡 10 分钟后，{joint}关节电机温度报警，外壳烫手，停机冷却后可恢复。",
        "diagnosis": "读取电机温度与电流曲线，发现持续大电流下散热不足；检查发现关节散热风道被灰尘堵塞，润滑脂黏度偏高。",
        "solution": "清理风道与散热片，更换指定型号高温润滑脂，降低连续高负载工况占空比，复测温度回落到阈值以下。",
        "parts": "高温润滑脂、密封圈",
    },
    "电池鼓包": {
        "symptom": "{model} 电池外观轻微变形，充电时充电器反复进入保护，电量显示跳变。",
        "diagnosis": "拆检可见电芯极耳处有轻微析锂痕迹，电池组绝缘电阻偏低；充电器输出正常，判定为电池本体异常。",
        "solution": "禁止继续充电并回收旧电池，整组更换原厂锂电池组；同步检查电池舱散热与充电接口接触状态。",
        "parts": "锂电池组",
    },
    "通信中断": {
        "symptom": "{model} 在 Wi-Fi 图传开启时，运动控制板与主控偶发失联，rt/lowstate 中断 2-3 秒。",
        "diagnosis": "抓包发现 2.4G 信道与图传信道重叠，丢包率 12%；检查网线水晶头存在松动。",
        "solution": "图传切换至 5.8G 信道，重新压接网线水晶头，连续测试 2 小时无中断。",
        "parts": "屏蔽网线、水晶头",
    },
    "IMU数据漂移": {
        "symptom": "{model} 静止站立时姿态角缓慢漂移，10 分钟累计约 3°，行走时出现轻微跑偏。",
        "diagnosis": "读取 IMU 原始数据，发现零偏随温度升高而增大；标定记录显示上次标定在高温环境下完成。",
        "solution": "在常温下重新执行 IMU 零偏标定，升级固件修复温漂补偿逻辑，复测 30 分钟漂移小于 0.5°。",
        "parts": "无",
    },
}

JOINTS = ["左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "腰部"]


def build_rows(count: int = 50) -> list[dict]:
    rows: list[dict] = []
    fault_names = list(FAULTS)
    for index in range(count):
        fault = fault_names[index % len(fault_names)]
        model = MODELS[index % len(MODELS)]
        joint = JOINTS[index % len(JOINTS)]
        template = FAULTS[fault]

        month = 1 + (index % 12)
        year = 2025 if index < 25 else 2026
        day = 1 + (index * 3) % 27
        order_id = f"WO-{year}{month:02d}-{index % 10 + 1:03d}"

        rows.append(
            {
                "工单ID": order_id,
                "产品型号": model,
                "故障现象": template["symptom"].format(model=model, joint=joint),
                "诊断过程": template["diagnosis"],
                "解决方案": template["solution"],
                "更换配件": template["parts"],
                "故障类型": fault,
                "发生日期": f"{year}-{month:02d}-{day:02d}",
                "处理工程师": ENGINEERS[index % len(ENGINEERS)],
                "处理时长(小时)": round(1.5 + (index % 8) * 0.5, 1),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="生成模拟工单")
    parser.add_argument("--force", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    settings = get_settings()
    path = settings.work_orders_file
    if path.exists() and not args.force:
        print(f"已存在：{path}\n如需覆盖请加 --force")
        return 1

    frame = pd.DataFrame(build_rows(50))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)
    print(f"已生成 {len(frame)} 条工单：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
