import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 配置 Matplotlib 以支持中文显示和负号 ---
plt.rcParams["font.sans-serif"] = [
    "SimHei"
]  # 用来正常显示中文标签，如果没有 SimHei 可尝试 'Microsoft YaHei'
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# --- 定义偏振状态数据和辅助函数 ---

# 生成一个周期内的时间步 (相位 t 从 0 到 2*pi)
t = np.linspace(0, 2 * np.pi, 500)


def get_E_field(Ax, Ay, delta_rad):
    """根据振幅和相位差计算Ex和Ey"""
    Ex = Ax * np.cos(t)
    # Ey滞后Ex delta相位. 如果delta > 0, Ey超前，逆时针(左旋，物理定义)
    Ey = Ay * np.cos(t + delta_rad)
    return Ex, Ey


# 定义六种状态的参数
# 注意：物理学惯例中，delta > 0通常对应左旋(CCW)，delta < 0对应右旋(CW)
states = [
    # (a) 水平线偏振
    {
        "title": "(a) 水平线偏振光",
        "Ax": 1.0,
        "Ay": 0.0,
        "delta": 0,
        "delta_str": r"0",
        "rot": None,
    },
    # (b) 45°线偏振
    {
        "title": "(b) 45°线偏振光",
        "Ax": 1.0,
        "Ay": 1.0,
        "delta": 0,
        "delta_str": r"0",
        "rot": None,
    },
    # (c) 左旋圆偏振 (LCP) - Ax=Ay, delta=pi/2 (逆时针 CCW)
    {
        "title": "(c) 左旋圆偏振光 (LCP)",
        "Ax": 1.0,
        "Ay": 1.0,
        "delta": np.pi / 2,
        "delta_str": r"+\pi/2",
        "rot": "ccw",
    },
    # (d) 右旋圆偏振 (RCP) - Ax=Ay, delta=-pi/2 (顺时针 CW)
    {
        "title": "(d) 右旋圆偏振光 (RCP)",
        "Ax": 1.0,
        "Ay": 1.0,
        "delta": -np.pi / 2,
        "delta_str": r"-\pi/2",
        "rot": "cw",
    },
    # (e) 左旋椭圆偏振 (LEP) - Ax!=Ay, delta在(0, pi)之间
    {
        "title": "(e) 左旋椭圆偏振光 (LEP)",
        "Ax": 1.2,
        "Ay": 0.6,
        "delta": np.pi / 3,
        "delta_str": r"+\pi/3",
        "rot": "ccw",
    },
    # (f) 右旋椭圆偏振 (REP) - Ax!=Ay, delta在(-pi, 0)之间
    {
        "title": "(f) 右旋椭圆偏振光 (REP)",
        "Ax": 1.2,
        "Ay": 0.6,
        "delta": -np.pi / 3,
        "delta_str": r"-\pi/3",
        "rot": "cw",
    },
]

# --- 绘图主逻辑 ---

# 创建 3x2 的子图网格，设置整体画布大小
fig, axes = plt.subplots(3, 2, figsize=(10, 14))
fig.suptitle(
    "典型电场偏振态矢量轨迹示意图\n(观察方向：迎着光传播方向)", fontsize=16, y=0.99
)

axes_flat = axes.flatten()  # 将二维数组展平以便迭代

for i, ax in enumerate(axes_flat):
    state = states[i]
    Ex, Ey = get_E_field(state["Ax"], state["Ay"], state["delta"])

    # 1. 绘制矢量轨迹图
    # 使用淡蓝色填充内部，深蓝色绘制边界线条
    ax.plot(Ex, Ey, color="#0055A4", linewidth=2, label="E 轨迹")
    ax.fill(Ex, Ey, color="#0055A4", alpha=0.1)

    # -------- 美化坐标轴 --------
    # 移除顶部和右侧的脊柱(Spines)
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    # 将底部和左侧的脊柱移动到数据中心(0,0)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    # 设置坐标轴样式
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")

    # 移除刻度数字，只保留坐标轴线
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加坐标轴末端的箭头和标签
    ax.annotate(
        "x",
        xy=(1, 0),
        xycoords=("axes fraction", "data"),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        "y",
        xy=(0, 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # 设置固定的显示范围，确保视图一致
    limit = 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    # 强制设置纵横比相等，保证圆形不会变成椭圆
    ax.set_aspect("equal", adjustable="box")
    # -------------------------

    # 2. 绘制一个瞬时电场矢量箭头 (例如在 t = pi/4 时，或者对线性偏振选择峰值)
    t_idx = int(len(t) / 8)  # default pi/4
    if i == 0:
        t_idx = 0  # HLP 在 t=0最大
    elif i == 1:
        t_idx = 0  # 45LP 在 t=0最大

    E_arrow = patches.FancyArrow(
        0,
        0,
        Ex[t_idx],
        Ey[t_idx],
        width=0.04,
        head_width=0.15,
        head_length=0.2,
        color="#D62728",
        zorder=5,
    )  # 红色箭头
    ax.add_patch(E_arrow)
    # 在旁边标注 E 矢量
    # ax.text(Ex[t_idx]*1.1, Ey[t_idx]*1.1, r'$\vec{E}$', color='#D62728', fontsize=14)

    # 3. 标注旋转方向如果需要)
    rot_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=rot_style, color="k")

    # 设置旋转指示箭头的位置和曲率
    rad = 1.0  # 半径
    if state["rot"] == "ccw":
        # 左旋 (逆时针 CCW): 从第一象限指向第二象限
        rot_arrow = patches.FancyArrowPatch(
            (rad * np.cos(np.pi / 6), rad * np.sin(np.pi / 6)),
            (rad * np.cos(np.pi / 3 + 0.2), rad * np.sin(np.pi / 3 + 0.2)),
            connectionstyle="arc3,rad=0.3",
            **kw,
        )
        ax.add_patch(rot_arrow)
        ax.text(-limit * 0.9, limit * 0.8, "旋转: 左旋(逆时针)", fontsize=10)
    elif state["rot"] == "cw":
        # 右旋 (顺时针 CW): 从第二象限指向第一象限
        rot_arrow = patches.FancyArrowPatch(
            (rad * np.cos(np.pi / 3 + 0.2), rad * np.sin(np.pi / 3 + 0.2)),
            (rad * np.cos(np.pi / 6), rad * np.sin(np.pi / 6)),
            connectionstyle="arc3,rad=-0.3",
            **kw,
        )
        ax.add_patch(rot_arrow)
        ax.text(-limit * 0.9, limit * 0.8, "旋转: 右旋(顺时针)", fontsize=10)
    else:
        if i < 2:
            ax.text(-limit * 0.9, limit * 0.8, "旋转: 无", fontsize=10)

    # 4. 添加标题和参数文本框
    ax.set_title(state["title"], fontsize=13, pad=15)

    # 构造参数文本字符串 (使用 LaTeX 数学公式)
    param_text = (
        f"$A_x = {state['Ax']:.1f}$\n"
        f"$A_y = {state['Ay']:.1f}$\n"
        f"$\delta = {state['delta_str']}$"
    )

    # 在右下角放置文本框
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        limit * 0.95,
        -limit * 0.95,
        param_text,
        transform=ax.transData,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

plt.tight_layout(rect=[0, 0, 1, 0.97])  # 调整布局以适应总标题

# 显示图像
# plt.show()

# 如果需要保存高质量图片，请取消下面一行的注释
plt.savefig("polarization_states_diagram_zh.png", dpi=300, bbox_inches="tight")

print("图像生成完毕。已保存为 'polarization_states_diagram_zh.png'")
