import warnings
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from matplotlib import font_manager
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# sklearn 版本兼容补丁
# ============================================================
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


# ============================================================
# 基础路径
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
GUI_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "gui"


CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Serif CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]

try:
    font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
preferred_fonts = [font for font in CHINESE_FONT_CANDIDATES if font in available_fonts]
dynamic_fonts = sorted(
    font
    for font in available_fonts
    if any(token in font for token in ["Noto Sans CJK", "Noto Serif CJK", "Source Han", "WenQuanYi"])
)
chinese_fonts = list(dict.fromkeys(preferred_fonts + dynamic_fonts))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = (chinese_fonts or CHINESE_FONT_CANDIDATES) + ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 改为 CPU 版 model / preprocessor
MODEL_PATH = GUI_ARTIFACTS_DIR / "TabPFN_CPU_ST_model.joblib"
PREPROCESSOR_PATH = GUI_ARTIFACTS_DIR / "TabPFN_CPU_ST_preprocessor.joblib"

# 优先寻找 CPU 版 explainer；若没有，再尝试旧的 GPU 版 explainer
EXPLAINER_PATH_CPU = GUI_ARTIFACTS_DIR / "shap_explainer_TabPFN_CPU_ST.joblib"
EXPLAINER_PATH_GPU = GUI_ARTIFACTS_DIR / "shap_explainer_TabPFN_GPU_ST.joblib"


# ============================================================
# 原始输入特征（14个）
# ============================================================
RAW_FEATURES = [
    "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
    "Ag2.36", "Ag4.75", "Ag9.5", "FT", "FC", "FL", "TS"
]

INPUT_LABELS = {
    "Pe": "Pe (针入度, 0.1 mm)",
    "Du": "Du (延度, cm)",
    "SP": "SP (软化点, °C)",
    "AC": "AC (沥青用量, wt.%)",
    "AV": "AV (空隙率, %)",
    "VMA": "VMA (矿料间隙率, %)",
    "VFA": "VFA (沥青饱和度, %)",
    "Ag2.36": "Ag2.36 (2.36 mm 通过率, %)",
    "Ag4.75": "Ag4.75 (4.75 mm 通过率, %)",
    "Ag9.5": "Ag9.5 (9.5 mm 通过率, %)",
    "FT": "FT (纤维类型)",
    "FC": "FC (纤维掺量, wt.%)",
    "FL": "FL (纤维长度, mm)",
    "TS": "TS (纤维抗拉强度, MPa)",
}

DEFAULT_INPUT = {
    "Pe": 91.8,
    "Du": 150.0,
    "SP": 46.9,
    "AC": 5.02,
    "AV": 4.22,
    "VMA": 16.0,
    "VFA": 73.8,
    "Ag2.36": 37.0,
    "Ag4.75": 53.0,
    "Ag9.5": 76.5,
    "FT": "Basalt fiber",
    "FC": 0.25,
    "FL": 6.0,
    "TS": 2320.0,
}

RANGES = {
    "Pe": (0.0, 200.0, 0.1),
    "Du": (0.0, 300.0, 1.0),
    "SP": (0.0, 100.0, 0.1),
    "AC": (0.0, 20.0, 0.1),
    "AV": (0.0, 30.0, 0.01),
    "VMA": (0.0, 80.0, 0.01),
    "VFA": (0.0, 100.0, 0.01),
    "Ag2.36": (0.0, 100.0, 0.01),
    "Ag4.75": (0.0, 100.0, 0.01),
    "Ag9.5": (0.0, 100.0, 0.01),
    "FC": (0.0, 10.0, 0.01),
    "FL": (0.0, 100.0, 0.01),
    "TS": (0.0, 10000.0, 1.0),
}

FALLBACK_FT_OPTIONS = [
    "no_fiber",
    "basalt fiber",
    "glass fiber",
    "polyester fiber",
    "steel fiber",
]

FT_DISPLAY_LABELS = {
    "no_fiber": "无纤维 (no_fiber)",
    "basalt fiber": "玄武岩纤维 (basalt fiber)",
    "glass fiber": "玻璃纤维 (glass fiber)",
    "polyester fiber": "聚酯纤维 (polyester fiber)",
    "steel fiber": "钢纤维 (steel fiber)",
    "No_fiber": "无纤维 (No_fiber)",
    "Basalt fiber": "玄武岩纤维 (Basalt fiber)",
    "Glass fiber": "玻璃纤维 (Glass fiber)",
    "Polyester Fiber": "聚酯纤维 (Polyester Fiber)",
    "Steel fiber": "钢纤维 (Steel fiber)",
}


def format_ft_option(option):
    return FT_DISPLAY_LABELS.get(str(option), str(option))


# ============================================================
# 页面设置
# ============================================================
st.set_page_config(page_title="AC 劈裂强度 ST 预测 GUI", layout="wide")
st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", sans-serif;
    }

    /* 数值输入框里的数字 */
    [data-testid="stNumberInput"] input {
        font-size: 24px !important;
        font-weight: normal !important;
    }

    /* 下拉框当前选中的文字 */
    div[data-baseweb="select"] > div {
        font-size: 20px !important;
        font-weight: normal !important;
    }

    /* 各输入组件上方的标签文字，如 Pe、Du、FT */
    label[data-testid="stWidgetLabel"] p {
        font-size: 26px !important;
        font-weight: normal !important;
    }

    /* number_input 右侧 ± 按钮 */
    [data-testid="stNumberInput"] button {
        font-size: 22px !important;
        font-weight: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='background-color:#0B5ED7;padding:8px;border-radius:10px;text-align:center;'>
        <h2 style='color:white;margin:0;'>沥青混凝土劈裂强度 (ST) 预测与 SHAP 分析</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 资源加载
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    info = {
        "model": None,
        "preprocessor": None,
        "explainer": None,
        "errors": {},
        "explainer_source": None,
    }

    try:
        info["model"] = joblib.load(MODEL_PATH)
    except Exception:
        info["errors"]["model"] = traceback.format_exc()

    try:
        info["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    except Exception:
        info["errors"]["preprocessor"] = traceback.format_exc()

    # 1) 优先加载 CPU explainer
    info["explainer_cpu_path"] = str(EXPLAINER_PATH_CPU)
    info["explainer_cpu_exists"] = EXPLAINER_PATH_CPU.exists()
    if EXPLAINER_PATH_CPU.exists():
        try:
            info["explainer"] = joblib.load(EXPLAINER_PATH_CPU)
            info["explainer_source"] = "cpu_joblib"
        except Exception:
            info["errors"]["explainer_cpu"] = traceback.format_exc()
    else:
        info["errors"]["explainer_cpu_missing"] = f"未找到 CPU explainer 文件: {EXPLAINER_PATH_CPU}"

    # 2) 若 CPU explainer 不可用，再尝试旧 GPU explainer
    info["explainer_gpu_path"] = str(EXPLAINER_PATH_GPU)
    info["explainer_gpu_exists"] = EXPLAINER_PATH_GPU.exists()
    if info["explainer"] is None and EXPLAINER_PATH_GPU.exists():
        try:
            info["explainer"] = joblib.load(EXPLAINER_PATH_GPU)
            info["explainer_source"] = "gpu_joblib"
        except Exception:
            info["errors"]["explainer_gpu"] = traceback.format_exc()
    elif info["explainer"] is None and not EXPLAINER_PATH_GPU.exists():
        info["errors"]["explainer_gpu_missing"] = f"未找到 GPU explainer 文件: {EXPLAINER_PATH_GPU}"

    return info


artifacts = load_artifacts()


# ============================================================
# 工具函数
# ============================================================
def get_feature_names_from_preprocessor(preprocessor):
    if preprocessor is None:
        return None
    try:
        names = preprocessor.get_feature_names_out()
        names = [str(x) for x in names]
        cleaned = []
        for n in names:
            n2 = n.replace("num__", "").replace("cat__", "").replace("ft__", "")
            cleaned.append(n2)
        return cleaned
    except Exception:
        return None


def get_ft_options_from_preprocessor(preprocessor):
    if preprocessor is None:
        return FALLBACK_FT_OPTIONS

    try:
        if hasattr(preprocessor, "transformers_"):
            for name, transformer, cols in preprocessor.transformers_:
                if transformer == "drop":
                    continue
                if isinstance(cols, (list, tuple)) and "FT" in list(cols):
                    if hasattr(transformer, "categories_") and len(transformer.categories_) > 0:
                        return [str(x) for x in transformer.categories_[0].tolist()]
                    if hasattr(transformer, "named_steps"):
                        for _, step in transformer.named_steps.items():
                            if hasattr(step, "categories_") and len(step.categories_) > 0:
                                return [str(x) for x in step.categories_[0].tolist()]
    except Exception:
        pass

    return FALLBACK_FT_OPTIONS


def normalize_raw_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    numeric_cols = [
        "Pe", "Du", "SP", "AC", "AV", "VMA", "VFA",
        "Ag2.36", "Ag4.75", "Ag9.5", "FC", "FL", "TS"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["FT"] = df["FT"].astype(str).str.strip().str.lower()
    return df


def transform_input(preprocessor, raw_df):
    if preprocessor is None:
        raise RuntimeError("未能加载预处理器，无法进行模型输入转换。")

    raw_df = normalize_raw_input(raw_df)

    X = preprocessor.transform(raw_df)
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    if feature_names is not None and len(feature_names) == X.shape[1]:
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    return X_df


def safe_predict(model, X_df):
    if model is None:
        raise RuntimeError("未能加载模型，无法进行预测。")
    pred = model.predict(X_df)
    if isinstance(pred, (list, tuple, np.ndarray)):
        return float(np.asarray(pred).reshape(-1)[0])
    return float(pred)


def build_fallback_background(preprocessor):
    """
    当没有可直接加载的 explainer 时，使用默认输入构造一个 CPU 可用的 background，
    以便临时生成本地 SHAP 解释。
    """
    default_raw = pd.DataFrame([DEFAULT_INPUT.copy()])
    default_raw = normalize_raw_input(default_raw)
    bg = transform_input(preprocessor, default_raw)

    # 用少量扰动复制多行，避免背景只有 1 行时解释退化过于明显
    rows = []
    base = bg.iloc[0].values.astype(float)
    for scale in [0.0, 0.01, -0.01, 0.02, -0.02]:
        rows.append(base * (1.0 + scale))
    bg_arr = np.vstack(rows)
    return pd.DataFrame(bg_arr, columns=bg.columns)


@st.cache_resource(show_spinner=False)
def get_runtime_fallback_explainer(_model, _preprocessor):
    background_df = build_fallback_background(_preprocessor)
    try:
        explainer = shap.SamplingExplainer(_model.predict, background_df)
        return explainer, "runtime_sampling"
    except Exception:
        explainer = shap.Explainer(_model.predict, background_df)
        return explainer, "runtime_generic"


def make_local_shap_explanation(explainer, X_df):
    if explainer is None:
        raise RuntimeError("未能加载 SHAP explainer，无法生成单样本 SHAP 解释。")
    return explainer(X_df)


def plot_waterfall_from_explanation(sample_exp, max_display=12):
    plt.close("all")
    plt.figure(figsize=(4.6, 3.2))
    shap.plots.waterfall(sample_exp, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_force_from_explanation(sample_exp, top_n=8):
    """
    自定义 force-style 图：
    1. 使用真实特征名，而不是 Feature 1/2/3 这类占位符；
    2. 控制画布比例，避免 shap.force_plot(matplotlib=True) 在 Streamlit 中变形；
    3. 仅展示绝对贡献最大的 top_n 个特征，提升可读性。
    """
    plt.close("all")

    values = np.asarray(sample_exp.values).reshape(-1)
    feature_names = list(sample_exp.feature_names) if sample_exp.feature_names is not None else [f"f{i}" for i in range(len(values))]
    feature_data = np.asarray(sample_exp.data).reshape(-1) if getattr(sample_exp, "data", None) is not None else np.array([np.nan] * len(values))
    base_value = float(np.asarray(sample_exp.base_values).reshape(-1)[0])
    pred_value = base_value + float(values.sum())

    top_idx = np.argsort(np.abs(values))[-min(top_n, len(values)):]
    top_idx = top_idx[np.argsort(np.abs(values[top_idx]))[::-1]]

    top_vals = values[top_idx]
    top_names = [feature_names[i] for i in top_idx]
    top_data = feature_data[top_idx]

    labels = []
    for n, d in zip(top_names, top_data):
        if isinstance(d, (float, int, np.floating, np.integer)):
            labels.append(f"{n} = {d:.3g}")
        else:
            labels.append(f"{n} = {d}")

    fig, ax = plt.subplots(figsize=(8.0, 3.6))

    left_neg = base_value
    left_pos = base_value
    y_neg = 0.25
    y_pos = -0.25
    bar_h = 0.32

    neg_items, pos_items = [], []
    for lab, val in zip(labels, top_vals):
        if val < 0:
            neg_items.append((lab, val))
        else:
            pos_items.append((lab, val))

    for lab, val in neg_items:
        width = abs(val)
        start = left_neg - width
        ax.barh(y_neg, width, left=start, height=bar_h, color="#1E88E5", edgecolor="white")
        ax.text(start + width / 2, y_neg, f"{val:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(start + width / 2, y_neg - 0.32, lab, ha="center", va="top", color="#1E88E5", fontsize=10)
        left_neg = start

    for lab, val in pos_items:
        width = abs(val)
        start = left_pos
        ax.barh(y_pos, width, left=start, height=bar_h, color="#FF0051", edgecolor="white")
        ax.text(start + width / 2, y_pos, f"+{val:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(start + width / 2, y_pos + 0.32, lab, ha="center", va="bottom", color="#FF0051", fontsize=10)
        left_pos = start + width

    ax.axvline(base_value, color="#888888", linestyle="--", linewidth=1.2)
    ax.axvline(pred_value, color="#222222", linestyle="-", linewidth=1.5)

    xmin = min(left_neg, base_value, pred_value) - 0.05
    xmax = max(left_pos, base_value, pred_value) + 0.05
    if xmin == xmax:
        xmin -= 0.1
        xmax += 0.1
    ax.set_xlim(xmin, xmax)

    ax.text(base_value, 0.72, f"基准值 = {base_value:.3f}", ha="center", va="bottom", fontsize=11, color="#666666")
    ax.text(pred_value, 0.86, f"预测值 = {pred_value:.3f}", ha="center", va="bottom", fontsize=12, color="#111111", fontweight="bold")

    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("模型输出")
    ax.set_title("Force 风格特征贡献图", fontsize=12, fontweight="bold")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    plt.tight_layout()
    return fig


def build_raw_input_df(ft_options):
    st.markdown(
        """
        <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
            <h3 style='color:white;margin:0;'>输入参数</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_ft = str(DEFAULT_INPUT["FT"]).strip()
    default_idx = 0
    for i, opt in enumerate(ft_options):
        if str(opt).strip() == default_ft or str(opt).strip().lower() == default_ft.lower():
            default_idx = i
            break

    values = {}
    for row_start in range(0, len(RAW_FEATURES), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, RAW_FEATURES[row_start:row_start + 3], strict=False):
            with col:
                if feature == "FT":
                    values[feature] = st.selectbox(
                        INPUT_LABELS[feature],
                        options=ft_options,
                        index=default_idx,
                        format_func=format_ft_option,
                    )
                else:
                    values[feature] = st.number_input(
                        INPUT_LABELS[feature],
                        *RANGES[feature][:2],
                        value=DEFAULT_INPUT[feature],
                        step=RANGES[feature][2],
                    )

    raw_df = pd.DataFrame([{feature: values[feature] for feature in RAW_FEATURES}])
    return raw_df


# ============================================================
# 初始化 session_state
# ============================================================
if "local_shap_exp" not in st.session_state:
    st.session_state["local_shap_exp"] = None
if "local_shap_ok" not in st.session_state:
    st.session_state["local_shap_ok"] = None
if "local_shap_error" not in st.session_state:
    st.session_state["local_shap_error"] = ""
if "local_shap_source" not in st.session_state:
    st.session_state["local_shap_source"] = None
if "shap_in_progress" not in st.session_state:
    st.session_state["shap_in_progress"] = False


# ============================================================
# 加载状态提示
# ============================================================
if artifacts["errors"]:
    st.warning("部分本地模型文件未能加载。")

    with st.expander("显示加载详情"):
        st.write("### Artifact 路径检查")
        st.code(
            "\n".join([
                f"MODEL_PATH = {MODEL_PATH}",
                f"MODEL_EXISTS = {MODEL_PATH.exists()}",
                f"PREPROCESSOR_PATH = {PREPROCESSOR_PATH}",
                f"PREPROCESSOR_EXISTS = {PREPROCESSOR_PATH.exists()}",
                f"EXPLAINER_PATH_CPU = {artifacts.get('explainer_cpu_path')}",
                f"EXPLAINER_CPU_EXISTS = {artifacts.get('explainer_cpu_exists')}",
                f"EXPLAINER_PATH_GPU = {artifacts.get('explainer_gpu_path')}",
                f"EXPLAINER_GPU_EXISTS = {artifacts.get('explainer_gpu_exists')}",
            ])
        )
        for k, v in artifacts["errors"].items():
            st.write(f"### {k}")
            st.code(v)


# ============================================================
# 输入区
# ============================================================
ft_options = get_ft_options_from_preprocessor(artifacts["preprocessor"])
raw_input_df = build_raw_input_df(ft_options)


# ============================================================
# 预测按钮
# ============================================================
predict_clicked = st.button("预测 ST 并生成 SHAP 分析", use_container_width=True)

if predict_clicked:
    # 每次点击都先重置本轮 SHAP 状态
    st.session_state["local_shap_exp"] = None
    st.session_state["local_shap_ok"] = None
    st.session_state["local_shap_error"] = ""
    st.session_state["local_shap_source"] = None
    st.session_state["shap_in_progress"] = False

    try:
        X_input = transform_input(artifacts["preprocessor"], raw_input_df)
        y_pred = safe_predict(artifacts["model"], X_input)

        st.session_state["raw_input_df"] = raw_input_df.copy()
        st.session_state["X_input"] = X_input.copy()
        st.session_state["y_pred"] = y_pred
        st.session_state["shap_in_progress"] = True

        # 先刷新页面，让 ST 预测结果可见，再进入 SHAP 计算阶段。
        st.rerun()

    except Exception:
        st.session_state["shap_in_progress"] = False
        st.error("预测失败。详情如下。")
        st.code(traceback.format_exc())


# ============================================================
# 预测结果区
# ============================================================
st.markdown(
    """
    <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
        <h3 style='color:white;margin:0;'>预测结果</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

if "y_pred" in st.session_state:
    st.success("预测完成。")
    st.markdown(
        f"""
        <div style="background-color:#F3F3F3;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:28px;font-weight:800;color:#000000;line-height:1.4;">
                预测 ST = {st.session_state['y_pred']:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("显示转换后的模型输入"):
        st.dataframe(st.session_state["X_input"], use_container_width=True)
else:
    st.info("点击预测按钮生成 ST 结果。")


# ============================================================
# 单样本 SHAP 解释区
# ============================================================
st.markdown(
    """
    <div style='background-color:orange;padding:4px 10px;border-radius:6px;display:inline-block;margin-top:18px;margin-bottom:18px;'>
        <h3 style='color:white;margin:0;'>当前样本的 SHAP 分析</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("shap_in_progress", False):
    st.info("ST 结果已生成，正在为当前样本计算 SHAP 分析，请稍候。")

    try:
        current_explainer = artifacts["explainer"]
        current_explainer_source = artifacts.get("explainer_source")

        if current_explainer is None:
            current_explainer, current_explainer_source = get_runtime_fallback_explainer(
                artifacts["model"],
                artifacts["preprocessor"],
            )

        local_exp = make_local_shap_explanation(
            current_explainer,
            st.session_state["X_input"],
        )

        st.session_state["local_shap_exp"] = local_exp
        st.session_state["local_shap_ok"] = True
        st.session_state["local_shap_error"] = ""
        st.session_state["local_shap_source"] = current_explainer_source
        st.session_state["shap_in_progress"] = False

        st.rerun()

    except Exception:
        st.session_state["local_shap_exp"] = None
        st.session_state["local_shap_ok"] = False
        st.session_state["local_shap_error"] = traceback.format_exc()
        st.session_state["local_shap_source"] = None
        st.session_state["shap_in_progress"] = False

        st.rerun()

elif "local_shap_ok" in st.session_state and st.session_state["local_shap_ok"]:
    local_exp = st.session_state["local_shap_exp"]
    sample_exp = local_exp[0]

    # if st.session_state.get("local_shap_source"):
    #     st.caption(f"Explainer source: {st.session_state['local_shap_source']}")

    st.write("### Waterfall 图")
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        try:
            fig = plot_waterfall_from_explanation(sample_exp, max_display=12)
            st.pyplot(fig, clear_figure=True, use_container_width=False)
        except Exception:
            st.error("Waterfall 图绘制失败。")
            st.code(traceback.format_exc())

    st.write("### 特征贡献表")
    try:
        feature_names = list(sample_exp.feature_names)
        values = np.asarray(sample_exp.values).reshape(-1)
        feature_data = np.asarray(sample_exp.data).reshape(-1)

        contrib_df = pd.DataFrame({
            "特征": feature_names,
            "特征值": feature_data,
            "SHAP 值": values,
            "|SHAP|": np.abs(values),
        }).sort_values("|SHAP|", ascending=False)
        st.dataframe(contrib_df, use_container_width=True)
    except Exception:
        st.error("特征贡献表生成失败。")
        st.code(traceback.format_exc())

elif "local_shap_ok" in st.session_state and st.session_state["local_shap_ok"] is False:
    st.warning("当前样本预测已完成，但未能生成本地 SHAP 解释。")
    with st.expander("显示 SHAP 错误详情"):
        st.code(st.session_state.get("local_shap_error", "暂无详细信息。"))

else:
    st.info("完成预测后，如果 explainer 可以加载，系统会尝试生成本地 SHAP 图。")
