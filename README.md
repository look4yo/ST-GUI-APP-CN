# AC 劈裂强度 ST 预测中文 GUI

这是 `apps/gui/` 英文 Streamlit GUI 的中文副本。界面文案已汉化，模型加载、预处理、预测和 SHAP 解释逻辑保持不变。

## 启动命令

```powershell
streamlit run apps/gui_cn/ST_GUI_app_CPU.py
```

## 模型文件

中文 GUI 与英文 GUI 共用根目录下的 `artifacts/gui/`：

- `TabPFN_CPU_ST_model.joblib`
- `TabPFN_CPU_ST_preprocessor.joblib`
- `shap_explainer_TabPFN_CPU_ST.joblib`

这些文件默认不纳入 Git。

## 说明

- 输入特征名保留论文和数据表中的原始缩写，例如 `ST`, `SHAP`, `FT`, `Pe`, `Du`, `AC`。
- `FT` 下拉框显示中文说明，但传递给模型的类别值保持原始英文标签不变。
- Matplotlib 已配置中文字体候选，优先使用 `Microsoft YaHei`, `SimHei`, `SimSun`，并保留 `Noto Sans CJK` 系列作为跨平台候选。
