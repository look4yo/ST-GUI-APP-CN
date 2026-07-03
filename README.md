# AC 劈裂强度 ST 预测中文 GUI

这是面向 Streamlit Cloud 的中文 GUI 独立部署仓库。界面文案已汉化，模型加载、预处理、预测和 SHAP 解释逻辑与本地项目版本保持一致。

## 启动命令

```powershell
streamlit run ST_GUI_app_CPU.py
```

## Streamlit Cloud

在 Streamlit Cloud 中将 app 入口设置为：

```text
ST_GUI_app_CPU.py
```

仓库根目录的 `packages.txt` 会安装 `fonts-noto-cjk`，用于避免云端 Matplotlib 中文图表显示方框。

## 模型文件

程序默认从以下路径加载 GUI 运行所需模型文件：

```text
artifacts/gui/
  TabPFN_CPU_ST_model.joblib
  TabPFN_CPU_ST_preprocessor.joblib
  shap_explainer_TabPFN_CPU_ST.joblib
```

上述三个 `.joblib` 文件已随仓库提交，用于让 Streamlit Cloud 部署后可以直接完成预测和 SHAP 分析。

## 说明

- 输入特征名保留论文和数据表中的原始缩写，例如 `ST`, `SHAP`, `FT`, `Pe`, `Du`, `AC`。
- `FT` 下拉框显示中文说明，但传递给模型的类别值保持原始英文标签不变。
- Matplotlib 已配置中文字体候选，优先使用 `Microsoft YaHei`, `SimHei`, `SimSun`，并保留 `Noto Sans CJK` 系列作为跨平台候选。
