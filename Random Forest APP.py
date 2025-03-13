import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的ExtraTrees模型
model = joblib.load('Random Forest.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "AGE (Age in years)": {"type": "numerical", "min": 0.0, "max": 18.0, "default": 5.0},
    "WT (Weight in kg)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 20.0},
    "Daily_Dose (Daily Dose)": {"type": "numerical", "min": 0.0, "max": 4000.0, "default": 2000.0},
    "Single_Dose (Single Dose)": {"type": "numerical", "min": 0.0, "max": 4000.0, "default": 450.0},
    "VPA (1 = Combined with VPA, 0 = Combined without VPA)": {"type": "categorical", "options": [0, 1], "default": 0},
    "Terms (Outpatient or Hospitalized)": {"type": "categorical", "options": [0, 1], "default": 0},
    "Cmin (Trough concentration)": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 15.0},
    "DBIL (Direct Bilirubin)": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 5.0},
    "TBIL (Total Bilirubin)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 5.0},
    "ALT (Alanine Aminotransferase)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 20.0},
    "AST (Aspartate Aminotransferase)": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 20.0},
    "SCR (Serum Creatinine)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 35.0},
    "BUN (Blood Urea Nitrogen)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 5.0},
    "CLCR (Creatinine Clearance Rate)": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 90.0},
    "HGB (Hemoglobin)": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 120.0},
    "HCT (Hematocrit)": {"type": "numerical", "min": 0.0, "max": 200.0, "default": 35.0},
    "MCH (Mean Corpuscular Hemoglobin)": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 30.0},
    "MCHC (Mean Corpuscular Hemoglobin Concentration)": {"type": "numerical", "min": 0.0, "max": 500.0, "default": 345.0}
}


# Streamlit 界面
st.title("Antiepileptic Drug (OXC) Treatment Outcome Prediction with SHAP Visualization")

# Description 
st.write("""
This app predicts the likelihood of antiepileptic drug (OXC) treatment outcome based on input features.
Select the Random Forest (RF) model, input feature values, and get predictions and probability estimates.
""")


# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of good responder (GR) is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
