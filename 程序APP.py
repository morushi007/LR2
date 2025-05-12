# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# —— 必须是第一个 Streamlit 调用 —— #
st.set_page_config(
    page_title="PCNL Post-Operative Fever Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# —— 自定义 CSS —— #
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
    }
    h1 { color: #2C3E50; }
    h2 { color: #3498DB; }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# —— 加载模型（缓存） —— #
@st.cache_resource
def load_model():
    try:
        return joblib.load('LR.pkl')
    except FileNotFoundError:
        st.error("模型文件 'LR.pkl' 未找到，请确保它与本脚本在同一目录下。")
        return None

# —— 页面主体 —— #
st.title("PCNL Post-Operative Fever Prediction Model")
st.markdown("### 基于机器学习的经皮肾镜术后发热风险预测工具")

with st.sidebar:
    st.header("About this Model")
    st.info("""
        本模型基于历史临床数据，使用逻辑回归评估经皮肾镜术后发热风险。
        请在右侧输入患者术前和围术期相关参数，点击“Predict Fever Risk”查看预测结果。
    """)
    st.header("Feature Description")
    st.markdown("""
    - **LMR**: Lymphocyte to Monocyte Ratio  
    - **PLR**: Platelet to Lymphocyte Ratio  
    - **BMI**: Body Mass Index  
    - **Mayo Score**: Mayo Surgical Complexity Score  
    """)

# —— 特征定义 —— #
feature_ranges = {
    "LMR": {"type":"numerical","min":0.0,"max":100.0,"default":5.0,"description":"Lymphocyte to Monocyte Ratio"},
    # （此处略去其余特征定义，与你原来一致）
}

# —— 用户输入 —— #
st.header("Enter Patient Information")
cols = st.columns(3)
input_features = {}
for i, (feat, prop) in enumerate(feature_ranges.items()):
    col = cols[i % 3]
    with col:
        if prop["type"] == "numerical":
            input_features[feat] = st.number_input(
                label=f"{prop['description']} ({feat})",
                min_value=prop["min"],
                max_value=prop["max"],
                value=prop["default"],
                help=f"Range: {prop['min']} – {prop['max']}"
            )
        else:
            input_features[feat] = st.selectbox(
                label=f"{prop['description']} ({feat})",
                options=prop["options"],
                index=prop["options"].index(prop["default"])
            )

st.markdown("---")
if st.button("Predict Fever Risk", use_container_width=True):
    model = load_model()
    if model:
        df = pd.DataFrame([input_features])
        # —— 编码省略（同原来） —— #
        proba = model.predict_proba(df)[0][1] * 100

        # 风险分级
        if proba < 25:
            level, color = "Low Risk", "green"
        elif proba < 50:
            level, color = "Moderate-Low Risk", "lightgreen"
        elif proba < 75:
            level, color = "Moderate-High Risk", "orange"
        else:
            level, color = "High Risk", "red"

        # 展示结果
        st.markdown("## Prediction Results")
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown(f"""
            <div style="padding:20px;border-radius:10px;background-color:{color};text-align:center;">
                <h2 style="color:white;">Risk: {level}</h2>
                <h3 style="color:white;">P(Fever) = {proba:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            ### Interpretation
            - 预测术后发热概率：**{proba:.2f}%**  
            - 风险级别：**{level}**  
            **注意**：本结果仅供参考，不能替代临床决策。
            """)
        with c2:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie([proba, 100-proba], labels=["发热","不发热"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # —— SHAP 解释（同原来） —— #

# —— 页脚和使用说明（同原来） —— #
