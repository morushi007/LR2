# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:47:20 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  1 21:14:33 2025
@author: LENOVO
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# —— 页面配置 —— #
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

# —— 标题与说明 —— #
st.title("PCNL Post-Operative Fever Prediction Model")
st.markdown("### 基于机器学习的经皮肾镜术后发热风险预测工具")

# —— 侧边栏 —— #
with st.sidebar:
    st.header("About this Model")
    st.info(
        """
        本模型基于历史临床数据，使用逻辑回归评估经皮肾镜术后发热风险。
        请在右侧输入患者术前和围术期相关参数，点击“Predict Fever Risk”查看预测结果。
        """
    )
    st.header("Feature Description")
    st.markdown("""
    - **LMR**: Lymphocyte to Monocyte Ratio  
    - **PLR**: Platelet to Lymphocyte Ratio  
    - **BMI**: Body Mass Index  
    - **Mayo Score**: Mayo Surgical Complexity Score  
    """)

# —— 加载模型 —— #
@st.cache_resource
def load_model():
    try:
        return joblib.load('LR.pkl')
    except Exception:
        st.error("模型文件 'LR.pkl' 未找到，请确保它与本脚本在同一目录下。")
        return None

# —— 特征定义 —— #
feature_ranges = {
    "LMR": {"type":"numerical","min":0.0,"max":100.0,"default":5.0,"description":"Lymphocyte to Monocyte Ratio"},
    "Preoperative_N": {"type":"numerical","min":0.0,"max":30.0,"default":4.0,"description":"Preoperative Neutrophil Count (×10^9/L)"},
    "Operative_time": {"type":"numerical","min":10,"max":300,"default":60,"description":"Operative Time (minutes)"},
    "Preoperative_WBC": {"type":"numerical","min":0.0,"max":30.0,"default":7.0,"description":"Preoperative White Blood Cell Count (×10^9/L)"},
    "Preoperative_L": {"type":"numerical","min":0.0,"max":10.0,"default":1.8,"description":"Preoperative Lymphocyte Count (×10^9/L)"},
    "PLR": {"type":"numerical","min":0.0,"max":1000.0,"default":120.0,"description":"Platelet to Lymphocyte Ratio"},
    "Preoperative_hemoglobin": {"type":"numerical","min":50.0,"max":200.0,"default":130.0,"description":"Preoperative Hemoglobin (g/L)"},
    "Number_of_stones": {"type":"numerical","min":1,"max":20,"default":1,"description":"Number of Stones"},
    "BMI": {"type":"numerical","min":10.0,"max":50.0,"default":24.0,"description":"Body Mass Index (kg/m²)"},
    "Sex": {"type":"categorical","options":["Male","Female"],"default":"Male","description":"Sex"},
    "Diabetes_mellitus": {"type":"categorical","options":["No","Yes"],"default":"No","description":"Diabetes Mellitus"},
    "UrineLeuk_bin": {"type":"categorical","options":["=0",">0"],"default":"=0","description":"Urine Leukocytes"},
    "Channel_size": {"type":"categorical","options":["18F","20F"],"default":"18F","description":"Channel Size"},
    "degree_of_hydronephrosis": {"type":"categorical","options":["None","Mild","Moderate","Severe"],"default":"None","description":"Degree of Hydronephrosis"},
    "MayoScore_bin": {"type":"categorical","options":["<3","≥3"],"default":"<3","description":"Mayo Score"}
}

# —— 用户输入界面 —— #
st.header("Enter Patient Information")
cols = st.columns(3)
input_features = {}
i = 0
for feat, prop in feature_ranges.items():
    col = cols[i % 3]
    with col:
        if prop["type"] == "numerical":
            input_features[feat] = st.number_input(
                label=f"{prop['description']} ({feat})",
                min_value=float(prop["min"]),
                max_value=float(prop["max"]),
                value=float(prop["default"]),
                help=f"Range: {prop['min']} – {prop['max']}"
            )
        else:
            input_features[feat] = st.selectbox(
                label=f"{prop['description']} ({feat})",
                options=prop["options"],
                index=prop["options"].index(prop["default"]),
                help="请选择"
            )
    i += 1

st.markdown("---")
predict_button = st.button("Predict Fever Risk", use_container_width=True)

# —— 预测与结果展示 —— #
if predict_button:
    model = load_model()
    if model:
        # 准备 DataFrame
        df = pd.DataFrame([input_features])

        # 编码分类变量
        df["Sex"] = df["Sex"].map({"Male":1,"Female":0})
        df["Diabetes_mellitus"] = df["Diabetes_mellitus"].map({"Yes":1,"No":0})
        df["UrineLeuk_bin"] = df["UrineLeuk_bin"].map({">0":1,"=0":0})
        df["Channel_size"] = df["Channel_size"].map({"18F":1,"20F":0})
        hyd_map = {"None":0,"Mild":1,"Moderate":2,"Severe":3}
        df["degree_of_hydronephrosis"] = df["degree_of_hydronephrosis"].map(hyd_map)
        df["MayoScore_bin"] = df["MayoScore_bin"].map({"≥3":1,"<3":0})

        # 做预测
        proba = model.predict_proba(df)[0][1] * 100  # 取“发热”概率
        # 风险分级
        if proba < 25:
            level, color = "Low Risk", "green"
        elif proba < 50:
            level, color = "Moderate-Low Risk", "lightgreen"
        elif proba < 75:
            level, color = "Moderate-High Risk", "orange"
        else:
            level, color = "High Risk", "red"

        # 左侧结果展示
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
            ax.pie([proba, 100-proba],
                   labels=["发热","不发热"],
                   autopct="%1.1f%%",
                   startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        # —— 修正后的 SHAP 解释 —— #
        try:
            st.markdown("## Feature Impact Analysis")
            st.info(
                "下图展示各特征对“发热”概率的影响：向右（红色）推高发热风险，向左（蓝色）降低风险。"
            )

            # 仅解释 P(fever)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:,1],
                shap.sample(df, 5)
            )
            shap_values = explainer.shap_values(df)  # 形状 (1×n_features)

            # Summary Plot
            fig, ax = plt.subplots(figsize=(10,6))
            shap.summary_plot(
                shap_values,
                df,
                feature_names=df.columns.tolist(),
                show=False
            )
            st.pyplot(fig)
            plt.clf()

            # Waterfall Plot
            fig, ax = plt.subplots(figsize=(10,6))
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_values[0],
                feature_names=df.columns.tolist(),
                show=False
            )
            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.warning(f"无法生成 SHAP 解释：{e}")
            st.markdown("""
            可能原因：  
            1. SHAP 版本兼容问题  
            2. 模型类型不支持当前解释器  
            3. 输入数据格式与训练时不符  
            """)

# —— 页脚 —— #
st.markdown("""
<div class="footer">
    <p>© 2025 PCNL Post-Operative Fever Prediction Model | 本工具仅供学术参考，不作临床诊断依据</p>
</div>
""", unsafe_allow_html=True)

# —— 使用说明 —— #
with st.expander("How to Use This Tool"):
    st.markdown("""
    1. 填入各项临床参数  
    2. 点击 “Predict Fever Risk”  
    3. 查看预测结果及特征影响分析  

    **注意**：所有输入应在指定范围内，本模型基于历史数据训练，可能不适用于所有临床场景。
    """)
