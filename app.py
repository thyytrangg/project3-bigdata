import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ================== CONFIG ==================
st.set_page_config(
    page_title="HR Attrition AI Dashboard",
    page_icon="🤖",
    layout="wide"
)

model = joblib.load("best_model.pkl")

# ================== STYLE ==================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.title {
    font-size: 38px;
    font-weight: 700;
    color: #1f2c44;
}
.subtitle {
    font-size: 16px;
    color: #6c757d;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.badge-low {
    color: white;
    background: #2ecc71;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}
.badge-mid {
    color: white;
    background: #f1c40f;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}
.badge-high {
    color: white;
    background: #e74c3c;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown('<div class="title">🤖 HR Attrition AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Du doan kha nang nghi viec cua nhan vien bang Machine Learning</div>', unsafe_allow_html=True)
st.write("")

# ================== INPUT PANEL ==================
st.markdown("## 🧩 Employee Information")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 60, 30)
    dailyrate = st.number_input("Daily Rate", 100, 2000, 800)
    distancefromhome = st.number_input("Distance From Home", 1, 50, 10)

with c2:
    education = st.selectbox("Education (1-5)", [1, 2, 3, 4, 5])
    environmentsatisfaction = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
    jobinvolvement = st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4])

with c3:
    jobsatisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
    totalworkingyears = st.number_input("Total Working Years", 0, 40, 5)
    overtime_text = st.selectbox("Overtime", ["No", "Yes"])

overtime_yes = 1 if overtime_text == "Yes" else 0

st.write("")
predict_btn = st.button("🚀 Predict Attrition Risk", use_container_width=True)

# ================== PREDICTION ==================
if predict_btn:
    input_data = pd.DataFrame([[age, dailyrate, distancefromhome, education,
                                environmentsatisfaction, jobinvolvement,
                                jobsatisfaction, totalworkingyears, overtime_yes]],
                              columns=[
                                  "age",
                                  "dailyrate",
                                  "distancefromhome",
                                  "education",
                                  "environmentsatisfaction",
                                  "jobinvolvement",
                                  "jobsatisfaction",
                                  "totalworkingyears",
                                  "overtime_Yes"
                              ])

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    # ================== RESULT DASHBOARD ==================
    st.divider()
    st.markdown("## 📌 Prediction Result")

    colA, colB, colC, colD = st.columns(4)

    # Risk level
    if proba < 0.3:
        risk = "LOW"
        badge = "badge-low"
    elif proba < 0.6:
        risk = "MEDIUM"
        badge = "badge-mid"
    else:
        risk = "HIGH"
        badge = "badge-high"

    with colA:
        st.metric("Attrition Probability", f"{proba*100:.2f}%")

    with colB:
        st.markdown(f"<span class='{badge}'>Risk: {risk}</span>", unsafe_allow_html=True)

    with colC:
        status = "Leave" if prediction == 1 else "Stay"
        st.metric("Prediction", status)

    with colD:
        st.metric("Confidence", f"{max(proba, 1-proba)*100:.2f}%")

    st.progress(int(proba * 100))

    # ================== CHART 1: DONUT CHART ==================
    st.markdown("## 🍩 Attrition Probability")

    fig1, ax1 = plt.subplots()
    ax1.pie([proba, 1-proba],
            labels=["Leave", "Stay"],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.4))
    ax1.set_title("Probability Distribution")
    st.pyplot(fig1)

    # ================== CHART 2: FEATURE BAR ==================
    st.markdown("## 📊 Employee Feature Profile")

    features = input_data.columns
    values = input_data.values[0]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(features, values)
    ax2.set_title("Feature Values")
    st.pyplot(fig2)

    # ================== CHART 3: RADAR CHART ==================
    st.markdown("## 🧭 Employee Profile Radar")

    radar_features = ["age", "distancefromhome", "education", "jobinvolvement", "jobsatisfaction"]
    radar_values = input_data[radar_features].values[0]
    radar_norm = radar_values / np.max(radar_values)

    angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False)
    radar_norm = np.concatenate((radar_norm, [radar_norm[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig3 = plt.figure(figsize=(6, 6))
    ax3 = plt.subplot(111, polar=True)
    ax3.plot(angles, radar_norm)
    ax3.fill(angles, radar_norm, alpha=0.3)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_features)
    ax3.set_title("Employee Profile Radar")
    st.pyplot(fig3)

    # ================== CHART 4: FEATURE IMPORTANCE ==================
    if hasattr(model, "feature_importances_"):
        st.markdown("## 🧠 Model Feature Importance")

        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.barh(imp_df["Feature"], imp_df["Importance"])
        ax4.invert_yaxis()
        ax4.set_title("Feature Importance")
        st.pyplot(fig4)

    # ================== INSIGHT TEXT ==================
    st.markdown("## 💡 AI Insight")

    insight = []
    if overtime_yes == 1:
        insight.append("Nhan vien lam them gio -> nguy co nghi viec cao hon.")
    if jobsatisfaction <= 2:
        insight.append("Muc do hai long cong viec thap.")
    if environmentsatisfaction <= 2:
        insight.append("Moi truong lam viec khong tot.")
    if distancefromhome > 20:
        insight.append("Khoang cach nha xa noi lam viec.")

    if len(insight) == 0:
        st.success("Khong co yeu to rui ro ro rang.")
    else:
        for i in insight:
            st.warning("• " + i)

# ================== SIDEBAR ==================
st.sidebar.title("📊 About Dashboard")
st.sidebar.write("""
HR Attrition Prediction System

Model: Machine Learning  
Input: HR Employee Data  
Output: Attrition Risk  

Use case:
- HR Analytics
- Employee Retention
- Decision Support
""")
