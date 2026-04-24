import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dari artifacts
clf_model = joblib.load("classification_pipeline.pkl")
reg_model = joblib.load("regression_pipeline.pkl")


def main():
    st.set_page_config(page_title="Student Placement Prediction", layout="wide")
    st.title("🎓 Student Placement & Salary Prediction")
    st.markdown("---")

    # Sidebar untuk navigasi
    st.sidebar.header("Navigation")
    task = st.sidebar.radio("Choose Task", ["Classification", "Regression"])

    st.sidebar.markdown("---")
    st.sidebar.info("Dataset B – Student Academic & Career Profile")

    if task == "Classification":
        st.header("📋 Placement Status Prediction")
        st.write("Masukkan data mahasiswa untuk memprediksi apakah akan mendapat penempatan kerja.")

    else:
        st.header("💰 Salary Package Prediction")
        st.write("Masukkan data mahasiswa yang sudah ditempatkan untuk memprediksi estimasi gaji.")

    st.markdown("---")

    # ===== Input Form =====
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
            ssc_percentage = st.slider("SSC Percentage (%)", 50, 95, 70)
            hsc_percentage = st.slider("HSC Percentage (%)", 50, 94, 70)
            degree_percentage = st.slider("Degree Percentage (%)", 55, 89, 70)
            cgpa = st.number_input("CGPA", min_value=5.5, max_value=9.8, value=7.5, step=0.1)

        with col2:
            entrance_exam_score = st.slider("Entrance Exam Score", 40, 99, 70)
            technical_skill_score = st.slider("Technical Skill Score", 40, 99, 70)
            soft_skill_score = st.slider("Soft Skill Score", 40, 99, 70)
            internship_count = st.number_input("Internship Count", min_value=0, max_value=4, value=1)
            live_projects = st.number_input("Live Projects", min_value=0, max_value=5, value=2)

        with col3:
            work_experience_months = st.number_input("Work Experience (months)", min_value=0, max_value=24, value=6)
            certifications = st.number_input("Certifications", min_value=0, max_value=5, value=2)
            attendance_percentage = st.slider("Attendance Percentage (%)", 60, 99, 80)
            backlogs = st.number_input("Backlogs", min_value=0, max_value=5, value=0)
            extracurricular_activities = st.radio("Extracurricular Activities", ["Yes", "No"])

        submitted = st.form_submit_button("Make Prediction", use_container_width=True)

    if submitted:
        data = {
            "gender": gender,
            "ssc_percentage": int(ssc_percentage),
            "hsc_percentage": int(hsc_percentage),
            "degree_percentage": int(degree_percentage),
            "cgpa": float(cgpa),
            "entrance_exam_score": int(entrance_exam_score),
            "technical_skill_score": int(technical_skill_score),
            "soft_skill_score": int(soft_skill_score),
            "internship_count": int(internship_count),
            "live_projects": int(live_projects),
            "work_experience_months": int(work_experience_months),
            "certifications": int(certifications),
            "attendance_percentage": int(attendance_percentage),
            "backlogs": int(backlogs),
            "extracurricular_activities": extracurricular_activities
        }

        df_input = pd.DataFrame([data])

        if task == "Classification":
            prediction = clf_model.predict(df_input)[0]
            proba = clf_model.predict_proba(df_input)[0]

            st.markdown("---")
            if prediction == 1:
                st.success(f"✅ Prediction: **PLACED** (Probability: {proba[1]*100:.1f}%)")
            else:
                st.error(f"❌ Prediction: **NOT PLACED** (Probability: {proba[0]*100:.1f}%)")

            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Status": ["Not Placed", "Placed"],
                "Probability": [proba[0], proba[1]]
            })
            st.bar_chart(prob_df.set_index("Status"))

        else:
            prediction = reg_model.predict(df_input)[0]
            st.markdown("---")
            st.success(f"💰 Predicted Salary: **{prediction:.2f} LPA**")

            st.subheader("Input Summary")
            st.dataframe(df_input)


if __name__ == "__main__":
    main()
