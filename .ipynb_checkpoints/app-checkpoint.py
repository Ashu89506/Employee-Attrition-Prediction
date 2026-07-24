import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide"
)

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
model = joblib.load("Models/employee_attrition_model.pkl")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("👨‍💼 Employee Attrition AI")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    ### About Project

    This Machine Learning application predicts whether an employee is likely to leave the company.

    **Algorithm**
    - Logistic Regression

    **Developer**
    - Hemant Kumar

    **Version**
    - 1.0
    """
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded Successfully")

# -------------------------------------------------
# Main Title
# -------------------------------------------------

st.title("🏢 Employee Attrition Prediction System")

st.markdown(
"""
This application predicts whether an employee is **likely to leave the company** based on HR-related information.

Fill all employee details and click **Predict Attrition**.
"""
)

st.markdown("---")

# -------------------------------------------------
# Layout
# -------------------------------------------------

col1, col2 = st.columns(2)

# =================================================
# LEFT COLUMN
# =================================================

with col1:

    st.subheader("👤 Personal Information")

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    Gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Divorced",
            "Married",
            "Single"
        ]
    )

    Education = st.selectbox(
        "Education",
        [
            1,
            2,
            3,
            4,
            5
        ]
    )

    EducationField = st.selectbox(
        "Education Field",
        [
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"
        ]
    )

    DistanceFromHome = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
        value=5
    )

    BusinessTravel = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )
# =================================================
# RIGHT COLUMN
# =================================================

with col2:

    st.subheader("💼 Job Information")

    Department = st.selectbox(
        "Department",
        [
            "Human Resources",
            "Research & Development",
            "Sales"
        ]
    )

    JobRole = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    JobLevel = st.selectbox(
        "Job Level",
        [
            1,
            2,
            3,
            4,
            5
        ]
    )

    JobInvolvement = st.selectbox(
        "Job Involvement",
        [
            1,
            2,
            3,
            4
        ]
    )

    JobSatisfaction = st.selectbox(
        "Job Satisfaction",
        [
            1,
            2,
            3,
            4
        ]
    )

    EnvironmentSatisfaction = st.selectbox(
        "Environment Satisfaction",
        [
            1,
            2,
            3,
            4
        ]
    )

    RelationshipSatisfaction = st.selectbox(
        "Relationship Satisfaction",
        [
            1,
            2,
            3,
            4
        ]
    )

    WorkLifeBalance = st.selectbox(
        "Work Life Balance",
        [
            1,
            2,
            3,
            4
        ]
    )

    OverTime = st.selectbox(
        "Over Time",
        [
            "No",
            "Yes"
        ]
    )

st.markdown("---")

st.subheader("💰 Salary & Experience")

col3, col4 = st.columns(2)

with col3:

    DailyRate = st.number_input(
        "Daily Rate",
        min_value=100,
        max_value=1500,
        value=800
    )

    HourlyRate = st.number_input(
        "Hourly Rate",
        min_value=30,
        max_value=100,
        value=60
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=200000,
        value=5000
    )

    MonthlyRate = st.number_input(
        "Monthly Rate",
        min_value=1000,
        max_value=30000,
        value=15000
    )

with col4:

    NumCompaniesWorked = st.number_input(
        "Companies Worked",
        min_value=0,
        max_value=10,
        value=2
    )

    PercentSalaryHike = st.number_input(
        "Salary Hike %",
        min_value=10,
        max_value=30,
        value=15
    )

    PerformanceRating = st.selectbox(
        "Performance Rating",
        [
            3,
            4
        ]
    )

    StockOptionLevel = st.selectbox(
        "Stock Option Level",
        [
            0,
            1,
            2,
            3
        ]
    )
st.markdown("---")

st.subheader("📈 Company Experience")

col5, col6 = st.columns(2)

with col5:

    TotalWorkingYears = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=10
    )

    YearsAtCompany = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    YearsInCurrentRole = st.number_input(
        "Years In Current Role",
        min_value=0,
        max_value=20,
        value=3
    )

with col6:

    YearsSinceLastPromotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        max_value=15,
        value=1
    )

    YearsWithCurrManager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        max_value=20,
        value=3
    )

    TrainingTimesLastYear = st.number_input(
        "Training Times Last Year",
        min_value=0,
        max_value=10,
        value=2
    )

st.markdown("---")

# ================================================
# Prediction Button
# ================================================

if st.button(
    "🔍 Predict Attrition",
    use_container_width=True
):

    input_data = {

        # Numeric Features
        "Age": Age,
        "DailyRate": DailyRate,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EmployeeCount": 1,
        "EmployeeNumber": 1,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobSatisfaction": JobSatisfaction,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StandardHours": 80,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,

        # Default One-Hot Encoded Features
        "BusinessTravel_Travel_Frequently": 0,
        "BusinessTravel_Travel_Rarely": 0,

        "Department_Research & Development": 0,
        "Department_Sales": 0,

        "EducationField_Life Sciences": 0,
        "EducationField_Marketing": 0,
        "EducationField_Medical": 0,
        "EducationField_Other": 0,
        "EducationField_Technical Degree": 0,

        "Gender_Male": 0,

        "JobRole_Human Resources": 0,
        "JobRole_Laboratory Technician": 0,
        "JobRole_Manager": 0,
        "JobRole_Manufacturing Director": 0,
        "JobRole_Research Director": 0,
        "JobRole_Research Scientist": 0,
        "JobRole_Sales Executive": 0,
        "JobRole_Sales Representative": 0,

        "MaritalStatus_Married": 0,
        "MaritalStatus_Single": 0,

        "OverTime_Yes": 0
    }
    # ==========================================
    # One-Hot Encoding
    # ==========================================

    # Business Travel
    if BusinessTravel == "Travel_Frequently":
        input_data["BusinessTravel_Travel_Frequently"] = 1
    elif BusinessTravel == "Travel_Rarely":
        input_data["BusinessTravel_Travel_Rarely"] = 1

    # Department
    if Department == "Research & Development":
        input_data["Department_Research & Development"] = 1
    elif Department == "Sales":
        input_data["Department_Sales"] = 1

    # Education Field
    if EducationField == "Life Sciences":
        input_data["EducationField_Life Sciences"] = 1
    elif EducationField == "Marketing":
        input_data["EducationField_Marketing"] = 1
    elif EducationField == "Medical":
        input_data["EducationField_Medical"] = 1
    elif EducationField == "Other":
        input_data["EducationField_Other"] = 1
    elif EducationField == "Technical Degree":
        input_data["EducationField_Technical Degree"] = 1

    # Gender
    if Gender == "Male":
        input_data["Gender_Male"] = 1

    # Job Role
    if JobRole == "Human Resources":
        input_data["JobRole_Human Resources"] = 1
    elif JobRole == "Laboratory Technician":
        input_data["JobRole_Laboratory Technician"] = 1
    elif JobRole == "Manager":
        input_data["JobRole_Manager"] = 1
    elif JobRole == "Manufacturing Director":
        input_data["JobRole_Manufacturing Director"] = 1
    elif JobRole == "Research Director":
        input_data["JobRole_Research Director"] = 1
    elif JobRole == "Research Scientist":
        input_data["JobRole_Research Scientist"] = 1
    elif JobRole == "Sales Executive":
        input_data["JobRole_Sales Executive"] = 1
    elif JobRole == "Sales Representative":
        input_data["JobRole_Sales Representative"] = 1

    # Marital Status
    if MaritalStatus == "Married":
        input_data["MaritalStatus_Married"] = 1
    elif MaritalStatus == "Single":
        input_data["MaritalStatus_Single"] = 1

    # Over Time
    if OverTime == "Yes":
        input_data["OverTime_Yes"] = 1
            # ==========================================
    # Create DataFrame
    # ==========================================

    input_df = pd.DataFrame([input_data])

    # Keep feature order exactly as during training
    input_df = input_df[model.feature_names_in_]

    # ==========================================
    # Prediction
    # ==========================================

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    stay_prob = probability[0][0] * 100
    leave_prob = probability[0][1] * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave the company.")
    else:
        st.success("✅ Employee is likely to stay in the company.")

    # ==========================================
    # Prediction Probability
    # ==========================================

    col7, col8 = st.columns(2)

    with col7:
        st.metric(
            label="Stay Probability",
            value=f"{stay_prob:.2f}%"
        )

    with col8:
        st.metric(
            label="Leave Probability",
            value=f"{leave_prob:.2f}%"
        )
    # ==========================================
    # Plotly Bar Chart
    # ==========================================

    st.markdown("---")
    st.subheader("📊 Prediction Probability")

    probability_df = pd.DataFrame({
        "Prediction": [
            "Stay",
            "Leave"
        ],
        "Probability": [
            stay_prob,
            leave_prob
        ]
    })

    fig_bar = px.bar(
        probability_df,
        x="Prediction",
        y="Probability",
        text="Probability",
        color="Prediction",
        title="Employee Attrition Probability"
    )

    fig_bar.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_bar.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Prediction",
        height=450,
        showlegend=False
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )
    # ==========================================
    # Plotly Pie Chart
    # ==========================================

    st.markdown("---")
    st.subheader("🥧 Prediction Distribution")

    fig_pie = px.pie(
        probability_df,
        names="Prediction",
        values="Probability",
        title="Employee Attrition Probability Distribution",
        hole=0.45
    )

    fig_pie.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ==========================================
    # Employee Summary
    # ==========================================

    st.markdown("---")
    st.subheader("📋 Employee Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Age",
            "Monthly Income",
            "Total Working Years",
            "Years At Company",
            "Job Level",
            "Department",
            "Job Role",
            "Over Time"
        ],
        "Value": [
            Age,
            MonthlyIncome,
            TotalWorkingYears,
            YearsAtCompany,
            JobLevel,
            Department,
            JobRole,
            OverTime
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )
    # ==========================================
    # Employee Metrics Dashboard
    # ==========================================

    st.markdown("---")
    st.subheader("📊 Employee Dashboard")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Age",
            value=Age
        )

    with metric_col2:
        st.metric(
            label="Monthly Income",
            value=f"₹ {MonthlyIncome:,}"
        )

    with metric_col3:
        st.metric(
            label="Experience",
            value=f"{TotalWorkingYears} Years"
        )

    with metric_col4:
        st.metric(
            label="Years At Company",
            value=YearsAtCompany
        )

    # ==========================================
    # Recommendation
    # ==========================================

    st.markdown("---")
    st.subheader("💡 Recommendation")

    if prediction[0] == 1:

        st.warning(
            """
### Employee is at risk of attrition.

Recommended Actions:

- Improve employee engagement.
- Review salary and benefits.
- Discuss career growth opportunities.
- Reduce overtime if possible.
- Conduct a one-to-one feedback session.
            """
        )

    else:

        st.success(
            """
### Employee is likely to stay.

Positive Indicators:

- Good employee stability.
- Continue recognition and rewards.
- Maintain work-life balance.
- Encourage continuous learning.
            """
        )
    # ==========================================
    # Download Prediction Report
    # ==========================================

    st.markdown("---")
    st.subheader("📥 Download Prediction Report")

    report_df = pd.DataFrame({
        "Field": [
            "Prediction",
            "Stay Probability (%)",
            "Leave Probability (%)",
            "Age",
            "Gender",
            "Department",
            "Job Role",
            "Monthly Income",
            "Years At Company",
            "Total Working Years",
            "Over Time"
        ],
        "Value": [
            "Stay" if prediction[0] == 0 else "Leave",
            round(stay_prob, 2),
            round(leave_prob, 2),
            Age,
            Gender,
            Department,
            JobRole,
            MonthlyIncome,
            YearsAtCompany,
            TotalWorkingYears,
            OverTime
        ]
    })

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download CSV Report",
        data=csv,
        file_name="employee_attrition_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ==========================================
    # Footer
    # ==========================================

    st.markdown("---")

    st.markdown(
        """
        <div style='text-align:center; color:gray;'>

        ### 🏢 Employee Attrition Prediction System

        Developed using **Python • Streamlit • Scikit-learn • Plotly**

        **Developer:** Hemant Kumar

        </div>
        """,
        unsafe_allow_html=True
    )
# =====================================================
# Custom CSS
# =====================================================

st.markdown("""
<style>

/* Hide Streamlit default menu & footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main App */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#f5f7fa;
}

/* Buttons */
.stButton > button{
    width:100%;
    height:3.2em;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    border:1px solid #E5E7EB;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# End Message
# =====================================================

st.markdown("---")

st.caption(
    "© 2026 Employee Attrition Prediction AI | Built with Streamlit, Scikit-learn & Plotly"
)