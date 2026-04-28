# A STACKED LSTM &FRAMEWORK FOR REAL-TIME MATERNAL HEALTH RISK STRATIFICATION SYSTEM

**📌 Project Overview**

This project presents a real-time maternal health risk stratification system using IoT sensors and a Stacked LSTM model. The system continuously monitors health parameters of pregnant women and predicts risk levels (Low, Medium, High) to enable early medical intervention.


---

**🎯 Objectives**

Reduce maternal mortality through early risk detection

Provide real-time monitoring using IoT sensors

Build an accurate prediction system using deep learning

Support data-driven healthcare decisions



---

**❗ Problem Statement**

Traditional maternal healthcare systems:

Rely on manual monitoring

Lack real-time tracking

Cannot effectively handle time-series health data

Suffer from imbalanced datasets


This results in delayed diagnosis and increased risk of complications.


---

**💡 Proposed Solution**

I developed an IoT-driven predictive system using machine learning and deep learning techniques:

**🔄 Workflow**

1. Data Collection

IoT sensors collect:

->Age

->Blood Pressure (Systolic & Diastolic)

->Glucose Level

->Body Temperature

->Heart Rate




2. Data Preprocessing

->Noise removal

->Handling missing values

->Normalization



3. Data Balancing

->SMOTE used to handle class imbalance



4. Feature Extraction (Stacked Ensemble)

Base Learners:

->Random Forest Classifier (RFC)

->Gradient Boosting


Meta Learner:

->Extra Trees Classifier (ETC)




5. Prediction Model

->Stacked LSTM captures temporal patterns

->Predicts maternal risk level



6. Output

->Displayed via a Tkinter GUI application





---

**🛠️ Technologies Used**

  **💻 Software**

  Python 3.12

  Tkinter (GUI)

  Windows OS


**🤖 Machine Learning & Deep Learning**

  ->LSTM (Long Short-Term Memory)

  ->Random Forest Classifier (RFC)

  ->Gradient Boosting

  ->Extra Trees Classifier (ETC)

  ->SMOTE (Data Balancing)

  ->EDA (Exploratory Data Analysis)



---

**🏗️ System Architecture**

IoT Sensors → Data Processing → Feature Extraction → LSTM Model → GUI Output



---

**📊 Key Features**

Real-time health monitoring

Early detection of maternal risks

Handles time-series data effectively

Improves prediction accuracy using ensemble techniques

User-friendly GUI for healthcare professionals



---

**📈 Advantages**

Reduces manual effort

Enables remote healthcare monitoring

Improves maternal safety

Supports proactive healthcare decisions




---


**🚀 Future Scope**

Integration with mobile apps

Cloud-based real-time monitoring

Advanced AI models for higher accuracy

Integration with hospital management systems
