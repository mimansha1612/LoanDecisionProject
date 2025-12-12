# Policy Optimization for Financial Decision-Making

## Project Overview
This project demonstrates how a fintech company can improve its loan approval process using **Machine Learning** and **Offline Reinforcement Learning**.  
The goal is to build models that help decide whether to approve or deny a loan application in order to maximize financial return.

We used the **LendingClub Loan Data (2007–2018)** dataset for this project.

---

## Dataset
- Source: [LendingClub Loan Data on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- File used: `accepted_2007_to_2018Q4.csv`
- Columns used (features):  
  - `loan_amnt` – Loan amount requested  
  - `int_rate` – Interest rate  
  - `annual_inc` – Applicant's annual income  
  - `dti` – Debt-to-income ratio  
  - `fico_range_low`, `fico_range_high` – Credit scores  
  - `purpose`, `term` – Loan purpose and term  
- Target: `loan_status` (0 = Fully Paid, 1 = Default)

---

## Task 1: Exploratory Data Analysis & Preprocessing
- Selected meaningful features from 151 columns to simplify analysis.  
- Handled missing values: numeric columns filled with median, categorical columns filled with mode.  
- Encoded categorical variables using one-hot encoding (`purpose` and `term`).  
- Scaled all numeric features for better model performance.

---

## Task 2: Predictive Deep Learning Model
- Built a **Multi-Layer Perceptron (MLP)** using TensorFlow.  
- Train/test split = 80/20.  
- Metrics:
  - **AUC = 0.704**  
  - **F1-score = 0.092**  

> Note: F1 is low due to highly imbalanced dataset (many fully paid loans, few defaults).

---

## Task 3: Offline Reinforcement Learning Agent
- **State (s)** = applicant features  
- **Action (a)** = {0: Deny, 1: Approve}  
- **Reward (r)**:
  - Deny → 0  
  - Approve & Fully Paid → + (loan_amnt × int_rate)  
  - Approve & Default → – (loan_amnt)  
- Policy: Approve loan if predicted default probability < 0.5  
- Estimated Policy Value (average reward) = **–1644.23**

---

## Task 4: Analysis & Comparison
- DL Model predicts default probability → can be converted into approval policy.  
- RL Agent learns policy directly using reward function → may approve high-risk loans for potential profit.  
- Comparison highlights differences in strategy:
  - DL model is more conservative  
  - RL policy can take calculated risks based on reward  
- Limitations:
  - Dataset imbalance affects metrics  
  - RL model is simple, can be improved with more advanced offline RL algorithms  
- Future Steps:
  - Collect more features (e.g., credit history, past loans)  
  - Use ensemble models for better DL prediction  
  - Explore advanced RL algorithms for higher average reward  

---

## How to Run
1. Clone this repo  
2. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   
3.Open Jupyter Notebook and run cells in order:

- EDA + Preprocessing

- Deep Learning Model

- Offline RL Agent

- Metrics are printed at the end of each section.

---

## Project Structure
Loan-Approval-ML-RL-Project/
│
├── data/
│   └── accepted_2007_to_2018Q4.csv
│
├── notebooks/
│   ├── 1_EDA_Preprocessing.ipynb
│   ├── 2_DL_Model.ipynb
│   └── 3_Offline_RL.ipynb
│
├── requirements.txt
├── README.md
└── report.pdf


---

## Result
DL Model: AUC = 0.704, F1-score = 0.092

Offline RL Agent: Estimated Policy Value (Average Reward) = -1644.23

---

## Contact
For Feedbacks and queries kindly contact [bhandarimimansha1612@gmail.com]

