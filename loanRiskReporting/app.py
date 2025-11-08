import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')



# --- Load configuration ---
# In your Streamlit script (around line 7 or wherever you open config.json)
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    st.error("`config.json` not found...")
    st.stop()

# --- 1. Agentic Logic (Enhanced with new factors) ---
# --- 1. Agentic Logic (Enhanced with new factors) ---
def load_sample_loan_data():
    """Generates a sample DataFrame for the mock model training."""
    # This data is only used to train the mock internal model, not your uploaded data.
    data = {
        'age': [35, 45, 28, 52, 41, 33, 58, 29, 38, 49],
        'income': [60000, 90000, 45000, 120000, 75000, 50000, 110000, 55000, 65000, 95000],
        'debt_to_income_ratio': [0.15, 0.22, 0.35, 0.1, 0.28, 0.40, 0.08, 0.25, 0.18, 0.12],
        'credit_history_length': [5, 10, 2, 20, 8, 4, 25, 3, 7, 15],
        'loan_amount': [10000, 40000, 15000, 80000, 20000, 25000, 60000, 12000, 18000, 50000],
        'collateral_value': [20000, 70000, 35000, 120000, 40000, 50000, 90000, 30000, 25000, 60000],
        'total_monthly_debt': [200, 500, 1000, 300, 700, 1200, 150, 400, 350, 250],
        'monthly_net_operating_income': [3000, 6000, 2500, 10000, 3000, 2000, 8000, 4000, 3000, 5000],
        'revolving_credit_balance': [2000, 5000, 8000, 1000, 6000, 9000, 500, 3000, 2500, 1500],
        'total_credit_limit': [10000, 20000, 10000, 50000, 15000, 10000, 30000, 10000, 8000, 20000],
        'payment_history_delinquencies': [0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
        'loan_purpose': ['Home Imp', 'Auto', 'Debt Cons', 'Home', 'Auto', 'Debt Cons', 'Home', 'Auto', 'Debt Cons', 'Home'],
        'is_default': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

# The rest of the script continues below this function...


class CreditRiskTools:
    # ... [init and _train_mock_pd_model remain the same] ...
    def __init__(self):
        self.model = LogisticRegression(random_state=0)
        self._train_mock_pd_model()
        self.lgd = CONFIG['default_lgd']

    def _train_mock_pd_model(self):
        df = load_sample_loan_data()
        feature_cols = ['age', 'income', 'debt_to_income_ratio', 'credit_history_length', 'loan_amount', 'collateral_value', 'total_monthly_debt', 'monthly_net_operating_income', 'revolving_credit_balance', 'total_credit_limit', 'payment_history_delinquencies']
        X = df[feature_cols]
        y = df['is_default']
        self.model.fit(X, y)

    def calculate_credit_score(self, loan_data: dict) -> int:
        # Simplified for demo:
        score = 600 - int(loan_data['payment_history_delinquencies'] * 30)
        return min(max(score, 300), 850)

    def estimate_probability_of_default(self, loan_data: dict) -> float:
        feature_cols = ['age', 'income', 'debt_to_income_ratio', 'credit_history_length', 'loan_amount', 'collateral_value', 'total_monthly_debt', 'monthly_net_operating_income', 'revolving_credit_balance', 'total_credit_limit', 'payment_history_delinquencies']
        features = pd.DataFrame([loan_data], columns=feature_cols)
        pd_estimate = self.model.predict_proba(features)[:, 1] * CONFIG['economic_outlook_index'] # Adjust PD by economic outlook
        return round(float(pd_estimate), 4)

    def forecast_expected_loss(self, loan_data: dict, pd: float) -> float:
        # Use configurable LGD
        expected_loss = pd * self.lgd * loan_data['loan_amount']
        return round(expected_loss, 2)
  
    # --- New Metric Calculators ---
    def calculate_dscr(self, loan_data: dict) -> float:
        # DSCR = Net Operating Income / Total Debt Service (assumes monthly for this example)
        dscr = loan_data['monthly_net_operating_income'] / loan_data['total_monthly_debt'] if loan_data['total_monthly_debt'] > 0 else 0
        return round(dscr, 2)

    def calculate_ltv(self, loan_data: dict) -> float:
        # LTV = Loan Amount / Collateral Value
        ltv = loan_data['loan_amount'] / loan_data['collateral_value'] if loan_data['collateral_value'] > 0 else 0
        return round(ltv, 2)

    def calculate_credit_utilization(self, loan_data: dict) -> float:
        # Credit Utilization = Revolving Balance / Total Credit Limit
        utilization = loan_data['revolving_credit_balance'] / loan_data['total_credit_limit'] if loan_data['total_credit_limit'] > 0 else 0
        # FIX: Add the comma between 'utilization' and '2'
        return round(utilization, 2) 


    def generate_early_warning_signals(self, loan_data: dict, credit_score: int, pd: float, dscr: float, ltv: float) -> list[str]:
        signals = []
        
        # Retrieve the max DTI threshold from the config file
        max_dti_allowed = CONFIG.get('max_dti_allowed', 0.3) # Use 0.3 as a default fallback

        if credit_score < 650: 
            signals.append("Low credit score: Higher risk potential.")
            
        # Use the config parameter for comparison and messaging
        if loan_data['debt_to_income_ratio'] > max_dti_allowed: 
            signals.append(f"High DTI ratio (>{max_dti_allowed*100:.0f}%).")
            
        if pd > 0.2: 
            signals.append("High probability of default estimate.")
            
        if dscr < CONFIG['min_dscr_required'] and dscr > 0: 
            signals.append(f"DSCR below minimum requirement (<{CONFIG['min_dscr_required']}x).")
            
        if ltv > 0.85: 
            signals.append("High LTV ratio (>85%): Lower collateral cushion.")
            
        return signals if signals else ["No significant early warning signals detected."]


# --- 2. Streamlit Dashboard, Factor Selection, and Download Code ---

st.set_page_config(layout="wide", page_title="Loan Risk Dashboard")
st.title("Loan Risk Assessment Dashboard")

risk_tools = CreditRiskTools()

# List of all available factors/metrics
AVAILABLE_FACTORS = [
    'credit_score', 'probability_of_default (PD)', 'expected_loss (EL)', 
    'debt_to_income_ratio', 'dscr', 'loan_to_value_ratio (LTV)', 'credit_utilization_ratio',
    'payment_history_delinquencies', 'loan_purpose', 'economic_outlook', 'loan_amount', 'income'
]

with st.sidebar:
    st.header("Configuration & Reporting Options")
    uploaded_file = st.file_uploader("Upload Loan Data (Excel File)", type=["xlsx"])
    
    st.subheader("Select Factors for Detailed Report")
    selected_factors = st.multiselect(
        "Choose key metrics/factors to include in the summary report:",
     options=AVAILABLE_FACTORS,
        default=CONFIG['risk_factors_to_highlight']
    )

if uploaded_file:
    try:
        loan_df = pd.read_excel(uploaded_file)
        # Ensure new input columns are present
        required_inputs = ['collateral_value', 'total_monthly_debt', 'monthly_net_operating_income', 'revolving_credit_balance', 'total_credit_limit', 'payment_history_delinquencies', 'loan_purpose']
        for col in required_inputs:
            if col not in loan_df.columns:
                st.warning(f"Missing required input column in Excel file: '{col}'. Some metrics may not be accurate.")
                loan_df[col] = 0 # Add default value to allow processing to continue

        st.subheader("Uploaded Loan Data")
        st.dataframe(loan_df)

        progress_bar = st.progress(0)
        # Lists to store all new metrics
        metrics_results = {key: [] for key in AVAILABLE_FACTORS}
        metrics_results['early_warning_signals'] = []
        metrics_results['economic_outlook'] = []
        
        for i, (index, loan) in enumerate(loan_df.iterrows()):
            loan_data = loan.to_dict()
            credit_score = risk_tools.calculate_credit_score(loan_data)
            pd_estimate = risk_tools.estimate_probability_of_default(loan_data)
            dscr = risk_tools.calculate_dscr(loan_data)
            ltv = risk_tools.calculate_ltv(loan_data)
            expected_loss = risk_tools.forecast_expected_loss(loan_data, pd_estimate)
            credit_utilization = risk_tools.calculate_credit_utilization(loan_data)
            warnings = risk_tools.generate_early_warning_signals(loan_data, credit_score, pd_estimate, dscr, ltv)

            # Map results to storage dictionary
            metrics_results['credit_score'].append(credit_score)
            metrics_results['probability_of_default (PD)'].append(pd_estimate)
            metrics_results['expected_loss (EL)'].append(expected_loss)
            metrics_results['debt_to_income_ratio'].append(loan_data['debt_to_income_ratio'])
            metrics_results['dscr'].append(dscr)
            metrics_results['loan_to_value_ratio (LTV)'].append(ltv)
            metrics_results['credit_utilization_ratio'].append(credit_utilization)
            metrics_results['payment_history_delinquencies'].append(loan_data['payment_history_delinquencies'])
            metrics_results['loan_purpose'].append(loan_data['loan_purpose'])
            metrics_results['economic_outlook'].append(f"Index: {CONFIG['economic_outlook_index']}")
            metrics_results['loan_amount'].append(loan_data['loan_amount'])
            metrics_results['income'].append(loan_data['income'])
            metrics_results['early_warning_signals'].append("; ".join(warnings))

            progress_bar.progress((i + 1) / len(loan_df))
        # Add new metric columns to the DataFrame
        for key, value_list in metrics_results.items():
            if key not in loan_df.columns: # Avoid overwriting input columns like DTI if they existed
                 loan_df[key] = value_list
        
        st.subheader("Loan Data with All Generated Metrics (Full Report)")
        st.dataframe(loan_df)

        # --- Section for the selective report based on user choices ---
        if selected_factors:
            st.subheader("Selected Factors Summary Report")
            summary_cols = list(set(selected_factors + ['early_warning_signals'])) # Ensure warnings are always shown
            summary_df = loan_df[summary_cols]
            st.dataframe(summary_df)
        
        # --- Download Button Code ---
        st.markdown("*Download Processed Data:*")
        csv_data = loan_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Report as CSV",
            data=csv_data,
            file_name='loan_risk_metrics_full_report.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"Error processing the file. Ensure correct columns are present in your config and Excel file. Error details: {e}")

# Run the app using: streamlit run app.py