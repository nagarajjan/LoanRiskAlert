import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
from sklearn.linear_model import LogisticRegression
import warnings
import io
warnings.filterwarnings('ignore')

# Set pandas options to display all columns regardless of count
pd.set_option('display.max_columns', None)
pd.set_option('styler.render.max_columns', None) # Ensures styler uses all columns


# --- 0. Global Column Name Placeholders (used for dynamic naming) ---
COL_CREDIT_SCORE_FORMULA_PLACEHOLDER = 'Credit Score ({} - Delinq * 30)'
COL_PD_NAME = 'Probability of Default (PD)'
COL_EL_NAME = 'Expected Loss (EL = PD * LGD * EAD)'
COL_DTI_NAME = 'Debt-to-Income Ratio (DTI = Debt / Income)'
COL_DSCR_NAME = 'DSCR (NOI / Debt Service)'
COL_LTV_NAME = 'Loan-to-Value Ratio (LTV = Loan / Collateral)'
COL_CU_NAME = 'Credit Utilization (Balance / Limit)'
COL_EWS_NAME = 'early_warning_signals'

# --- 1. Define Default Configuration Directly in Code ---
DEFAULT_CONFIG = {
    'default_lgd': 0.45,
    'economic_outlook_index': 1.0,
    'max_dti_allowed': 0.30,
    'min_dscr_required': 1.25,
    'low_credit_score_threshold': 650,
    'high_pd_threshold': 0.20,
    'high_ltv_threshold': 0.85,
    'base_credit_score_value': 900,
    'risk_factors_to_highlight': [
        'Credit Score (900 - Delinq * 30)',
        COL_PD_NAME,
        COL_EL_NAME,
        COL_DTI_NAME,
        COL_DSCR_NAME,
        COL_LTV_NAME
    ]
}

# --- Initialize Session State for Configuration ---
if 'config' not in st.session_state:
    st.session_state['config'] = DEFAULT_CONFIG

def get_available_factors(base_score):
    return [
        COL_CREDIT_SCORE_FORMULA_PLACEHOLDER.format(base_score),
        COL_PD_NAME,
        COL_EL_NAME,
        COL_DTI_NAME,
        COL_DSCR_NAME,
        COL_LTV_NAME,
        COL_CU_NAME,
        'payment_history_delinquencies (Input Value)',
        'loan_purpose (Input Value)',
        'economic_outlook (Config Value)',
        'loan_amount (Input Value)',
        'income (Input Value)'
    ]

AVAILABLE_FACTORS = get_available_factors(DEFAULT_CONFIG['base_credit_score_value'])

def get_config_download_link(config_dict):
    json_str = json.dumps(config_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="updated_config.json">Download Current Config JSON</a>'

# --- 2. Agentic Logic (Enhanced with new factors) ---
def load_sample_loan_data():
    """Generates a sample DataFrame for the mock model training. (Data placeholders filled with sample data)"""
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


class CreditRiskTools:
    def __init__(self, config):
        self.model = LogisticRegression(random_state=0)
        self._train_mock_pd_model()
        self.lgd = config['default_lgd']
        self.economic_outlook_index = config['economic_outlook_index']
        self.max_dti_allowed = config['max_dti_allowed']
        self.min_dscr_required = config['min_dscr_required']
        self.low_credit_score_threshold = config['low_credit_score_threshold']
        self.high_pd_threshold = config['high_pd_threshold']
        self.high_ltv_threshold = config['high_ltv_threshold']
        self.base_credit_score_value = config['base_credit_score_value']

    def _train_mock_pd_model(self):
        df = load_sample_loan_data()
        feature_cols = ['age', 'income', 'debt_to_income_ratio', 'credit_history_length', 'loan_amount', 'collateral_value', 'total_monthly_debt', 'monthly_net_operating_income', 'revolving_credit_balance', 'total_credit_limit', 'payment_history_delinquencies']
        X = df[feature_cols]
        y = df['is_default']
        self.model.fit(X, y)

    def calculate_credit_score(self, loan_data: dict) -> int:
        score = self.base_credit_score_value - int(loan_data['payment_history_delinquencies'] * 30)
        return min(max(score, 300), 850)
    
    # ... (Rest of the calculation methods remain the same) ...
    def estimate_probability_of_default(self, loan_data: dict) -> float:
        feature_cols = ['age', 'income', 'debt_to_income_ratio', 'credit_history_length', 'loan_amount', 'collateral_value', 'total_monthly_debt', 'monthly_net_operating_income', 'revolving_credit_balance', 'total_credit_limit', 'payment_history_delinquencies']
        features = pd.DataFrame([loan_data], columns=feature_cols)
        pd_estimate = self.model.predict_proba(features)[:, 1] * self.economic_outlook_index
        return round(float(pd_estimate), 4)
    def forecast_expected_loss(self, loan_data: dict, pd: float) -> float:
        expected_loss = pd * self.lgd * loan_data['loan_amount']
        return round(expected_loss, 2)
    def calculate_dscr(self, loan_data: dict) -> float:
        dscr = loan_data['monthly_net_operating_income'] / loan_data['total_monthly_debt'] if loan_data['total_monthly_debt'] > 0 else 0
        return round(dscr, 2)
    def calculate_ltv(self, loan_data: dict) -> float:
        ltv = loan_data['loan_amount'] / loan_data['collateral_value'] if loan_data['collateral_value'] > 0 else 0
        return round(ltv, 2)
    def calculate_credit_utilization(self, loan_data: dict) -> float:
        utilization = loan_data['revolving_credit_balance'] / loan_data['total_credit_limit'] if loan_data['total_credit_limit'] > 0 else 0
        return round(utilization, 2) 
    def generate_early_warning_signals(self, loan_data: dict, credit_score: int, pd: float, dscr: float, ltv: float) -> list[str]:
        signals = []
        if credit_score < self.low_credit_score_threshold: 
            signals.append(f"Low credit score: Below configured threshold of {self.low_credit_score_threshold}.")
        if loan_data['debt_to_income_ratio'] > self.max_dti_allowed: 
            signals.append(f"High DTI ratio (>{self.max_dti_allowed*100:.0f}%).")
        if pd > self.high_pd_threshold: 
            signals.append(f"High probability of default estimate (>{self.high_pd_threshold*100:.0f}%).")
        if dscr < self.min_dscr_required and dscr > 0: 
            signals.append(f"DSCR below minimum requirement (<{self.min_dscr_required}x).")
        if ltv > self.high_ltv_threshold: 
            signals.append(f"High LTV ratio (>{self.high_ltv_threshold*100:.0f}%): Lower collateral cushion.")
        return signals if signals else ["No significant early warning signals detected."]


def color_code_dataframe(df, config):
    def highlight_risk_level(val, threshold, high_color='#ffcccb', low_color='#90ee90', greater_is_risky=True):
        if isinstance(val, (int, float)):
            is_risky = val > threshold if greater_is_risky else val < threshold
            return f'background-color: {high_color}' if is_risky else f'background-color: {low_color}'
        return ''

    def highlight_signals(val):
        if "No significant" in str(val):
            return 'background-color: #90ee90'
        else:
            return 'background-color: #ffcccb'

    current_credit_score_col_name = COL_CREDIT_SCORE_FORMULA_PLACEHOLDER.format(config['base_credit_score_value'])

    column_thresholds = {
        current_credit_score_col_name: {'threshold': config['low_credit_score_threshold'], 'greater_is_risky': False},
        COL_PD_NAME: {'threshold': config['high_pd_threshold'], 'greater_is_risky': True},
        COL_DTI_NAME: {'threshold': config['max_dti_allowed'], 'greater_is_risky': True},
        COL_DSCR_NAME: {'threshold': config['min_dscr_required'], 'greater_is_risky': False},
        COL_LTV_NAME: {'threshold': config['high_ltv_threshold'], 'greater_is_risky': True}
    }
    
    styled_df = df.style

    for col_name, params in column_thresholds.items():
        if col_name in df.columns:
            styled_df = styled_df.applymap(
                lambda v: highlight_risk_level(v, params['threshold'], greater_is_risky=params['greater_is_risky']), 
                subset=[col_name]
            )

    if COL_EWS_NAME in df.columns:
        styled_df = styled_df.applymap(
            highlight_signals,
            subset=[COL_EWS_NAME]
        )

    format_mapping = {
        COL_PD_NAME: "{:.2%}", 
        COL_DTI_NAME: "{:.2%}", 
        COL_LTV_NAME: "{:.2%}"
    }
    
    existing_formats = {col: fmt for col, fmt in format_mapping.items() if col in df.columns}
    
    styled_df = styled_df.format(existing_formats)
        
    return styled_df


def get_html_download_link(styled_df, current_base_score_value, current_eoi_value):
    """Generates an HTML download link, ensuring the filename and content reflect dynamic values."""
    
    # Set display options globally before calling to_html
    # pd.set_option('display.max_columns', None) # Already done at top level
    
    html_string = styled_df.to_html(escape=False)

    # Inject dynamic configuration details into the HTML content itself for complete context in the report
    dynamic_header_html = f"""
    <h1>Loan Risk Assessment Report</h1>
    <p>Report generated with the following configuration values:</p>
    <ul>
        <li>Base Credit Score Value: {current_base_score_value}</li>
        <li>Economic Outlook Index: {current_eoi_value}</li>
        <li>Default LGD: {st.session_state['config']['default_lgd']:.2%}</li>
        <li>Max DTI Allowed (EWS Trigger): {st.session_state['config']['max_dti_allowed']*100:.0f}%</li>
        <li>Min DSCR Required (EWS Trigger): {st.session_state['config']['min_dscr_required']}x</li>
        <li>High PD Threshold (EWS Trigger): {st.session_state['config']['high_pd_threshold']*100:.0f}%</li>
        <li>High LTV Threshold (EWS Trigger): {st.session_state['config']['high_ltv_threshold']*100:.0f}%</li>
    </ul>
    """
    full_html_output = dynamic_header_html + html_string

    b64 = base64.b64encode(full_html_output.encode()).decode()
    filename = f"loan_risk_report_BaseScore{current_base_score_value}_EOI{current_eoi_value}.html"
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Color-Coded HTML Report</a>'


# --- 3. Streamlit Dashboard Setup ---

st.set_page_config(layout="wide", page_title="Loan Risk Dashboard")

# DYNAMIC MAIN DASHBOARD TITLE (Reflects current state)
st.markdown(f"# Loan Risk Assessment Dashboard")
st.markdown(f"**Current Configuration Used:** Base Score = `{st.session_state['config']['base_credit_score_value']}`, Economic Outlook Index = `{st.session_state['config']['economic_outlook_index']}`")


risk_tools = CreditRiskTools(st.session_state['config'])


with st.sidebar:
    st.header("Configuration & Reporting Options")
    uploaded_file = st.file_uploader("Upload Loan Data (Excel File)", type=["xlsx"])
    
    st.subheader("General Risk Parameters & Base Values")

    st.session_state['config']['base_credit_score_value'] = st.number_input(
        "Base Credit Score Value (Max Possible Score)",
        min_value=300, max_value=900, value=st.session_state['config']['base_credit_score_value'], step=10,
        key='base_score_input'
    )
    
    st.session_state['config']['default_lgd'] = st.slider("Default LGD (%)", min_value=0.1, max_value=1.0, value=st.session_state['config']['default_lgd'], step=0.01, format="%.2f", key='lgd_slider')
    st.session_state['config']['economic_outlook_index'] = st.number_input("Economic Outlook Index (PD Multiplier)", min_value=0.5, max_value=2.0, value=st.session_state['config']['economic_outlook_index'], step=0.1, key='eoi_input')
    st.subheader("Early Warning Signal (EWS) Thresholds")
    st.session_state['config']['max_dti_allowed'] = st.slider("DTI Ratio Alert Threshold (%)", min_value=0.05, max_value=0.50, value=st.session_state['config']['max_dti_allowed'], step=0.01, format="%.2f", key='dti_slider_ews')
    st.session_state['config']['min_dscr_required'] = st.number_input("DSCR Minimum Alert Threshold (x)", min_value=0.5, max_value=5.0, value=st.session_state['config']['min_dscr_required'], step=0.05, key='dscr_input_ews')
    st.session_state['config']['low_credit_score_threshold'] = st.number_input("Low Credit Score Alert Below:", min_value=300, max_value=850, value=st.session_state['config']['low_credit_score_threshold'], step=10, key='score_input_ews')
    st.session_state['config']['high_pd_threshold'] = st.slider("High PD Alert Threshold (%)", min_value=0.01, max_value=0.50, value=st.session_state['config']['high_pd_threshold'], step=0.01, format="%.2f", key='pd_slider_ews')
    st.session_state['config']['high_ltv_threshold'] = st.slider("High LTV Alert Threshold (%)", min_value=0.50, max_value=1.0, value=st.session_state['config']['high_ltv_threshold'], step=0.01, format="%.2f", key='ltv_slider_ews')
    st.subheader("Report Display Options")

    current_available_factors = get_available_factors(st.session_state['config']['base_credit_score_value'])

    st.session_state['config']['risk_factors_to_highlight'] = st.multiselect(
        "Choose key metrics/factors to include in the summary report:",
        options=current_available_factors,
        default=[f for f in st.session_state['config']['risk_factors_to_highlight'] if f in current_available_factors],
        key='multiselect_factors'
    )
    st.markdown(get_config_download_link(st.session_state['config']), unsafe_allow_html=True)


# --- 4. Main application logic ---

if uploaded_file:
    try:
        risk_tools = CreditRiskTools(st.session_state['config'])
        loan_df = pd.read_excel(uploaded_file)
        st.subheader("Uploaded Loan Data")
        st.dataframe(loan_df)

        progress_bar = st.progress(0)
        
        current_credit_score_col_name = COL_CREDIT_SCORE_FORMULA_PLACEHOLDER.format(st.session_state['config']['base_credit_score_value'])

        metrics_results = {key: [] for key in get_available_factors(st.session_state['config']['base_credit_score_value'])} 
        metrics_results[COL_EWS_NAME] = []
        
        for i, (index, loan) in enumerate(loan_df.iterrows()):
            loan_data = loan.to_dict()
            credit_score = risk_tools.calculate_credit_score(loan_data)
            pd_estimate = risk_tools.estimate_probability_of_default(loan_data)
            dscr = risk_tools.calculate_dscr(loan_data)
            ltv = risk_tools.calculate_ltv(loan_data)
            expected_loss = risk_tools.forecast_expected_loss(loan_data, pd_estimate)
            credit_utilization = risk_tools.calculate_credit_utilization(loan_data)
            signals = risk_tools.generate_early_warning_signals(loan_data, credit_score, pd_estimate, dscr, ltv)

            metrics_results[current_credit_score_col_name].append(credit_score)
            metrics_results[COL_PD_NAME].append(pd_estimate)
            metrics_results[COL_EL_NAME].append(expected_loss)
            metrics_results[COL_DTI_NAME].append(loan_data['debt_to_income_ratio'])
            metrics_results[COL_DSCR_NAME].append(dscr)
            metrics_results[COL_LTV_NAME].append(ltv)
            metrics_results[COL_CU_NAME].append(credit_utilization)
            metrics_results['payment_history_delinquencies (Input Value)'].append(loan_data['payment_history_delinquencies'])
            metrics_results['loan_purpose (Input Value)'].append(loan_data['loan_purpose'])
            metrics_results['economic_outlook (Config Value)'].append(st.session_state['config']['economic_outlook_index'])
            metrics_results['loan_amount (Input Value)'].append(loan_data['loan_amount'])
            metrics_results['income (Input Value)'].append(loan_data['income'])
            metrics_results[COL_EWS_NAME].append(", ".join(signals))

            progress_bar.progress((i + 1) / len(loan_df))

        # --- Display and Download Logic ---
        results_df_cols = st.session_state['config']['risk_factors_to_highlight'] + [COL_EWS_NAME]
        raw_results_df = pd.DataFrame(metrics_results)[results_df_cols]

        st.subheader("Risk Assessment Summary Report (Color Coded)")
        
        styled_df = color_code_dataframe(raw_results_df, st.session_state['config'])
        st.dataframe(styled_df) 

        st.markdown("### Download Report")
        col_csv, col_html = st.columns(2)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_download_data = convert_df_to_csv(raw_results_df)

        with col_csv:
            st.download_button(
                label="Download as CSV (No Colors)",
                data=csv_download_data,
                file_name='loan_risk_report_raw.csv',
                mime='text/csv'
            )
        
        with col_html:
            st.markdown(get_html_download_link(
                styled_df, 
                st.session_state['config']['base_credit_score_value'], 
                st.session_state['config']['economic_outlook_index']
            ), unsafe_allow_html=True)


    except ValueError as e:
        st.error(f"Error processing file: {e}")
        st.stop()
    except Exception as e:
        st.exception(e) 
        st.stop()

else:
    st.info("Please upload an Excel file using the sidebar to run the loan risk assessment.")
