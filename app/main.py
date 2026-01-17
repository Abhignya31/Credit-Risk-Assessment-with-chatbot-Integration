# main.py
import streamlit as st, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
# NOTE: database.py and virtual_assistant.py must exist and contain the code from previous steps
from database import create_db, insert_applicant, fetch_all_applicants
from utils import build_feature_row, scale_features, predict_proba, save_uploaded_file, docs_to_json, SHAP_EXPLAINER, get_shap_explanation, XGB_MODEL
from virtual_assistant import assistant_dialogue

st.set_page_config(page_title='Credit Risk Pro | Advanced ML + LLM', layout='wide')
st.title('Credit Risk Assessment Pro — Advanced ML & LLM Guidance')
create_db()

# Initialize session state for context data
if 'current_context' not in st.session_state:
    st.session_state['current_context'] = {}

col1, col2 = st.columns([2,1])

with col1:
    st.header('Applicant & Loan Details')
    with st.form('form'):
        # --- Input Fields ---
        name = st.text_input('Full Name', value='John Doe')
        dob = st.date_input('Date of Birth')
        address = st.text_area('Address')
        phone = st.text_input('Phone')
        email = st.text_input('Email')
        st.subheader('Loan')
        loan_amnt = st.number_input('Loan Amount', value=15000.0, step=1000.0)
        int_rate = st.number_input('Interest Rate (%)', value=12.5, step=0.1)
        fico_low = st.number_input('FICO Score (Low)', min_value=300, max_value=850, value=680, step=1)
        annual_inc = st.number_input('Annual Income', value=55000.0, step=1000.0)
        st.subheader('Guarantor (optional)')
        guarantor_name = st.text_input('Guarantor Name')
        guarantor_relation = st.text_input('Relation')
        guarantor_phone = st.text_input('Guarantor Phone')
        guarantor_address = st.text_area('Guarantor Address')
        st.subheader('Upload Documents')
        aadhaar = st.file_uploader('Aadhaar / ID', type=['pdf','jpg','png','jpeg'])
        pan = st.file_uploader('PAN', type=['pdf','jpg','png','jpeg'])
        income_file = st.file_uploader('Income Proof', type=['pdf','jpg','png','jpeg'])
        property_file = st.file_uploader('Property Document', type=['pdf','jpg','png','jpeg'])
        guarantor_proof = st.file_uploader('Guarantor Proof', type=['pdf','jpg','png','jpeg'])
        st.subheader('Insurance Option')
        insurance_opted = st.checkbox('Opt for Loan Protection Insurance')
        insurance_rate = st.number_input('Insurance premium rate (%)', value=2.0, step=0.1)
        submitted = st.form_submit_button('Assess & Save')
    
    if submitted:
        # --- 1. Preprocessing & Prediction ---
        doc_paths = {}
        doc_paths['aadhaar'] = save_uploaded_file(aadhaar, 'aadhaar') if aadhaar else None
        doc_paths['pan'] = save_uploaded_file(pan, 'pan') if pan else None
        doc_paths['income'] = save_uploaded_file(income_file, 'income') if income_file else None
        doc_paths['property'] = save_uploaded_file(property_file, 'property') if property_file else None
        doc_paths['guarantor'] = save_uploaded_file(guarantor_proof, 'guarantor') if guarantor_proof else None
        
        # Build the initial 15-feature DataFrame
        df_row = build_feature_row(loan_amnt, int_rate, fico_low, annual_inc)
        
        # Scale the data (this function selects only the 12 training features and returns the scaled DataFrame)
        Xs_df = scale_features(df_row) 
        
        # Base ML Prediction Score
        base_prob = predict_proba(Xs_df) 
        
        # --- 2. Business Rule Adjustments (Manual/Heuristic) ---
        notes = []
        adj = 0.0
        if guarantor_name and guarantor_phone:
            notes.append('Guarantor provided - reduces risk')
            adj -= 0
        if property_file:
            notes.append('Collateral provided - reduces risk')
            adj -= 0.10
        if income_file:
            notes.append('Income proof uploaded - reduces risk slightly')
            adj -= 0.05
        if insurance_opted:
            prem = (insurance_rate/100.0)*loan_amnt
            notes.append(f'Insurance opted - premium ₹{prem:.2f}')
            adj -= 0.07
        if fico_low < 600:
            notes.append('Low FICO - increases risk')
            adj += 0.18
        if int_rate > 20:
            notes.append('Very high interest rate - increases risk')
            adj += 0.12
        
        # Ensure 'DebtRatio' is extracted correctly from the 15-feature df_row before adjustment
        debt = df_row['DebtRatio'].values[0]
        if debt > 1.0:
            notes.append('High debt-to-income - increases risk')
            adj += 0.20
            
        adjusted_prob = base_prob + adj
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))

        # --- 3. Display Results ---
        st.subheader('Advanced ML Model Assessment')
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric('Base ML Risk (%)', f'{base_prob*100:.2f}%')
        col_m2.metric('Adjustment', f'{adj*100:+.2f}%')
        col_m3.metric('Final Adjusted Risk (%)', f'{adjusted_prob*100:.2f}%')

        # --- SHAP Explanation (Advanced Visualization) ---
        top_features = ['N/A']
        
        if XGB_MODEL is not None and SHAP_EXPLAINER is not None and not Xs_df.empty and not pd.isna(base_prob):
            
            explanation = get_shap_explanation(Xs_df)
            
            if explanation is not None and hasattr(explanation, 'values') and len(explanation.values) > 0:
                st.write('### Feature Contribution to Risk Score (SHAP Waterfall Plot)')
                st.markdown('**:red[Red] features push risk higher], :blue[Blue] features push risk lower.**')
                
                # 1. Create a new Matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 2. Generate the waterfall plot directly onto the figure
                shap.plots.waterfall(explanation, show=False)
                
                # 3. Pass the figure object to Streamlit
                st.pyplot(fig, bbox_inches='tight')
                
                # 4. CRITICAL: Close the figure to free memory and avoid conflicts
                plt.close(fig) 
                
                # Get top features from the explanation object
                top_feature_indices = np.argsort(np.abs(explanation.values))[-3:][::-1]
                top_features = [explanation.feature_names[i] for i in top_feature_indices]
                
            else:
                st.error("SHAP plot could not be generated. (Explainer returned invalid data or error occurred during plotting.)")
        else:
            if pd.isna(base_prob):
                st.error('Prediction failed due to NaN/Inf values. Check input or training data consistency.')
            else:
                st.error('Model or SHAP Explainer not loaded. Please run train_model.py first.')

        # --- 4. Save Record and Update Context ---
        record = {
            'name': name, 'dob': str(dob), 'address': address, 'phone': phone, 'email': email,
            'loan_amnt': loan_amnt, 'int_rate': int_rate, 'fico_low': fico_low, 'annual_inc': annual_inc,
            'guarantor_name': guarantor_name, 'guarantor_relation': guarantor_relation, 'guarantor_phone': guarantor_phone, 'guarantor_address': guarantor_address,
            'insurance_opted': insurance_opted, 'insurance_premium': (insurance_rate/100.0)*loan_amnt,
            'documents_json': docs_to_json(doc_paths), 'assistant_notes': '\n'.join(notes),
            'base_prob': float(base_prob) if not pd.isna(base_prob) else 0.5,
            'adjusted_prob': float(adjusted_prob)
        }
        insert_applicant(record)
        st.success('Assessment saved.')
        
        # Save context for the Virtual Assistant
        st.session_state['current_context'] = {
            'risk_score': adjusted_prob,
            'top_features': top_features,
            'applicant_name': name
        }

with col2:
    # --- Virtual Assistant & Dashboard ---
    st.header('Assistant & Dashboard')
    # The assistant gets dynamic context after submission
    assistant_dialogue(st.session_state.get('current_context'))
    
    st.markdown('---')
    st.subheader('Recent Applicants Dashboard')
    df_history = fetch_all_applicants()
    if df_history is None or df_history.empty:
        st.write('No applicants yet.')
    else:
        df_display = df_history[['id','name','loan_amnt','fico_low','base_prob','adjusted_prob']].copy()
        df_display.rename(columns={'base_prob': 'ML Risk', 'adjusted_prob': 'Final Risk'}, inplace=True)
        df_display['ML Risk'] = (df_display['ML Risk'] * 100).round(2).astype(str) + '%'
        df_display['Final Risk'] = (df_display['Final Risk'] * 100).round(2).astype(str) + '%'
        st.dataframe(df_display.head(50))