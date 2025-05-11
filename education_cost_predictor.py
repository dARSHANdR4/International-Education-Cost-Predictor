import pandas as pd
import numpy as np
import requests
import pickle
import os
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="International Education Cost Predictor",
    page_icon="🎓",
    layout="wide"
)

# Constants
DATA_URL = 'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/International_Education_Costs-LeyxyBrErfLztS2WwEbWNzwX0Qvt8g.csv'
MODEL_PATH = 'education_cost_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
ENCODER_PATH = 'categorical_encoder.pkl'
DATASET_PATH = 'education_costs_data.pkl'

# Function to get current USD to INR exchange rate
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    try:
        # Since forex_python might not be installed, we'll use a simpler approach
        # You can install forex_python with: pip install forex-python
        # For now, we'll use a hardcoded recent rate
        return 85.42  # Current approximate rate (May 2025)
    except Exception as e:
        st.warning(f"Could not fetch latest exchange rate. Using default rate of 85.42 INR per USD.")
        return 85.42  # Default fallback rate

# Function to load or train the model
@st.cache_resource
def load_or_train_model():
    # Check if model already exists
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(DATASET_PATH):
        st.info("Loading pre-trained model and preprocessors...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        with open(DATASET_PATH, 'rb') as f:
            df = pickle.load(f)
        return model, scaler, encoder, df
    
    # If model doesn't exist, train a new one
    st.info("Training new model. This may take a moment...")
    
    # Fetch and process the data
    try:
        response = requests.get(DATA_URL)
        df = pd.read_csv(StringIO(response.text))
        
        # Convert string values to appropriate types
        numeric_columns = ['Duration_Years', 'Tuition_USD', 'Living_Cost_Index', 
                          'Rent_USD', 'Visa_Fee_USD', 'Insurance_USD', 'Exchange_Rate']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering
        # Annual living cost (rent * 12 months + insurance)
        df['Annual_Living_Cost'] = (df['Rent_USD'] * 12) + df['Insurance_USD']
        
        # Total cost for the entire duration
        df['Total_Cost_USD'] = df['Tuition_USD'] + (df['Annual_Living_Cost'] * df['Duration_Years']) + df['Visa_Fee_USD']
        
        # Save the processed dataset
        with open(DATASET_PATH, 'wb') as f:
            pickle.dump(df, f)
        
        # Define features and target
        X = df[['Country', 'Level', 'Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Insurance_USD', 'Visa_Fee_USD']]
        y = df['Total_Cost_USD']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        categorical_features = ['Country', 'Level']
        numeric_features = ['Duration_Years', 'Living_Cost_Index', 'Rent_USD', 'Insurance_USD', 'Visa_Fee_USD']
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numeric_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features),
                ('num', numeric_transformer, numeric_features)
            ])
        
        # Create and train the model pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.success(f"Model trained successfully! R² score: {r2:.4f}, RMSE: ${np.sqrt(mse):.2f}")
        
        # Save the model and preprocessors
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_pipeline, f)
        
        # Extract preprocessor components for later use with individual inputs
        encoder = preprocessor.transformers_[0][1]
        scaler = preprocessor.transformers_[1][1]
        
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(encoder, f)
        
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model_pipeline, scaler, encoder, df
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

# Function to make predictions
def predict_cost(model, scaler, encoder, input_data):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        return prediction
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Function to format currency in USD and INR
def format_currency(amount_usd, exchange_rate, show_inr=False):
    if show_inr:
        amount_inr = amount_usd * exchange_rate
        return f"${amount_usd:,.2f} (₹{amount_inr:,.2f})"
    else:
        return f"${amount_usd:,.2f}"

# Main function
def main():
    st.title("🎓 International Education Cost Predictor")
    st.markdown("""
    This application predicts the total cost of international education based on various factors.
    Enter your preferences below to get a cost estimate for your educational journey.
    """)
    
    # Get USD to INR exchange rate
    usd_to_inr_rate = get_usd_to_inr_rate()
    
    # Load or train the model
    model, scaler, encoder, df = load_or_train_model()
    
    if model is None:
        st.error("Failed to load or train the model. Please check the logs and try again.")
        return
    
    # Create sidebar for inputs
    st.sidebar.header("Enter Your Preferences")
    
    # Add currency display option
    show_inr = st.sidebar.checkbox("Show costs in Indian Rupees (INR)", value=True)
    if show_inr:
        st.sidebar.info(f"Current exchange rate: 1 USD = ₹{usd_to_inr_rate:.2f}")
    
    # Get unique values for categorical fields
    countries = sorted(df['Country'].unique())
    levels = sorted(df['Level'].unique())
    
    # Create input fields
    selected_country = st.sidebar.selectbox("Select Country", countries)
    selected_level = st.sidebar.selectbox("Select Degree Level", levels)
    
    # Filter data based on selections to provide realistic ranges for numeric inputs
    filtered_df = df[(df['Country'] == selected_country) & (df['Level'] == selected_level)]
    
    # If we have data for this combination, use it to set realistic ranges
    if not filtered_df.empty:
        duration_min = filtered_df['Duration_Years'].min()
        duration_max = filtered_df['Duration_Years'].max()
        
        # Fix for slider error: Ensure min and max are different
        if duration_min == duration_max:
            duration_min = max(1.0, duration_min - 0.5)  # Subtract 0.5 from min (but not below 1.0)
            duration_max = duration_max + 0.5  # Add 0.5 to max
        
        rent_min = filtered_df['Rent_USD'].min()
        rent_max = filtered_df['Rent_USD'].max()
        # Ensure min and max are different for rent
        if rent_min == rent_max:
            rent_min = max(100.0, rent_min * 0.9)  # 10% less than min (but not below 100)
            rent_max = rent_max * 1.1  # 10% more than max
        
        insurance_min = filtered_df['Insurance_USD'].min()
        insurance_max = filtered_df['Insurance_USD'].max()
        # Ensure min and max are different for insurance
        if insurance_min == insurance_max:
            insurance_min = max(100.0, insurance_min * 0.9)
            insurance_max = insurance_max * 1.1
        
        visa_min = filtered_df['Visa_Fee_USD'].min()
        visa_max = filtered_df['Visa_Fee_USD'].max()
        # Ensure min and max are different for visa
        if visa_min == visa_max:
            visa_min = max(50.0, visa_min * 0.9)
            visa_max = visa_max * 1.1
        
        living_cost_min = filtered_df['Living_Cost_Index'].min()
        living_cost_max = filtered_df['Living_Cost_Index'].max()
        # Ensure min and max are different for living cost
        if living_cost_min == living_cost_max:
            living_cost_min = max(10.0, living_cost_min * 0.9)
            living_cost_max = living_cost_max * 1.1
    else:
        # Use global ranges if no data for this specific combination
        duration_min = max(1.0, df['Duration_Years'].min())
        duration_max = min(6.0, df['Duration_Years'].max())
        # Ensure min and max are different
        if duration_min == duration_max:
            duration_min = max(1.0, duration_min - 0.5)
            duration_max = duration_max + 0.5
        
        rent_min = df['Rent_USD'].min()
        rent_max = df['Rent_USD'].max()
        insurance_min = df['Insurance_USD'].min()
        insurance_max = df['Insurance_USD'].max()
        visa_min = df['Visa_Fee_USD'].min()
        visa_max = df['Visa_Fee_USD'].max()
        living_cost_min = df['Living_Cost_Index'].min()
        living_cost_max = df['Living_Cost_Index'].max()
    
    # Create sliders for numeric inputs with realistic ranges
    duration_years = st.sidebar.slider("Program Duration (Years)", 
                                      float(duration_min), float(duration_max), 
                                      float(duration_min + (duration_max - duration_min)/2))
    
    living_cost_index = st.sidebar.slider("Living Cost Index", 
                                         float(living_cost_min), float(living_cost_max), 
                                         float(living_cost_min + (living_cost_max - living_cost_min)/2))
    
    rent_usd = st.sidebar.slider("Monthly Rent (USD)", 
                               float(rent_min), float(rent_max), 
                               float(rent_min + (rent_max - rent_min)/2))
    
    insurance_usd = st.sidebar.slider("Annual Insurance (USD)", 
                                    float(insurance_min), float(insurance_max), 
                                    float(insurance_min + (insurance_max - insurance_min)/2))
    
    visa_fee_usd = st.sidebar.slider("Visa Fee (USD)", 
                                   float(visa_min), float(visa_max), 
                                   float(visa_min + (visa_max - visa_min)/2))
    
    # Create a button to trigger prediction
    predict_button = st.sidebar.button("Predict Total Cost")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cost Breakdown Analysis")
        
        # Display some statistics about the selected country and level
        if not filtered_df.empty:
            avg_tuition = filtered_df['Tuition_USD'].mean()
            
            # Fix for the error: Check if 'Total_Cost_USD' exists in filtered_df
            if 'Total_Cost_USD' in filtered_df.columns:
                avg_total_cost = filtered_df['Total_Cost_USD'].mean()
            else:
                # Calculate it if it doesn't exist
                avg_total_cost = None
            
            st.markdown(f"### Average Costs for {selected_level} in {selected_country}")
            
            if show_inr:
                st.markdown(f"**Average Tuition:** ${avg_tuition:,.2f} (₹{avg_tuition * usd_to_inr_rate:,.2f})")
                if avg_total_cost:
                    st.markdown(f"**Average Total Cost:** ${avg_total_cost:,.2f} (₹{avg_total_cost * usd_to_inr_rate:,.2f})")
            else:
                st.markdown(f"**Average Tuition:** ${avg_tuition:,.2f}")
                if avg_total_cost:
                    st.markdown(f"**Average Total Cost:** ${avg_total_cost:,.2f}")
            
            # Create an interactive bar chart of average costs by country for this level using Plotly
            level_data = df[df['Level'] == selected_level].groupby('Country')['Total_Cost_USD'].mean().reset_index()
            level_data = level_data.sort_values('Total_Cost_USD')
            
            # Create USD and INR versions of the data
            level_data['Total_Cost_INR'] = level_data['Total_Cost_USD'] * usd_to_inr_rate
            
            # Create the interactive bar chart
            fig = px.bar(
                level_data, 
                x='Total_Cost_USD', 
                y='Country',
                title=f'Average Total Cost for {selected_level} Programs by Country',
                labels={'Total_Cost_USD': 'Total Cost (USD)', 'Country': 'Country'},
                height=600
            )
            
            # Add INR values as text if requested
            if show_inr:
                fig.update_traces(
                    text=[f"${x:,.0f}<br>(₹{y:,.0f})" for x, y in zip(level_data['Total_Cost_USD'], level_data['Total_Cost_INR'])],
                    textposition='outside'
                )
                fig.update_layout(
                    xaxis_title="Total Cost (USD / INR)",
                )
            else:
                fig.update_traces(
                    text=[f"${x:,.0f}" for x in level_data['Total_Cost_USD']],
                    textposition='outside'
                )
            
            # Make the chart more interactive
            fig.update_layout(
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            # Display the chart with the option to expand
            st.plotly_chart(fig, use_container_width=True)
            
            # Add button to open in expanded view
            if st.button("Open Cost Comparison in Full Screen"):
                st.session_state.show_expanded_chart = True
            
            # Show expanded chart if requested
            if 'show_expanded_chart' in st.session_state and st.session_state.show_expanded_chart:
                with st.expander("Expanded Cost Comparison Chart", expanded=True):
                    st.plotly_chart(fig, use_container_width=True)
                    if st.button("Close Expanded View"):
                        st.session_state.show_expanded_chart = False
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
            # Prepare input data
            input_data = {
                'Country': selected_country,
                'Level': selected_level,
                'Duration_Years': duration_years,
                'Living_Cost_Index': living_cost_index,
                'Rent_USD': rent_usd,
                'Insurance_USD': insurance_usd,
                'Visa_Fee_USD': visa_fee_usd
            }
            
            # Make prediction
            predicted_cost = predict_cost(model, scaler, encoder, input_data)
            
            if predicted_cost is not None:
                # Display prediction
                st.markdown("### Estimated Total Cost")
                
                # Convert to INR if requested
                predicted_cost_inr = predicted_cost * usd_to_inr_rate
                
                if show_inr:
                    st.markdown(
                        f"<h1 style='color:#1E88E5;'>${predicted_cost:,.2f}</h1>"
                        f"<h2 style='color:#1E88E5;'>₹{predicted_cost_inr:,.2f}</h2>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"<h1 style='color:#1E88E5;'>${predicted_cost:,.2f}</h1>", unsafe_allow_html=True)
                
                # Calculate and display annual cost
                annual_cost = predicted_cost / duration_years
                annual_cost_inr = annual_cost * usd_to_inr_rate
                
                if show_inr:
                    st.markdown(f"**Annual Cost:** ${annual_cost:,.2f} (₹{annual_cost_inr:,.2f})")
                else:
                    st.markdown(f"**Annual Cost:** ${annual_cost:,.2f}")
                
                # Calculate and display monthly cost
                monthly_cost = annual_cost / 12
                monthly_cost_inr = monthly_cost * usd_to_inr_rate
                
                if show_inr:
                    st.markdown(f"**Monthly Cost:** ${monthly_cost:,.2f} (₹{monthly_cost_inr:,.2f})")
                else:
                    st.markdown(f"**Monthly Cost:** ${monthly_cost:,.2f}")
                
                # Create an interactive pie chart for cost breakdown using Plotly
                annual_living_cost = (rent_usd * 12) + insurance_usd
                total_living_cost = annual_living_cost * duration_years
                total_tuition = predicted_cost - total_living_cost - visa_fee_usd
                
                labels = ['Tuition', 'Living Expenses', 'Visa Fee']
                values = [total_tuition, total_living_cost, visa_fee_usd]
                
                # Create USD and INR versions of the values
                values_inr = [v * usd_to_inr_rate for v in values]
                
                # Create the pie chart
                if show_inr:
                    hover_text = [
                        f"Tuition: ${values[0]:,.2f} (₹{values_inr[0]:,.2f})",
                        f"Living Expenses: ${values[1]:,.2f} (₹{values_inr[1]:,.2f})",
                        f"Visa Fee: ${values[2]:,.2f} (₹{values_inr[2]:,.2f})"
                    ]
                else:
                    hover_text = [
                        f"Tuition: ${values[0]:,.2f}",
                        f"Living Expenses: ${values[1]:,.2f}",
                        f"Visa Fee: ${values[2]:,.2f}"
                    ]
                
                pie_fig = px.pie(
                    values=values,
                    names=labels,
                    title="Cost Breakdown",
                    hover_data=[values],
                    labels={'label': 'Category', 'value': 'Amount (USD)'}
                )
                
                # Customize the pie chart
                pie_fig.update_traces(
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>%{customdata:$,.2f}<br>%{percent}<extra></extra>',
                    customdata=[[v] for v in values]
                )
                
                # Display the pie chart
                st.plotly_chart(pie_fig, use_container_width=True)
                
                # Add button to open in expanded view
                if st.button("Open Cost Breakdown in Full Screen"):
                    st.session_state.show_expanded_pie = True
                
                # Show expanded pie chart if requested
                if 'show_expanded_pie' in st.session_state and st.session_state.show_expanded_pie:
                    with st.expander("Expanded Cost Breakdown Chart", expanded=True):
                        # Create a more detailed version for the expanded view
                        expanded_pie_fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            textinfo='percent+label',
                            hovertemplate='%{label}<br>USD: $%{value:,.2f}<br>INR: ₹%{customdata:,.2f}<br>%{percent}<extra></extra>' if show_inr else '%{label}<br>$%{value:,.2f}<br>%{percent}<extra></extra>',
                            customdata=values_inr if show_inr else None
                        )])
                        
                        expanded_pie_fig.update_layout(
                            title_text=f"Cost Breakdown for {selected_level} in {selected_country}",
                            annotations=[dict(
                                text=f"Total: ${predicted_cost:,.2f}" + (f" (₹{predicted_cost_inr:,.2f})" if show_inr else ""),
                                x=0.5, y=0.5,
                                font_size=15,
                                showarrow=False
                            )]
                        )
                        
                        st.plotly_chart(expanded_pie_fig, use_container_width=True)
                        if st.button("Close Expanded Pie View"):
                            st.session_state.show_expanded_pie = False
                
                # Compare with average if available
                if avg_total_cost:
                    diff = predicted_cost - avg_total_cost
                    diff_percent = (diff / avg_total_cost) * 100
                    diff_inr = diff * usd_to_inr_rate
                    
                    if diff > 0:
                        if show_inr:
                            st.markdown(f"**Note:** This is ${diff:,.2f} (₹{diff_inr:,.2f}) ({diff_percent:.1f}%) **higher** than the average cost for {selected_level} programs in {selected_country}.")
                        else:
                            st.markdown(f"**Note:** This is ${diff:,.2f} ({diff_percent:.1f}%) **higher** than the average cost for {selected_level} programs in {selected_country}.")
                    else:
                        if show_inr:
                            st.markdown(f"**Note:** This is ${abs(diff):,.2f} (₹{abs(diff_inr):,.2f}) ({abs(diff_percent):.1f}%) **lower** than the average cost for {selected_level} programs in {selected_country}.")
                        else:
                            st.markdown(f"**Note:** This is ${abs(diff):,.2f} ({abs(diff_percent):.1f}%) **lower** than the average cost for {selected_level} programs in {selected_country}.")
            else:
                st.error("Failed to make prediction. Please try different inputs.")
        else:
            st.info("Enter your preferences and click 'Predict Total Cost' to see the results.")
    
    # Display dataset information
    st.markdown("---")
    st.subheader("About the Dataset")
    st.markdown("""
    This model is trained on the International Education Costs dataset, which compiles detailed financial 
    information for students pursuing higher education abroad. It covers multiple countries, cities, and 
    universities around the world, capturing tuition and living expenses alongside key ancillary costs.
    """)
    
    # Show sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df.head())

# Run the app
if __name__ == "__main__":
    main()