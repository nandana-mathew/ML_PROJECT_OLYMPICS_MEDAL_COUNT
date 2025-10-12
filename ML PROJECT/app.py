import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Olympic Medal Predictor",
    page_icon="🏅",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    preprocessor = joblib.load('models/preprocessor.joblib')
    classifier = joblib.load('models/classifier.joblib')
    regressor = joblib.load('models/regressor.joblib')
    optimal_threshold = joblib.load('models/optimal_threshold.joblib')
    return preprocessor, classifier, regressor, optimal_threshold

@st.cache_data
def load_data():
    # Ensure 'olympics_merged.csv' has 'country_name' column or join it here
    df = pd.read_csv('data/processed/olympics_merged.csv')
    
    # --- CRITICAL CORRECTION: Add country_name back to main dataframe ---
    # Since 'olympics_merged.csv' only has 'country' (World Bank name) and 'iso3' 
    # but the mapping was done in download_data.ipynb, we must ensure 'country_name' (Olympic name) is present.
    # Assuming 'country' in olympics_merged.csv is the World Bank name, 
    # we need to reload the mapping file or ensure 'country_name' is saved in the merged CSV.
    
    # Check if country_name exists; if not, attempt to load mapping or use 'country'
    if 'country_name' not in df.columns:
        try:
            mapping_df = pd.read_csv('data/raw/country_iso3_mapping.csv')
            df = df.merge(mapping_df[['iso3', 'country_name']].drop_duplicates(), on='iso3', how='left')
        except FileNotFoundError:
            # Fallback if mapping not found, use World Bank's 'country' column
            df['country_name'] = df['country']

    return df

def find_most_recent_olympic_year(country_df, games_type, selected_year):
    """Finds the most recent Olympic year (<= selected_year) for the specific games type with data."""
    
    # Filter by games type
    filtered_df = country_df[country_df['games_type'] == games_type].sort_values('year', ascending=False)
    
    # Find the most recent year in the data that is less than or equal to the selected prediction year
    recent_historical = filtered_df[filtered_df['year'] <= selected_year].head(1)
    
    if not recent_historical.empty:
        # Return the data from the most recent historical Olympics
        return recent_historical.iloc[0], int(recent_historical.iloc[0]['year'])
    
    # Find the latest available data overall if no specific Olympic year found
    latest_data = country_df.sort_values('year', ascending=False).head(1)
    if not latest_data.empty:
        return latest_data.iloc[0], int(latest_data.iloc[0]['year'])

    return None, None

# Initialize
try:
    preprocessor, classifier, regressor, optimal_threshold = load_models()
    df = load_data()
    
    # Prepare feature columns
    exclude_cols = ['iso3', 'country_name', 'year', 'gold', 'silver', 'bronze', 'total_medals', 'country', 'has_medal'] # Exclude 'has_medal' too
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Please run train_two_stage_model.ipynb first to generate the models.")
    models_loaded = False

# App title
st.title("🏅 Olympic Medal Prediction")
st.markdown("Predict medal counts using World Development Indicators")

if models_loaded:
    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Country selection
    countries = sorted(df['country_name'].dropna().unique())
    selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index('United States') if 'United States' in countries else 0)
    
    # Get country's ISO3 code
    country_df_all = df[df['country_name'] == selected_country]
    if country_df_all.empty:
        st.warning(f"No data available for {selected_country}")
        st.info("Please select a different country from the dropdown.")
        models_loaded = False # Stop further processing if country data is missing
    else:
        iso3 = country_df_all.iloc[0]['iso3']

    # Year selection
    min_year = int(df['year'].min())
    max_year_data = int(df['year'].max())
    
    # Default to the most recent Olympic year (2024 for Summer, 2022 for Winter, or max_year_data)
    default_year = max(2024 if 'Summer' in df['games_type'].unique() else 2022, max_year_data)

    selected_year = st.sidebar.number_input(
        "Select Prediction Year",
        min_value=min_year,
        max_value=max_year_data + 10,  # Allow future predictions
        value=default_year,
        step=1
    )
    
    # Games type
    games_type = st.sidebar.selectbox("Games Type", ["Summer", "Winter"], 
                                      index=0 if selected_year % 4 == 0 else 1)
    
    st.markdown("---")

    # --- Use the helper function to get the latest data point for prediction ---
    latest_data_point, latest_year = find_most_recent_olympic_year(country_df_all, games_type, selected_year)

    if latest_data_point is not None:
        
        # --- Display Input Data (from the most recent historical point) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader(f"Input Data (from {latest_year})")
        if 'population' in latest_data_point:
            st.sidebar.write(f"Population: {latest_data_point['population']:,.0f}")
        if 'gdp_per_capita' in latest_data_point:
            st.sidebar.write(f"GDP per capita: ${latest_data_point['gdp_per_capita']:,.0f}")
        
        # Prepare input data for the model (using the latest available indicators)
        input_data = latest_data_point[feature_cols].to_frame().T
        
        # --- CRITICAL: Override year and games_type based on selection ---
        # The prediction should be for the selected_year and games_type
        if 'games_type' in input_data.columns:
            input_data['games_type'] = games_type
        if 'year' in input_data.columns:
            # While 'year' is often dropped, it might be in the feature set or needed for checks
            input_data['year'] = selected_year 

        
        # Make prediction
        try:
            X_input = preprocessor.transform(input_data)
            
            # Stage 1: Classification
            prob_has_medal = classifier.predict_proba(X_input)[0, 1]
            has_medal_pred = int(prob_has_medal >= optimal_threshold)
            
            # Stage 2: Regression
            if has_medal_pred:
                # The regressor was trained on log-transformed data in the original notebook 
                # (using Poisson objective), so a simple predict should work, but ensure
                # the features passed match the training features (X_input).
                predicted_medals = max(0, regressor.predict(X_input)[0])
            else:
                predicted_medals = 0
            
            # --- FIND ACTUAL MEDALS FOR COMPARISON ---
            
            # Find the actual data for the selected prediction year and games type
            actual_data_row = country_df_all[
                (country_df_all['year'] == selected_year) & 
                (country_df_all['games_type'] == games_type)
            ]
            
            is_historical = not actual_data_row.empty
            actual_medals_value = 0
            
            if is_historical:
                actual_medals_value = actual_data_row.iloc[0]['total_medals']
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probability of Winning Medals",
                    f"{prob_has_medal:.1%}",
                    help="Predicted probability of winning at least one medal"
                )
            
            with col2:
                st.metric(
                    "Predicted Medal Count",
                    f"{predicted_medals:.1f}",
                    help=f"Expected number of medals for {selected_year} {games_type} Games"
                )
            
            with col3:
                if is_historical and actual_medals_value >= 0:
                    st.metric(
                        f"Actual Medals ({selected_year})",
                        f"{actual_medals_value:.0f}",
                        help=f"Historical medal count for the {selected_year} {games_type} Games"
                    )
                else:
                    st.metric(
                        f"Prediction Year",
                        f"{selected_year}",
                        help="The prediction is for a future or non-Olympic year with no known actual result in the dataset."
                    )
            
            # Historical performance
            st.markdown("---")
            st.subheader(f"Historical Performance: {selected_country}")
            
            country_history = country_df_all.sort_values('year')
            
            if not country_history.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Medal Count by Year**")
                    # Filter to only show actual Olympic years (where medals >= 0)
                    history_display = country_history[country_history['total_medals'].notna()].copy()
                    
                    # Merge prediction into history if it's a future year
                    if not is_historical:
                        future_row = pd.DataFrame({
                            'year': [selected_year],
                            'games_type': [games_type],
                            'total_medals': [predicted_medals],
                            'gold': [np.nan], 'silver': [np.nan], 'bronze': [np.nan] # Use NaN for missing breakdown
                        })
                        history_display = pd.concat([history_display, future_row], ignore_index=True)
                        history_display['year'] = history_display['year'].astype(int)
                        history_display = history_display.sort_values('year', ascending=False)
                        
                    
                    st.dataframe(history_display[['year', 'games_type', 'gold', 'silver', 'bronze', 'total_medals']], 
                                 use_container_width=True, height=300)
                
                with col2:
                    st.write("**Medal Trend**")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    summer_data = country_history[country_history['games_type'] == 'Summer']
                    winter_data = country_history[country_history['games_type'] == 'Winter']
                    
                    if not summer_data.empty:
                        ax.plot(summer_data['year'], summer_data['total_medals'], 
                               marker='o', label='Summer', linewidth=2)
                    if not winter_data.empty:
                        ax.plot(winter_data['year'], winter_data['total_medals'], 
                               marker='s', label='Winter', linewidth=2)
                    
                    # Add prediction point to plot
                    ax.plot(selected_year, predicted_medals, 'r*', markersize=10, 
                            label=f'Prediction {selected_year}', zorder=10)

                    
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Total Medals')
                    ax.set_title(f'Medal Count Over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig) # Close figure to free memory
            
            # SHAP Explanation
            st.markdown("---")
            st.subheader("💡 Feature Importance for This Prediction")
            
            with st.expander("What influences this prediction?", expanded=True):
                st.write("""
                SHAP (SHapley Additive exPlanations) values show how each feature contributes 
                to the prediction. Red indicates features pushing the prediction higher, 
                blue indicates features pushing it lower. (Uses the Classification model for explanation).
                """)
                
                try:
                    # --- CRITICAL: Get Feature Names Correctly ---
                    # The get_feature_names_out() method handles both numeric and one-hot encoded categories
                    feature_names = preprocessor.get_feature_names_out()
                    
                    # Compute SHAP values
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(X_input)
                    
                    # For binary classification, take the positive class (index 1)
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]
                    else:
                         shap_values_to_plot = shap_values

                    # Create SHAP waterfall plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots._waterfall.waterfall_legacy(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        shap_values_to_plot[0],
                        feature_names=feature_names,
                        max_display=10,
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                    
                    # Top 3 features
                    shap_importance = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_values_to_plot[0],
                        'abs_shap': np.abs(shap_values_to_plot[0])
                    }).sort_values('abs_shap', ascending=False)
                    
                    st.write("**Top 3 Most Influential Features (for likelihood of winning a medal):**")
                    for i, row in shap_importance.head(3).iterrows():
                        direction = "increases" if row['shap_value'] > 0 else "decreases"
                        st.write(f"• **{row['feature'].split('__')[-1]}**: {direction} prediction probability by {abs(row['shap_value']):.3f}")
                
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {e}")
            
            # Comparison with similar countries
            st.markdown("---")
            st.subheader("👥 Comparison with Similar Countries")
            
            with st.expander("See similar performing countries"):
                # Find countries with similar GDP per capita
                if 'gdp_per_capita' in df.columns and pd.notna(latest_data_point.get('gdp_per_capita')):
                    target_gdp = latest_data_point['gdp_per_capita']
                    
                    # Get latest data for each country
                    latest_by_country = df.sort_values('year', ascending=False).groupby('country_name').first().reset_index()
                    latest_by_country['gdp_diff'] = abs(latest_by_country['gdp_per_capita'] - target_gdp)
                    
                    similar_countries = latest_by_country.nsmallest(6, 'gdp_diff')
                    similar_countries = similar_countries[similar_countries['country_name'] != selected_country]
                    
                    comparison_df = similar_countries[['country_name', 'gdp_per_capita', 'population', 'total_medals']].head(5)
                    comparison_df.columns = ['Country', 'GDP per Capita', 'Population', 'Last Medal Count']
                    
                    st.dataframe(comparison_df.set_index('Country'), use_container_width=True)
                else:
                    st.info("GDP per capita data not available for comparison")
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check that all required data is available.")
    
    else:
        st.warning(f"No historical data available to base the prediction for {selected_country} in {selected_year} for {games_type} Games.")
        st.info("Please select a different country or prediction year.")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This prediction model uses a two-stage approach:
1. **Classification**: Predicts whether a country will win any medals
2. **Regression**: Predicts the medal count for predicted winners

The model is trained on historical Olympic data combined with World Development Indicators 
from the World Bank (GDP, population, life expectancy, etc.).
""")