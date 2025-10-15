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
    df = pd.read_csv('data/processed/olympics_merged.csv')
    
    # Add country_name if missing
    if 'country_name' not in df.columns:
        try:
            mapping_df = pd.read_csv('data/raw/country_iso3_mapping.csv')
            df = df.merge(mapping_df[['iso3', 'country_name']].drop_duplicates(), on='iso3', how='left')
        except FileNotFoundError:
            df['country_name'] = df['country']

    return df

# Initialize
try:
    preprocessor, classifier, regressor, optimal_threshold = load_models()
    df = load_data()
    
    # Prepare feature columns
    exclude_cols = ['iso3', 'country_name', 'year', 'gold', 'silver', 'bronze', 'total_medals', 'country', 'has_medal']
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
    selected_country = st.sidebar.selectbox(
        "Select Country", 
        countries, 
        index=countries.index('United States') if 'United States' in countries else 0
    )
    
    # Get country data
    country_df_all = df[df['country_name'] == selected_country]
    if country_df_all.empty:
        st.warning(f"No data available for {selected_country}")
        st.info("Please select a different country from the dropdown.")
        st.stop()
    
    iso3 = country_df_all.iloc[0]['iso3']
    
    # Year and games type selection
    min_year = int(df['year'].min())
    max_year_data = int(df['year'].max())
    
    # Games type selection FIRST (before year)
    games_type = st.sidebar.selectbox("Games Type", ["Summer", "Winter"])
    
    # Determine valid Olympic years based on games type
    if games_type == "Summer":
        # Summer Olympics: 1896, 1900, 1904, ..., every 4 years (except 1916, 1940, 1944)
        VALID_YEARS = [y for y in range(1896, 2100, 4) if y not in [1916, 1940, 1944]]
    else:
        # Winter Olympics: 1924, 1928, ..., every 4 years (except 1940, 1944)
        VALID_YEARS = [y for y in range(1924, 2100, 4) if y not in [1940, 1944]]
    
    # Find most recent valid Olympic year as default
    valid_years_in_data = [y for y in VALID_YEARS if y <= max_year_data]
    default_year = max(valid_years_in_data) if valid_years_in_data else VALID_YEARS[0]
    
    # Year input
    selected_year = st.sidebar.number_input(
        f"Select {games_type} Olympic Year",
        min_value=min_year,
        max_value=2100,
        value=default_year,
        step=4,  # Olympic cycles are 4 years
        help=f"Valid {games_type} Olympic years: {..., {VALID_YEARS[-5]}, {VALID_YEARS[-4]}, {VALID_YEARS[-3]}, {VALID_YEARS[-2]}, {VALID_YEARS[-1]}}"
    )
    
    # Validate selected year is an Olympic year
    if selected_year not in VALID_YEARS:
        st.sidebar.warning(f"⚠️ {selected_year} is not a valid {games_type} Olympic year")
        # Find nearest valid year
        nearest_year = min(VALID_YEARS, key=lambda y: abs(y - selected_year))
        st.sidebar.info(f"Nearest {games_type} Olympic year: {nearest_year}")
        if st.sidebar.button(f"Use {nearest_year} instead"):
            selected_year = nearest_year
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Check if we have actual data for selected year/games
    actual_data_row = country_df_all[
        (country_df_all['year'] == selected_year) & 
        (country_df_all['games_type'] == games_type)
    ]
    
    is_historical = not actual_data_row.empty
    
    # Find most recent historical data for features
    historical_for_features = country_df_all[
        (country_df_all['games_type'] == games_type) &
        (country_df_all['year'] <= selected_year)
    ].sort_values('year', ascending=False)
    
    if historical_for_features.empty:
        # No historical data for this games type, use any recent data
        historical_for_features = country_df_all.sort_values('year', ascending=False)
    
    if historical_for_features.empty:
        st.error(f"No data available for {selected_country}")
        st.stop()
    
    latest_features = historical_for_features.iloc[0]
    feature_year = int(latest_features['year'])
    
    # Display input data info
    st.sidebar.subheader(f"Input Features (from {feature_year})")
    if 'population' in latest_features:
        st.sidebar.write(f"Population: {latest_features['population']:,.0f}")
    if 'gdp_per_capita' in latest_features:
        st.sidebar.write(f"GDP per capita: ${latest_features['gdp_per_capita']:,.0f}")
    if 'gdp_current_usd' in latest_features:
        st.sidebar.write(f"GDP: ${latest_features['gdp_current_usd']:,.0f}")
    
    if feature_year != selected_year:
        st.sidebar.info(f"Using {feature_year} economic data to predict {selected_year}")
    
    # Prepare input for model
    input_data = latest_features[feature_cols].to_frame().T
    
    # Override games_type to match selection (critical!)
    if 'games_type' in input_data.columns:
        input_data['games_type'] = games_type
    
    # Make predictions
    try:
        X_input = preprocessor.transform(input_data)
        
        # Stage 1: Classification
        prob_has_medal = classifier.predict_proba(X_input)[0, 1]
        has_medal_pred = int(prob_has_medal >= optimal_threshold)
        
        # Stage 2: Regression
        if has_medal_pred:
            predicted_medals = max(0, regressor.predict(X_input)[0])
        else:
            predicted_medals = 0
        
        # Display main results
        st.markdown("---")
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
                help=f"Expected number of medals for {selected_year} {games_type} Olympics"
            )
        
        with col3:
            if is_historical:
                actual_medals = int(actual_data_row.iloc[0]['total_medals'])
                error = actual_medals - predicted_medals
                st.metric(
                    f"Actual Medals ({selected_year})",
                    f"{actual_medals}",
                    delta=f"{error:+.1f} medals",
                    delta_color="inverse",
                    help=f"Historical result from {selected_year} {games_type} Olympics"
                )
            else:
                st.metric(
                    "Prediction Year",
                    f"{selected_year}",
                    help=f"Future {games_type} Olympics (no actual data yet)"
                )
        
        # Show data type indicator
        if is_historical:
            st.success(f"📊 Validation Mode: Comparing prediction vs actual results for {selected_year}")
        else:
            st.info(f"🔮 Prediction Mode: Forecasting {selected_year} {games_type} Olympics")
        
        # Historical performance
        st.markdown("---")
        st.subheader(f"Historical Performance: {selected_country}")
        
        country_history = country_df_all.sort_values('year')
        
        if not country_history.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Medal Count by Year**")
                history_display = country_history[country_history['total_medals'].notna()].copy()
                
                # Add prediction row if future year
                if not is_historical:
                    future_row = pd.DataFrame({
                        'year': [selected_year],
                        'games_type': [games_type],
                        'gold': ['?'],
                        'silver': ['?'],
                        'bronze': ['?'],
                        'total_medals': [f"{predicted_medals:.1f} (pred)"]
                    })
                    history_display = pd.concat([history_display, future_row], ignore_index=True)
                
                history_display = history_display.sort_values('year', ascending=False)
                st.dataframe(
                    history_display[['year', 'games_type', 'gold', 'silver', 'bronze', 'total_medals']].head(20),
                    use_container_width=True,
                    height=300
                )
            
            with col2:
                st.write("**Medal Trend**")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                summer_data = country_history[country_history['games_type'] == 'Summer']
                winter_data = country_history[country_history['games_type'] == 'Winter']
                
                if not summer_data.empty:
                    ax.plot(summer_data['year'], summer_data['total_medals'], 
                           marker='o', label='Summer', linewidth=2, alpha=0.7)
                if not winter_data.empty:
                    ax.plot(winter_data['year'], winter_data['total_medals'], 
                           marker='s', label='Winter', linewidth=2, alpha=0.7)
                
                # Add prediction point
                marker_style = 'r*' if not is_historical else 'g*'
                label_text = f'Prediction ({selected_year})' if not is_historical else f'Actual ({selected_year})'
                ax.plot(selected_year, predicted_medals if not is_historical else actual_medals, 
                       marker_style, markersize=15, label=label_text, zorder=10)
                
                ax.set_xlabel('Year')
                ax.set_ylabel('Total Medals')
                ax.set_title('Medal Count Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        
        # SHAP Explanation
        st.markdown("---")
        st.subheader("💡 Feature Importance for This Prediction")
        
        with st.expander("What influences this prediction?", expanded=True):
            st.write("""
            SHAP (SHapley Additive exPlanations) values show how each feature contributes 
            to the prediction. Red indicates features pushing the prediction higher, 
            blue indicates features pushing it lower.
            """)
            
            try:
                feature_names = preprocessor.get_feature_names_out()
                
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_input)
                
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values_to_plot = shap_values[1]
                    expected_value = explainer.expected_value[1]
                else:
                    shap_values_to_plot = shap_values
                    expected_value = explainer.expected_value
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots._waterfall.waterfall_legacy(
                    expected_value,
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
                
                st.write("**Top 3 Most Influential Features:**")
                for i, row in shap_importance.head(3).iterrows():
                    direction = "increases" if row['shap_value'] > 0 else "decreases"
                    feature_display = row['feature'].split('__')[-1]
                    st.write(f"• **{feature_display}**: {direction} medal probability by {abs(row['shap_value']):.3f}")
            
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")
        
        # Comparison with similar countries
        st.markdown("---")
        st.subheader("👥 Comparison with Similar Countries")
        
        with st.expander("See similar performing countries"):
            if 'gdp_per_capita' in df.columns and pd.notna(latest_features.get('gdp_per_capita')):
                target_gdp = latest_features['gdp_per_capita']
                
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
        import traceback
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This prediction model uses a two-stage approach:
1. **Classification**: Predicts whether a country will win any medals
2. **Regression**: Predicts the medal count for predicted winners

The model is trained on historical Olympic data combined with World Development Indicators 
from the World Bank (GDP, population, life expectancy, etc.).
""")
