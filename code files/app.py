import streamlit as st
import pandas as pd
import numpy as np
from app_inference_utils import predict_adverse_reactions
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="ADR Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load data for dropdowns
@st.cache_data
def load_data():
    """Load data from sample_data.csv for dropdowns with chunked reading for speed"""
    try:
        # Use chunked reading for better performance
        chunk_size = 10000
        chunks = []
        for chunk in pd.read_csv('data/sample_data.csv', chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        
        # Get unique drugs and their corresponding indications
        unique_drugs = sorted(df['drug_name'].unique())
        drug_indication_map = {}
        drug_indication_side_effects_map = {}
        
        for drug in unique_drugs:
            drug_data = df[df['drug_name'] == drug]
            drug_indications = drug_data['meddra_indication_name'].unique()
            drug_indication_map[drug] = sorted(drug_indications)
            
            # For each drug-indication pair, get the side effects
            drug_indication_side_effects_map[drug] = {}
            for indication in drug_indications:
                indication_data = drug_data[drug_data['meddra_indication_name'] == indication]
                side_effects_for_pair = indication_data['side_effect_name'].unique()
                drug_indication_side_effects_map[drug][indication] = sorted(side_effects_for_pair)
        
        return df, drug_indication_map, unique_drugs, drug_indication_side_effects_map
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def main():
    # Header
    st.title("🏥 Adverse Drug Reaction Prediction System")
    st.markdown("AI-powered prediction of potential adverse reactions for drug-indication combinations")
    
    # Demonstration notice
    st.info("""
    📊 **Demonstration Mode**: Currently using a subset of the original data with ~50k rows for faster performance. 
    The full dataset contains **9.3 million rows**. If your system has sufficient capacity, you can change the dataset 
    in the code from `sample_data.csv` to `merged_data.csv` on GitHub for access to the complete dataset.
    """)
    
    # Back to home button (only show if we're in prediction mode)
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    
    # Load data
    with st.spinner("Loading data..."):
        df, drug_indication_map, unique_drugs, drug_indication_side_effects_map = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar for input
    st.sidebar.header("🔍 Input Parameters")
    
    # Drug selection
    selected_drug = st.sidebar.selectbox(
        "Select Drug:",
        options=unique_drugs,
        help="Choose the drug you want to analyze"
    )
    
    # Display number of indications for selected drug
    if selected_drug:
        drug_indications = drug_indication_map.get(selected_drug, [])
        st.sidebar.info(f"**{selected_drug}** has **{len(drug_indications)}** available indications")
        
        # Indication selection
        if drug_indications:
            selected_indication = st.sidebar.selectbox(
                "Select Medical Indication:",
                options=drug_indications,
                help="Choose the medical condition the drug is used for"
            )
        else:
            st.sidebar.warning(f"No indications found for {selected_drug}")
            selected_indication = None
    else:
        selected_indication = None
    
    # Display number of available side effects for the drug-indication pair
    if selected_drug and selected_indication:
        # Get side effects ONLY for the specific drug-indication pair
        available_side_effects = drug_indication_side_effects_map.get(selected_drug, {}).get(selected_indication, [])
        
        st.sidebar.info(f"**{len(available_side_effects)}** side effects available for prediction")
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict Adverse Reactions", type="primary")
    
    # Main content area
    if predict_button:
        if not selected_drug or not selected_indication:
            st.warning("Please select both a drug and an indication.")
            return
        
        # Show selected parameters
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Selected Drug:** {selected_drug}")
        with col2:
            st.info(f"**Selected Indication:** {selected_indication}")
        
        # Run prediction
        with st.spinner("Running prediction... This may take a moment."):
            try:
                # Get side effects ONLY for the specific drug-indication pair
                side_effects_list = drug_indication_side_effects_map.get(selected_drug, {}).get(selected_indication, [])
                
                st.info(f"Predicting for {len(side_effects_list)} side effects for {selected_drug} + {selected_indication}.")
                
                if not side_effects_list:
                    st.error(f"No side effects found for {selected_drug} + {selected_indication}.")
                    return
                
                # Run prediction for ALL side effects
                predictions = predict_adverse_reactions(
                    selected_drug, 
                    selected_indication, 
                    side_effects_list
                )
                
                if not predictions:
                    st.error("No predictions could be generated. Please try different inputs.")
                    return
                
                # Display results
                st.success(f"✅ Generated {len(predictions)} predictions!")
                
                # Back to home button
                if st.button("🏠 Back to Home - Select Another Drug", type="secondary"):
                    st.session_state.predictions_made = False
                    st.rerun()
                
                # Pagination
                items_per_page = 10
                total_pages = (len(predictions) + items_per_page - 1) // items_per_page
                
                if total_pages > 1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        page = st.selectbox(
                            f"Page (1-{total_pages}):",
                            range(1, total_pages + 1),
                            index=0
                        )
                else:
                    page = 1
                
                # Calculate start and end indices for current page
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(predictions))
                page_predictions = predictions[start_idx:end_idx]
                
                # Display page info
                st.info(f"Showing predictions {start_idx + 1}-{end_idx} of {len(predictions)}")
                
                # Display predictions as cards
                for i, (side_effect, prob, conf) in enumerate(page_predictions, start_idx + 1):
                    # Determine confidence level
                    if conf >= 0.7:
                        conf_color = "🟢"
                        conf_text = "High"
                    elif conf >= 0.4:
                        conf_color = "🟡"
                        conf_text = "Medium"
                    else:
                        conf_color = "🔴"
                        conf_text = "Low"
                    
                    # Display directly without dropdown
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"**#{i} {side_effect}**")
                    with col2:
                        st.metric("Probability", f"{prob:.4f}")
                    with col3:
                        st.metric("Percentage", f"{prob*100:.2f}%")
                    with col4:
                        st.metric("Confidence", f"{conf_color} {conf_text}")
                    
                    st.divider()
                
                # Simple graph visualization
                st.subheader("📊 Prediction Visualization")
                
                # Create DataFrame for current page
                results_df = pd.DataFrame(page_predictions, columns=['Side Effect', 'Probability', 'Confidence'])
                results_df['Probability'] = results_df['Probability'].round(4)
                results_df['Confidence'] = results_df['Confidence'].round(3)
                
                # Bar chart for current page
                fig = px.bar(
                    results_df, 
                    x='Probability', 
                    y='Side Effect',
                    orientation='h',
                    title=f"Adverse Reaction Probabilities (Page {page})",
                    color='Confidence',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'},
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                with col2:
                    st.metric("Highest Probability", f"{max([p[1] for p in predictions]):.4f}")
                with col3:
                    st.metric("Average Probability", f"{np.mean([p[1] for p in predictions]):.4f}")
                with col4:
                    st.metric("High Confidence Predictions", 
                             len([p for p in predictions if p[2] >= 0.7]))
                
                # CSV Export
                st.subheader("📥 Export Results")
                all_results_df = pd.DataFrame(predictions, columns=['Side Effect', 'Probability', 'Confidence'])
                all_results_df['Probability'] = all_results_df['Probability'].round(4)
                all_results_df['Confidence'] = all_results_df['Confidence'].round(3)
                all_results_df['Percentage'] = (all_results_df['Probability'] * 100).round(2)
                
                csv = all_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results as CSV",
                    data=csv,
                    file_name=f"adr_predictions_{selected_drug}_{selected_indication}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Please check your model files and try again.")
    
    else:
        # Show instructions when no prediction is made
        st.markdown("""
        ## 🚀 How to Use This System
        
        1. **Select a Drug** from the dropdown in the sidebar
        2. **Select a Medical Indication** for the drug
        3. **Click "Predict Adverse Reactions"** to get AI-powered predictions
        
        ### 📋 What You'll Get
        
        - **Adverse Reactions** ranked by probability
        - **Confidence Scores** for each prediction
        - **Visualizations** of the results
        - **Downloadable CSV** with all predictions
        - **Pagination** for easy browsing
        
        ### ⚠️ Important Notes
        
        - This is a research tool and should not replace professional medical advice
        - Predictions are based on historical data and may not reflect individual patient outcomes
        - Always consult healthcare professionals for medical decisions
        """)
        
        # Show some sample data
        st.subheader("📊 Available Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Drugs", len(unique_drugs))
        with col2:
            total_indications = sum(len(indications) for indications in drug_indication_map.values())
            st.metric("Total Drug-Indication Pairs", total_indications)
        with col3:
            st.metric("Total Side Effects", len(df['side_effect_name'].unique()))

if __name__ == "__main__":
    main()
