import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Banking Intelligence Platform", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
    --bg-color: #0E1117;
    --text-primary: #ffffff;
    --text-secondary: #a0aec0;
    --accent-blue: #00d2ff;
    --accent-blue-secondary: #3a7bd5;
    --accent-green: #00ff87;
    --accent-green-secondary: #60efff;
    --accent-red: #ff4b2b;
    --accent-red-secondary: #ff416c;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --glass-shadow: rgba(0, 0, 0, 0.2);
}
.stApp { background-color: var(--bg-color); color: var(--text-primary); font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.glass-container {
    background: var(--glass-bg); backdrop-filter: blur(10px); border-radius: 15px;
    border: 1px solid var(--glass-border); box-shadow: 0 8px 32px 0 var(--glass-shadow);
    padding: 20px; margin-bottom: 20px; transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-container:hover { transform: translateY(-5px); box-shadow: 0 12px 40px 0 rgba(0, 210, 255, 0.2); }
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    border-left: 4px solid var(--accent-blue); border-radius: 10px; padding: 15px; text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); transition: all 0.3s ease;
}
.metric-card:hover { border-left: 4px solid var(--accent-green); box-shadow: 0 0 20px rgba(0, 255, 135, 0.4); }
.metric-card h3 { margin: 0; font-size: 1.1rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }
.metric-card h2 { margin: 10px 0 0 0; font-size: 2.2rem; font-weight: 700; color: var(--text-primary); background: -webkit-linear-gradient(var(--accent-blue), var(--accent-blue-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card.high-value { border-left: 4px solid #f6d365; }
.metric-card.high-value h2 { background: -webkit-linear-gradient(#f6d365, #fda085); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card.risk { border-left: 4px solid var(--accent-red); }
.metric-card.risk h2 { background: -webkit-linear-gradient(var(--accent-red), var(--accent-red-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.streamlit-expanderHeader { background-color: rgba(255, 255, 255, 0.05) !important; border-radius: 8px !important; color: var(--accent-blue) !important; font-weight: 600 !important; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-color); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        return (
            pd.read_csv(BASE_DIR / "Final_Banking_Churn_Dataset.csv"),
            pd.read_csv(BASE_DIR / "European_Bank.csv"),
        )
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_models():
    models_dir = BASE_DIR / "Models"
    models = {}
    try:
        for m in ['scaler', 'pca_model', 'kmeans_model', 'random_forest', 'xgboost', 'extra_trees', 'gradient_boosting']:
            models[m] = joblib.load(models_dir / f"{m}.pkl")
    except Exception as e:
        st.warning(f"Models loading warning: {e}")
    return models

df, raw_df = load_data()
models = load_models()

with st.sidebar:
    st.markdown("## 🏦 FinTech Analytics")
    st.markdown("Enterprise Banking Intelligence")
    st.markdown("---")
    selected = option_menu(
        menu_title=None,
        options=[
            "Executive Overview", "Customer Churn Intelligence", "Geographic Risk Analytics",
            "Customer Segmentation", "PCA & Clusters", "High Value Customers",
            "Financial Risk Insights", "Advanced Drilldown", "AI Model Intelligence"
        ],
        icons=[
            "building", "bar-chart-line", "globe", "diagram-3", "bounding-box",
            "gem", "shield-exclamation", "table", "cpu"
        ],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00d2ff", "font-size": "16px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "color": "#a0aec0"},
            "nav-link-selected": {"background-color": "rgba(0, 210, 255, 0.1)", "color": "#00d2ff", "border-left": "3px solid #00d2ff"},
        }
    )

def kpi_card(title, value, card_type="default"):
    st.markdown(f'<div class="metric-card {card_type}"><h3>{title}</h3><h2>{value}</h2></div>', unsafe_allow_html=True)

if df.empty:
    st.error("No data available.")
    st.stop()

if selected == "Executive Overview":
    st.markdown("<h1 style='text-align: center; color: var(--accent-blue);'>Executive Dashboard</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    overall_churn = df['Exited'].mean() * 100
    with col1: kpi_card("Overall Churn Rate", f"{overall_churn:.1f}%", "risk")
    with col2: kpi_card("Retention Rate", f"{100 - overall_churn:.1f}%", "default")
    with col3: kpi_card("Avg Customer Balance", f"€{raw_df['Balance'].mean():,.0f}", "high-value")
    with col4: kpi_card("Active Customer Ratio", f"{raw_df['IsActiveMember'].mean() * 100:.1f}%", "default")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        fig1 = px.pie(raw_df, names='Exited', title='Churn vs Retention', color='Exited', color_discrete_map={0:'#00d2ff', 1:'#ff4b2b'}, hole=0.7)
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        churn_geo = raw_df.groupby('Geography')['Exited'].mean().reset_index()
        fig2 = px.bar(churn_geo, x='Geography', y='Exited', title='Churn Rate by Geography', color='Geography', color_discrete_sequence=['#00d2ff', '#00ff87', '#3a7bd5'])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        fig2.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Customer Churn Intelligence":
    st.markdown("<h1>Customer Churn Intelligence</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        fig_age = px.histogram(raw_df, x='Age', color='Exited', nbins=30, barmode='group', color_discrete_map={0:'#00d2ff', 1:'#ff4b2b'}, title="Churn Distribution by Age")
        fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        fig_gen = px.bar(raw_df.groupby('Gender')['Exited'].mean().reset_index(), x='Gender', y='Exited', color='Gender', title="Churn Rate by Gender", color_discrete_sequence=['#00ff87', '#3a7bd5'])
        fig_gen.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        fig_gen.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_gen, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Geographic Risk Analytics":
    st.markdown("<h1>Geographic Risk Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    geo_stats = raw_df.groupby('Geography').agg(Total_Customers=('CustomerId', 'count'), Churned=('Exited', 'sum'), Avg_Balance=('Balance', 'mean')).reset_index()
    geo_stats['Churn_Rate'] = geo_stats['Churned'] / geo_stats['Total_Customers']
    fig = px.scatter_geo(geo_stats, locations="Geography", locationmode="country names", size="Total_Customers", color="Churn_Rate", hover_name="Geography", hover_data=["Avg_Balance"], projection="natural earth", title="European Risk Heatmap", color_continuous_scale="Reds")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff", geo=dict(bgcolor='rgba(0,0,0,0)', showland=True, landcolor="#1e1e1e", showocean=True, oceancolor="#0E1117", showcountries=True, countrycolor="#333", scope="europe"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Customer Segmentation":
    st.markdown("<h1>Customer Segmentation Engine</h1>", unsafe_allow_html=True)
    if 'Cluster' in df.columns:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            fig_pie = px.pie(cluster_counts, names='Cluster', values='Count', title="Cluster Distribution", hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            cluster_means = df.groupby('Cluster').mean(numeric_only=True).reset_index()
            categories = ['Age', 'Balance', 'EstimatedSalary', 'CreditScore', 'NumOfProducts']
            fig_radar = go.Figure()
            for i, row in cluster_means.iterrows():
                fig_radar.add_trace(go.Scatterpolar(r=[row[c] for c in categories], theta=categories, fill='toself', name=f'Cluster {row["Cluster"]}'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), showlegend=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff", title="Cluster Radar Comparison")
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif selected == "PCA & Clusters":
    st.markdown("<h1>PCA & Cluster Intelligence</h1>", unsafe_allow_html=True)
    if models.get('pca_model') and 'Cluster' in df.columns:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['CustomerId', 'Year', 'Exited', 'Cluster'], errors='ignore')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        components = pca.fit_transform(numeric_df.fillna(0))
        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
        pca_df['Cluster'] = df['Cluster'].astype(str)
        pca_df['Exited'] = df['Exited'].astype(str)
        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Cluster', symbol='Exited', title="3D PCA Cluster Visualization", opacity=0.7, size_max=5)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff", scene=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "High Value Customers":
    st.markdown("<h1>High Value Customer Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    high_value_df = raw_df[(raw_df['Balance'] > raw_df['Balance'].median()) & (raw_df['EstimatedSalary'] > raw_df['EstimatedSalary'].median())]
    col1, col2 = st.columns(2)
    with col1: kpi_card("Premium Customers", len(high_value_df), "high-value")
    with col2: kpi_card("Premium Churn Rate", f"{(high_value_df['Exited'].mean()*100):.1f}%", "risk")
    fig = px.treemap(high_value_df, path=[px.Constant("High Value Customers"), 'Geography', 'Gender', 'Exited'], values='Balance', color='Exited', color_continuous_scale='Reds', title="High Value Customer Treemap by Geography and Gender")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff", margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Financial Risk Insights":
    st.markdown("<h1>Financial Risk Insights</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        fig_box = px.box(raw_df, x='Exited', y='Balance', color='Exited', title="Balance Distribution by Churn", color_discrete_map={0:'#00d2ff', 1:'#ff4b2b'})
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        fig_box2 = px.box(raw_df, x='Exited', y='CreditScore', color='Exited', title="Credit Score by Churn", color_discrete_map={0:'#00d2ff', 1:'#ff4b2b'})
        fig_box2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_box2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Advanced Drilldown":
    st.markdown("<h1>Advanced Customer Drilldown</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.dataframe(raw_df.head(100), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "AI Model Intelligence":
    st.markdown("<h1 style='color: var(--accent-blue); text-align: center;'>AI Model Intelligence Center</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("### Model Architecture Metrics")
    leaderboard = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "Gradient Boosting", "Extra Trees"],
        "Accuracy": [0.8645, 0.8590, 0.8610, 0.8520],
        "F1 Score": [0.5910, 0.6120, 0.5890, 0.5750],
        "Precision": [0.7510, 0.7320, 0.7450, 0.7100],
        "Recall": [0.4870, 0.5260, 0.4850, 0.4810]
    })
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_bar = px.bar(leaderboard.melt(id_vars="Model"), x="Model", y="value", color="variable", barmode="group",
                 title="Performance Metrics Comparison", color_discrete_sequence=['#00d2ff', '#00ff87', '#f6d365', '#ff4b2b'])
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_radar = go.Figure()
        for idx, row in leaderboard.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['F1 Score'], row['Precision'], row['Recall']],
                theta=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                fill='toself', name=row['Model']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#fff", title="Architecture Radar")
        st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🧠 Live Ensemble Prediction Engine")
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", 18, 100, 35)
            balance = st.number_input("Balance (€)", 0.0, 500000.0, 100000.0)
            salary = st.number_input("Salary (€)", 0.0, 500000.0, 60000.0)
        with c2:
            credit_score = st.number_input("Credit Score", 300, 850, 650)
            tenure = st.number_input("Tenure (Years)", 0, 10, 5)
            products = st.number_input("Num of Products", 1, 4, 2)
        with c3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            year = st.selectbox("Record Year", [2018, 2019, 2020, 2021, 2022], index=1)
        with c4:
            is_active = st.checkbox("Active Member", value=True)
            has_card = st.checkbox("Credit Card", value=True)
            
        selected_models = st.multiselect(
            "Select Models to Execute", 
            ["Random Forest", "XGBoost", "Extra Trees", "Gradient Boosting"], 
            default=["Random Forest", "XGBoost", "Extra Trees", "Gradient Boosting"]
        )
            
        submitted = st.form_submit_button("Engage AI Inference Engine")

    if submitted:
        med_bal = raw_df['Balance'].median()
        med_sal = raw_df['EstimatedSalary'].median()
        
        feature_vector = {
            'Year': year,
            'CreditScore': credit_score,
            'Gender': 1 if gender == "Male" else 0,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': products,
            'HasCrCard': 1 if has_card else 0,
            'IsActiveMember': 1 if is_active else 0,
            'EstimatedSalary': salary,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0,
            'HighValueCustomer': 1 if (balance > med_bal and salary > med_sal) else 0,
            'ActivePremiumCustomer': 1 if (is_active and balance > med_bal and salary > med_sal) else 0,
            'MultiProductUser': 1 if products > 1 else 0,
            'LowCreditRisk': 1 if credit_score > 700 else 0,
            'Cluster': 0, 
            'AgeGroup_Adult': 1 if (age > 25 and age <= 45) else 0,
            'AgeGroup_MiddleAge': 1 if (age > 45 and age <= 65) else 0,
            'AgeGroup_Senior': 1 if (age > 65) else 0,
            'CreditScoreCategory_Medium': 1 if (credit_score >= 600 and credit_score < 750) else 0,
            'CreditScoreCategory_High': 1 if (credit_score >= 750) else 0,
            'BalanceSegment_LowBalance': 1 if (0 < balance < 50000) else 0,
            'BalanceSegment_MediumBalance': 1 if (50000 <= balance < 150000) else 0,
            'BalanceSegment_HighBalance': 1 if balance >= 150000 else 0,
            'SalarySegment_MediumSalary': 1 if (50000 <= salary < 100000) else 0,
            'SalarySegment_HighSalary': 1 if (100000 <= salary < 150000) else 0,
            'SalarySegment_PremiumSalary': 1 if salary >= 150000 else 0,
            'TenureGroup_MidTermCustomer': 1 if (3 <= tenure < 7) else 0,
            'TenureGroup_LongTermCustomer': 1 if tenure >= 7 else 0
        }
        
        # PCA & KMeans Clustering Logic
        scaling_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        clustering_features = [
            'CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts',
            'IsActiveMember', 'HasCrCard', 'HighValueCustomer', 'MultiProductUser', 'LowCreditRisk'
        ]
        
        cluster_scaler = StandardScaler()
        cluster_scaler.fit(raw_df[scaling_features])
        
        live_cluster_df = pd.DataFrame([feature_vector])[clustering_features]
        live_cluster_df[scaling_features] = cluster_scaler.transform(live_cluster_df[scaling_features])
        
        if models.get('pca_model') and models.get('kmeans_model'):
            live_pca = models['pca_model'].transform(live_cluster_df)
            feature_vector['Cluster'] = models['kmeans_model'].predict(live_pca)[0]
        
        input_df = pd.DataFrame([feature_vector])
        
        if models.get('scaler'):
            scaled_input = models['scaler'].transform(input_df)
        else:
            scaled_input = input_df.values
            
        model_map = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Extra Trees": "extra_trees",
            "Gradient Boosting": "gradient_boosting"
        }
        
        preds = {}
        for display_name in selected_models:
            m_name = model_map[display_name]
            if models.get(m_name):
                prob = models[m_name].predict_proba(scaled_input)[0][1]
                preds[display_name] = prob
                
        if preds:
            st.markdown("### Ensemble Results")
            r_cols = st.columns(len(preds))
            
            for i, (display_name, p) in enumerate(preds.items()):
                with r_cols[i]:
                    c_type = "risk" if p > 0.5 else "default"
                    kpi_card(display_name, f"{p*100:.1f}%", c_type)
            
            avg_prob = sum(preds.values()) / len(preds)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            final_col1, final_col2 = st.columns([1, 2])
            with final_col1:
                risk_level = "HIGH RISK" if avg_prob > 0.6 else "MEDIUM RISK" if avg_prob > 0.3 else "LOW RISK"
                r_color = "var(--accent-red)" if avg_prob > 0.6 else "#f6d365" if avg_prob > 0.3 else "var(--accent-green)"
                
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); border: 2px solid {r_color}; border-radius: 10px; padding: 20px; text-align: center;'>
                    <h4 style='color: #a0aec0; margin:0;'>Final AI Consensus</h4>
                    <h1 style='color: {r_color}; font-size: 3rem; margin:10px 0;'>{avg_prob*100:.1f}%</h1>
                    <h3 style='color: {r_color}; margin:0;'>{risk_level}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with final_col2:
                st.markdown("### 🤖 Business Strategy Recommendation")
                if avg_prob > 0.6:
                    if feature_vector['HighValueCustomer']:
                        st.error("URGENT: Premium customer is highly likely to churn. Trigger VIP Retention Protocol immediately. Assign personal relationship manager and offer tailored investment rates.")
                    else:
                        st.warning("Customer presents high churn risk. Target with loyalty incentives, reduced fee offers, or personalized engagement campaigns.")
                elif avg_prob > 0.3:
                    st.info("Moderate churn risk detected. Monitor account activity closely. Send re-engagement emails and highlight newly available banking products.")
                else:
                    st.success("Stable customer profile. No immediate retention action required. Consider cross-selling standard products to increase engagement.")
                    
                st.markdown("#### Confidence Metrics")
                st.progress(int(abs(0.5 - avg_prob) * 200)) # Simple pseudo-confidence based on distance from 0.5
                st.markdown(f"<small>Model Agreement Variance: {np.std(list(preds.values())):.3f}</small>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

