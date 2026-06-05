# ── DigiCow Adoption Prediction App ───────────────────────────────────────────
# Streamlit web application for predicting farmer adoption probability
# Uses logistic regression models trained in digicow_clean_analysis.ipynb

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ── Load saved models and artifacts ───────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    Load all saved models and lookup tables.
    @st.cache_resource means this runs once and stays in memory.
    """
    with open('models/lr_07.pkl', 'rb') as f:
        lr_07 = pickle.load(f)
    with open('models/lr_90.pkl', 'rb') as f:
        lr_90 = pickle.load(f)
    with open('models/lr_120.pkl', 'rb') as f:
        lr_120 = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/trainer_rate_map.pkl', 'rb') as f:
        trainer_rate_map = pickle.load(f)
    with open('models/county_rate_map.pkl', 'rb') as f:
        county_rate_map = pickle.load(f)
    with open('models/topic_rate_map.pkl', 'rb') as f:
        topic_rate_map = pickle.load(f)
    with open('models/global_rate.pkl', 'rb') as f:
        global_rate = pickle.load(f)

    return (lr_07, lr_90, lr_120, scaler,
            trainer_rate_map, county_rate_map,
            topic_rate_map, global_rate)

(lr_07, lr_90, lr_120, scaler,
 trainer_rate_map, county_rate_map,
 topic_rate_map, global_rate) = load_models()

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "DigiCow Adoption Predictor",
    page_icon  = "🐮",
    layout     = "wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title(" 🐮 DigiCow Farmer Adoption Predictor")
st.markdown("""
Predict the probability that a farmer adopts an agricultural practice
within **7, 90 and 120 days** of attending a training session.

Enter the farmer's details to generate a prediction.
""")

st.divider()

# ── Input sidebar ─────────────────────────────────────────────────────────────
st.sidebar.header("Farmer Details")

# Cooperative membership
cooperative = st.sidebar.selectbox(
    "Belongs to a cooperative?",
    options = [0, 1],
    format_func = lambda x: "Yes" if x == 1 else "No"
)

# Registration method
registration = st.sidebar.selectbox(
    "Registration method",
    options = ["Ussd", "Manual"]
)
is_ussd = 1 if registration == "Ussd" else 0

# Trainer
trainer = st.sidebar.selectbox(
    "Trainer",
    options = sorted(trainer_rate_map.keys())
)

# County
county = st.sidebar.selectbox(
    "County",
    options = sorted(county_rate_map.keys())
)

# Topics
all_topics = sorted(topic_rate_map.keys())
selected_topics = st.sidebar.multiselect(
    "Topics covered in session",
    options  = all_topics,
    default  = all_topics[:2]
)

# ── Feature engineering ───────────────────────────────────────────────────────
# Compute smoothed rates for selected trainer, county and topics

trainer_rate = trainer_rate_map.get(trainer, global_rate)
county_rate  = county_rate_map.get(county, global_rate)

if len(selected_topics) > 0:
    topic_rates  = [topic_rate_map.get(t, global_rate) for t in selected_topics]
    topic_rate   = sum(topic_rates) / len(topic_rates)
    n_topics     = len(selected_topics)
else:
    topic_rate   = global_rate
    n_topics     = 0

# Build feature vector
features = pd.DataFrame([{
    'belong_to_cooperative' : cooperative,
    'is_ussd_registered'    : is_ussd,
    'trainer_rate_smoothed' : trainer_rate,
    'county_rate_smoothed'  : county_rate,
    'topic_rate_smoothed'   : topic_rate,
    'n_topics'              : n_topics
}])

# Scale features
features_scaled = scaler.transform(features)

# ── Predictions ───────────────────────────────────────────────────────────────
prob_07  = lr_07.predict_proba(features_scaled)[0, 1]
prob_90  = lr_90.predict_proba(features_scaled)[0, 1]
prob_120 = lr_120.predict_proba(features_scaled)[0, 1]

# ── Display results ───────────────────────────────────────────────────────────
st.header("Predicted Adoption Probabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label = "Within 7 Days",
        value = f"{prob_07*100:.1f}%"
    )

with col2:
    st.metric(
        label = "Within 90 Days",
        value = f"{prob_90*100:.1f}%"
    )

with col3:
    st.metric(
        label = "Within 120 Days",
        value = f"{prob_120*100:.1f}%"
    )

st.divider()

# ── Probability chart ─────────────────────────────────────────────────────────
st.subheader("Adoption Probability by Time Window")

fig, ax = plt.subplots(figsize=(8, 4))

windows = ['7 Days', '90 Days', '120 Days']
probs   = [prob_07, prob_90, prob_120]
colours = ['#0072B2', '#0072B2', '#0072B2']

bars = ax.bar(windows, [p * 100 for p in probs],
              color=colours, edgecolor='white', width=0.4)

for bar, prob in zip(bars, probs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{prob*100:.1f}%',
        ha='center', va='bottom', fontsize=12
    )

ax.set_ylabel('Adoption Probability (%)', fontsize=11)
ax.set_title('Predicted Adoption Probability by Time Window',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(probs) * 100 * 1.4 + 5)
ax.spines[['top', 'right']].set_visible(False)

st.pyplot(fig)

st.divider()

# ── Feature breakdown ─────────────────────────────────────────────────────────
st.subheader("What Is Driving This Prediction?")

feature_display = pd.DataFrame({
    'Feature'     : [
        'Cooperative membership',
        'Registration method',
        'Trainer historical rate',
        'County historical rate',
        'Topic historical rate',
        'Number of topics'
    ],
    'Value' : [
        'Yes' if cooperative == 1 else 'No',
        registration,
        f"{trainer_rate*100:.2f}%",
        f"{county_rate*100:.2f}%",
        f"{topic_rate*100:.2f}%",
        str(n_topics)
    ]
})

st.dataframe(feature_display, hide_index=True, use_container_width=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("""
Built by Fiona Huini  
Models: Logistic Regression trained on DigiCow farmer training data  
Data source: Zindi Africa - DigiCow Farmer Adoption Prediction Challenge
""")

