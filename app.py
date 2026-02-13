import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import sys

# --- 1. CONFIG & ASSETS ---
# Change page_icon to your logo file if you want it to appear in the browser tab too!
st.set_page_config(page_title="Fitly-AI Pro", layout="wide", page_icon="🎯")

# --- 2. AUTO-TRAIN LOGIC ---
# This ensures the app works on Streamlit Cloud even without the .pkl files
def verify_models():
    required_files = ['scaler.pkl', 'pants_model.pkl', 'model_stats.pkl']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        st.info("🚀 First-time setup: Training AI models on the server...")
        try:
            # Uses sys.executable to ensure it uses the cloud's python environment
            subprocess.run([sys.executable, "train_models.py"], check=True)
            st.success("✅ Models ready!")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()

verify_models()

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    stats = joblib.load('model_stats.pkl')
    p_model = joblib.load('pants_model.pkl')
    p_scaler = joblib.load('pants_scaler.pkl')
    p_le = joblib.load('pants_le.pkl')
    return scaler, le, stats, p_model, p_scaler, p_le

try:
    scaler, le, stats, p_model, p_scaler, p_le = load_assets()
except Exception as e:
    st.error("⚠️ Data files not found. Check if 'train_models.py' ran correctly.")
    st.stop()

# --- 3. NUTRITION LOGIC ---
def get_nutrition_plan(bmi):
    if bmi < 18.5:
        return {"title": "Bulking Plan", "m1": "Meal 1: 3 Eggs, Toast, Avocado", "m2": "Meal 2: 150g Salmon, 200g Potato", "m3": "Meal 3: 200g Yogurt, Nuts"}
    elif bmi < 25:
        return {"title": "Maintenance", "m1": "Meal 1: 100g Oats, Protein Shake", "m2": "Meal 2: 150g Chicken, Rice, Salad", "m3": "Meal 3: 150g Beef Stir-fry"}
    elif bmi < 30:
        return {"title": "Lean Plan", "m1": "Meal 1: Egg White Omelet", "m2": "Meal 2: 150g White Fish, Quinoa", "m3": "Meal 3: 150g Turkey & Veggies"}
    else:
        return {"title": "Reset Plan", "m1": "Meal 1: 2 Boiled Eggs, Cucumber", "m2": "Meal 2: 200g Chicken, Large Salad", "m3": "Meal 3: 150g Baked Fish"}

# --- 4. DYNAMIC STYLING (Glassmorphism) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    [data-testid="stVerticalBlock"] > div:has(div.stMetric), .stTable, .stDataFrame, .glass {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px; border-radius: 15px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR NAVIGATION & LOGO ---
with st.sidebar:
    # --- LOGO SECTION ---
    # REPLACE "your_logo_filename.png" WITH YOUR IMAGE NAME
    logo_path = "fitly_logo.png" 
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.title("🎯 Fitly-AI")
    
    st.divider()
    st.subheader("🎵 Boutique Radio")
    music_file = "the-fashion-music-409865.mp3"
    if os.path.exists(music_file):
        st.audio(music_file, format="audio/mp3")

    st.divider()
    st.subheader("🚀 Navigation")
    app_mode = st.radio("Go to:", ["🎯 Shirt Predictor", "👖 Pants Predictor", "📊 Analytics"])

# --- SIDEBAR EXERCISES (Dynamic based on mode) ---
if app_mode == "🎯 Shirt Predictor":
    st.sidebar.subheader("💪 Upper Body Exercises")
    exercises = [("V-Lat Pull down", "V-bar-Lat-Pulldown.gif"), 
                 ("Seated Bench Press", "Seated-Bench-Press.gif"), 
                 ("Arm Curl Machine", "ARM_CURL_MC.gif")]
elif app_mode == "👖 Pants Predictor":
    st.sidebar.subheader("💪 Lower Body Exercises")
    exercises = [("Hack Squat", "HACK_SQT.gif"), 
                 ("Leg Extension", "LEG-EXTENSION.gif"), 
                 ("Standing Calf Raise", "STD_CALF_RAISE.gif")]
else:
    exercises = []

for name, file in exercises:
    st.sidebar.write(f"**{name}**")
    if os.path.exists(file): st.sidebar.image(file)

# --- 6. MAIN CONTENT AREA ---

if app_mode == "🎯 Shirt Predictor":
    st.title("Upper Body Size Predictor")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        item = st.selectbox("Clothing Item", ["T-Shirt", "Sweater", "Jacket"])
        img_dict = {"T-Shirt": "t-shirt.jpg", "Sweater": "sweater.jpg", "Jacket": "jacket.jpg"}
        if os.path.exists(img_dict.get(item, "")): st.image(img_dict[item], width=300)
        
        w = st.number_input("Weight (kg)", 30.0, 150.0, 70.0, key="sw")
        h = st.number_input("Height (cm)", 100.0, 220.0, 170.0, key="sh")
        age = st.number_input("Age", 10, 100, 25, key="sa")
        m_name = st.selectbox("AI Algorithm", ["Random Forest", "KNN", "Decision Tree", "Logistic Regression", "SVM"])
        
        if st.button("Calculate Top Fit"):
            bmi = w / ((h / 100) ** 2)
            m_key = m_name.lower().replace(" ", "_")
            model = joblib.load(f'{m_key}_model.pkl')
            res = model.predict(scaler.transform([[w, age, h, bmi]]))
            st.session_state['s_res'] = {"size": le.inverse_transform(res)[0], "bmi": bmi, "item": item}

    with col2:
        if 's_res' in st.session_state:
            data = st.session_state['s_res']
            st.success(f"Recommended {data['item']} Size: **{data['size']}**")
            st.metric("Body Mass Index", f"{data['bmi']:.1f}")
            plan = get_nutrition_plan(data['bmi'])
            st.markdown(f"<div class='glass'><h3>🥗 {plan['title']}</h3><p><b>Meal 1:</b> {plan['m1']}</p><p><b>Meal 2:</b> {plan['m2']}</p><p><b>Meal 3:</b> {plan['m3']}</p></div>", unsafe_allow_html=True)

elif app_mode == "👖 Pants Predictor":
    st.title("Pants Size Predictor (Waist/Length)")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        p_item = st.selectbox("Pants Type", ["Jeans", "Cargo", "Melton Wool"])
        p_img_dict = {"Jeans": "Jeans.jpg", "Cargo": "Cargo.jpg", "Melton Wool": "Melton.jpg"}
        if os.path.exists(p_img_dict.get(p_item, "")): st.image(p_img_dict[p_item], width=300)
            
        pw = st.number_input("Weight (kg)", 30.0, 150.0, 70.0, key="pw")
        ph = st.number_input("Height (cm)", 100.0, 220.0, 170.0, key="ph")
        pa = st.number_input("Age", 10, 100, 25, key="pa")
        
        if st.button("Calculate Pants Fit"):
            pbmi = pw / ((ph / 100) ** 2)
            res_p = p_model.predict(p_scaler.transform([[pw, pa, ph, pbmi]]))
            st.session_state['p_res'] = {"size": p_le.inverse_transform(res_p)[0], "bmi": pbmi, "item": p_item}

    with col2:
        if 'p_res' in st.session_state:
            data_p = st.session_state['p_res']
            st.success(f"Recommended {data_p['item']} Size: **{data_p['size']}**")
            st.metric("Body Mass Index", f"{data_p['bmi']:.1f}")
            plan = get_nutrition_plan(data_p['bmi'])
            st.markdown(f"<div class='glass'><h3>🥗 {plan['title']}</h3><p><b>Meal 1:</b> {plan['m1']}</p><p><b>Meal 2:</b> {plan['m2']}</p><p><b>Meal 3:</b> {plan['m3']}</p></div>", unsafe_allow_html=True)

elif app_mode == "📊 Analytics":
    st.title("Project Technical Analytics")
    st.subheader("🏆 Algorithm Performance Summary")
    acc_dict = {name.replace('_',' ').title(): data['Accuracy'] for name, data in stats.items()}
    acc_df = pd.DataFrame(list(acc_dict.items()), columns=['Algorithm', 'Accuracy'])
    st.table(acc_df.style.format({'Accuracy': '{:.2%}'}))

    st.subheader("📈 Performance Visualization")
    st.bar_chart(acc_df.set_index('Algorithm'))

    st.divider()
    if os.path.exists("correlation_heatmap.png"):
        st.image("correlation_heatmap.png", caption="Feature Correlation (Weight & BMI are strongest predictors)")