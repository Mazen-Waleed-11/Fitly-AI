import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="StyleFit AI Pro", layout="wide")

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
except:
    st.error("⚠️ Data files not found. Run 'train_models.py' first.")
    st.stop()

# --- 2. NUTRITION LOGIC ---
def get_nutrition_plan(bmi):
    if bmi < 18.5:
        return {"title": "Bulking Plan", "m1": "Meal 1: 3 Eggs, Toast, Avocado", "m2": "Meal 2: 150g Salmon, 200g Potato", "m3": "Meal 3: 200g Yogurt, Nuts"}
    elif bmi < 25:
        return {"title": "Maintenance", "m1": "Meal 1: 100g Oats, Protein Shake", "m2": "Meal 2: 150g Chicken, Rice, Salad", "m3": "Meal 3: 150g Beef Stir-fry"}
    elif bmi < 30:
        return {"title": "Lean Plan", "m1": "Meal 1: Egg White Omelet", "m2": "Meal 2: 150g White Fish, Quinoa", "m3": "Meal 3: 150g Turkey & Veggies"}
    else:
        return {"title": "Reset Plan", "m1": "Meal 1: 2 Boiled Eggs, Cucumber", "m2": "Meal 2: 200g Chicken, Large Salad", "m3": "Meal 3: 150g Baked Fish"}

# --- 3. DYNAMIC STYLING ---
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

# --- 4. SIDEBAR NAVIGATION & EXERCISES ---
st.sidebar.title("🎵 Boutique Radio")
music_file = "the-fashion-music-409865.mp3"
if os.path.exists(music_file):
    st.sidebar.audio(music_file, format="audio/mp3")

st.sidebar.divider()
st.sidebar.subheader("🚀 Navigation")
app_mode = st.sidebar.radio("Go to:", ["🎯 Shirt Predictor", "👖 Pants Predictor", "📊 Analytics"])

st.sidebar.divider()

# AUTOMATIC EXERCISE LOGIC (3 Exercises Each)
if app_mode == "🎯 Shirt Predictor":
    st.sidebar.subheader("💪 Upper Body Exercises")
    st.sidebar.write("**1. V-Lat Pull down**")
    if os.path.exists("V-bar-Lat-Pulldown.gif"): st.sidebar.image("V-bar-Lat-Pulldown.gif")
    st.sidebar.write("**2. Seated Bench Press**")
    if os.path.exists("Seated-Bench-Press.gif"): st.sidebar.image("Seated-Bench-Press.gif")
    st.sidebar.write("**3. Arm Curl Machine**")
    if os.path.exists("ARM_CURL_MC.gif"): st.sidebar.image("ARM_CURL_MC.gif")

elif app_mode == "👖 Pants Predictor":
    st.sidebar.subheader("💪 Lower Body Exercises")
    st.sidebar.write("**1. Hack Squat**")
    if os.path.exists("HACK_SQT.gif"): st.sidebar.image("HACK_SQT.gif")
    st.sidebar.write("**2. Leg Extension**")
    if os.path.exists("LEG-EXTENSION.gif"): st.sidebar.image("LEG-EXTENSION.gif")
    st.sidebar.write("**3. Standing Calf Raise**")
    if os.path.exists("STD_CALF_RAISE.gif"): st.sidebar.image("STD_CALF_RAISE.gif")

# --- 5. MAIN CONTENT AREA ---

if app_mode == "🎯 Shirt Predictor":
    st.title("Upper Body Size Predictor")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        item = st.selectbox("Clothing Item", ["T-Shirt", "Sweater", "Jacket"])
        img_dict = {"T-Shirt": "t-shirt.jpg", "Sweater": "sweater.jpg", "Jacket": "jacket.jpg"}
        if os.path.exists(img_dict[item]): st.image(img_dict[item], width=300)
        
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
        if os.path.exists(p_img_dict[p_item]): st.image(p_img_dict[p_item], width=300)
            
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
    
    # 1. Performance Table
    st.subheader("🏆 Algorithm Performance Summary")
    acc_dict = {name.replace('_',' ').title(): data['Accuracy'] for name, data in stats.items()}
    acc_df = pd.DataFrame(list(acc_dict.items()), columns=['Algorithm', 'Accuracy'])
    st.table(acc_df.style.format({'Accuracy': '{:.2%}'}))

    # 2. Accuracy Comparison Visual (The new chart)
    st.subheader("📈 Performance Visualization")
    st.bar_chart(acc_df.set_index('Algorithm'))

    # 3. Correlation Heatmap
    st.divider()
    if os.path.exists("correlation_heatmap.png"):
        st.image("correlation_heatmap.png", caption="Feature Correlation (Weight & BMI are strongest predictors)", width="stretch")