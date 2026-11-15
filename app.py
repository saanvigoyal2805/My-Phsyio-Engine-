import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
st.markdown(
    """
    <style>
    .stApp { background-color: #F0F4F8; }
    .stApp, .stApp p, .stApp label, .stApp li, .stApp h2, .stApp h3 { color: #1E1E2F; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; }
    </style>
    """,
    unsafe_allow_html=True,
)
# ==========================================
# 1. MAIN HEADERS
# ==========================================
st.markdown(
    '<h1 style="text-align: center; color: #FF4B2B; font-family: sans-serif;'
    ' font-weight: 900; letter-spacing: 2px; margin-bottom: 0px;">⚡ MY PHYSIO'
    ' ENGINE</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; color: #6C757D; font-family: sans-serif;'
    ' font-size: 18px; font-weight: 600; margin-top: 5px; margin-bottom:'
    ' 40px;">Smart Physiology & Nutrition Matrix</p>',
    unsafe_allow_html=True,
)

# ==========================================
# 2. DIET DATABASE
# ==========================================
DIET_DATABASE = {
    "Vegetarian": {
        1: {
            "Breakfast": "Paneer & Oats Scramble (Paneer: 150g, Oats: 40g).",
            "Lunch": (
                "High-Protein Soya Matar Curry with 2 Millet Rotis & Thick"
                " Curd."
            ),
            "Evening Snack": (
                "Roasted Makhana (30g) paired with a Double-Scoop Whey Shake."
            ),
            "Dinner": (
                "Grilled Tofu Steaks (150g) over a bed of Stir-Fried Broccoli"
                " and Brown Rice."
            ),
        },
        0: {
            "Breakfast": (
                "Moong Dal Vegetable Cheela (2 medium cheelas with spinach"
                " stuffing)."
            ),
            "Lunch": (
                "Sprouted Chickpea Salad Bowl (1.5 cups) paired with 1 cup"
                " low-fat curd."
            ),
            "Evening Snack": (
                "Boiled Kala Chana Chaat (1 cup) seasoned with roasted cumin."
            ),
            "Dinner": (
                "Lauki Moong Dal Soup (Large Bowl) with a mixed green salad"
                " platter."
            ),
        },
    },
    "Vegan": {
        1: {
            "Breakfast": (
                "Almond-Milk Protein Smoothie (Oats: 50g, Soy Protein Isolate: 1"
                " scoop)."
            ),
            "Lunch": (
                "Tempeh Stir-Fry with Quinoa (Tempeh: 150g, Quinoa: 80g"
                " cooked)."
            ),
            "Evening Snack": (
                "Roasted Organic Peanut Butter (2 tbsps) on 2 multi-grain"
                " toasts."
            ),
            "Dinner": (
                "High-Protein Black Bean & Soy Chunk Cutlets baked crisp."
            ),
        },
        0: {
            "Breakfast": (
                "Smashed Tofu Bhurji (Tofu: 150g, Turmeric, Capsicum, Tomatoes)."
            ),
            "Lunch": (
                "Boiled Brown Lentil (Masoor) Soup paired with a shredded"
                " cabbage bowl."
            ),
            "Evening Snack": (
                "Chia Seed Pudding prepared in unsweetened Coconut Milk."
            ),
            "Dinner": (
                "Mushroom & Spinach sauteed in broth, topped with boiled"
                " chickpeas."
            ),
        },
    },
    "Non-Vegetarian": {
        1: {
            "Breakfast": (
                "4 Whole Eggs Scramble + 1 Bowl Oatmeal topped with Walnuts."
            ),
            "Lunch": (
                "Grilled Chicken Breast (200g) + 1 Cup Brown Rice + Mixed"
                " Veggies."
            ),
            "Evening Snack": (
                "High-Protein Greek Yogurt + Handful of Almonds + 1 Apple."
            ),
            "Dinner": (
                "Pan-Seared Salmon/Fish + Quinoa + Steamed Broccoli and Moong"
                " Dal."
            ),
        },
        0: {
            "Breakfast": (
                "3 Egg White Omelet with Spinach & Mushrooms + 1 Slice Whole"
                " Wheat Toast."
            ),
            "Lunch": (
                "Lean Shredded Chicken Salad + Mixed Greens + Olive Oil"
                " Vinaigrette."
            ),
            "Evening Snack": "Roasted Makhana (Fox Nuts) + Green Tea.",
            "Dinner": (
                "Clear Chicken & Vegetable Broth + Side of Stir-fried Tofu or"
                " Paneer."
            ),
        },
    },
}

# ==========================================
# 3. BACKGROUND ML ENGINE
# ==========================================
@st.cache_resource
def train_comprehensive_model():
    np.random.seed(42)
    n_samples = 150
    df = pd.DataFrame({
        "Age": np.random.randint(15, 65, n_samples),
        "Weight_kg": np.random.normal(70, 15, n_samples),
        "Sleep_Hrs": np.random.normal(7.5, 1.2, n_samples),
        "Goal_Feature": np.random.choice([0, 1], n_samples),
    })
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_data)
    return kmeans, scaler, df

model, data_scaler, trained_df = train_comprehensive_model()

# ==========================================
# 4. USER INTERFACE (SIDEBAR)
# ==========================================
st.sidebar.markdown(
    '<p style="font-size: 20px; font-weight: bold; color: #1E1E2F;'
    ' margin-bottom: 10px;">🎛️ Configure Profile</p>',
    unsafe_allow_html=True,
)

# Unset defaults: Empty string for name
user_name = st.sidebar.text_input("Name", value="", placeholder="Enter your name")
age = st.sidebar.slider("Age (Years)", 15, 65, 18)
weight = st.sidebar.slider("Weight (kg)", 40, 120, 65)

height_unit = st.sidebar.radio(
    "Height Format", ["Centimeters", "Feet & Inches"], horizontal=True
)
if height_unit == "Centimeters":
    height_cm = st.sidebar.number_input("Height (cm)", 120, 220, 165)
else:
    col_ft, col_in = st.sidebar.columns(2)
    feet = col_ft.number_input("Feet", 3, 7, 5)
    inches = col_in.number_input("Inches", 0, 11, 5)
    height_cm = round((feet * 30.48) + (inches * 2.54), 1)

sleep = st.sidebar.slider("Average Sleep (Hours)", 4.0, 10.0, 7.5, step=0.5)

GOAL_OPTIONS = [
    "Build Muscle Mass",
    "Increase Strength",
    "Lose Body Fat",
    "Improve Endurance",
    "Better Posture",
    "General Health & Energy",
]

# Unset defaults: Blank list for goals selection
selected_goals = st.sidebar.multiselect(
    "Select Goals (Pick 2+):",
    options=GOAL_OPTIONS,
    default=[],
    placeholder="Choose your fitness goals...",
)

# Unset defaults: Selectbox starts with no selection
dietary_restriction = st.sidebar.selectbox(
    "Diet Preference",
    ["Vegetarian", "Vegan", "Non-Vegetarian"],
    index=None,
    placeholder="Select diet preference...",
)

# ==========================================
# 5. DYNAMIC OUTPUT DASHBOARD
# ==========================================
execute_pipeline = st.button(
    "🚀 GENERATE MY OPTIMIZED PROTOCOL", use_container_width=True
)

if execute_pipeline:
    # Check input conditions
    if not user_name.strip():
        st.warning("⚠️ Please enter your name in the sidebar.")
    elif len(selected_goals) < 2:
        st.warning("⚠️ Please select at least 2 goals in the sidebar.")
    elif dietary_restriction is None:
        st.warning("⚠️ Please select a diet preference in the sidebar.")
    else:
        # Calculate scores when all inputs are valid
        goal_numeric = sum(
            1 for g in selected_goals if "Muscle" in g or "Strength" in g
        ) / len(selected_goals)
        bmi = round(weight / ((height_cm / 100) ** 2), 1)
        water_intake = round((weight * 0.033), 1)

        scaled_metrics = data_scaler.transform(
            np.array([[age, weight, sleep, goal_numeric]])
        )
        assigned_group = model.predict(scaled_metrics)[0]

        st.markdown(
            f'<h3 style="font-family: sans-serif; color: #1E1E2F;">Welcome to'
            f' the Grid, {user_name}!</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div style="display: flex; gap: 20px; margin-bottom: 25px;">
            <div style="background:#F8F9FA; border-top: 5px solid #FF9F43; padding: 15px; border-radius: 8px; width: 48%;">BMI: <strong style="font-size:22px;">{bmi}</strong></div>
            <div style="background:#F8F9FA; border-top: 5px solid #00CFDD; padding: 15px; border-radius: 8px; width: 48%;">Water: <strong style="font-size:22px;">{water_intake}L</strong></div>
        </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            '<h3 style="font-family: sans-serif; color: #1E1E2F;">🏋️‍♂️ Your'
            ' Prescribed Training Strategy</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:{"#E8F5E9" if assigned_group == 1 else "#E3F2FD"};'
            ' border-left: 6px solid #2E7D32; padding: 15px; border-radius:'
            ' 4px;">'
            f'{"Focus: Heavy compound lifts and progressive overload." if assigned_group == 1 else "Focus: High-intensity cardio and metabolic conditioning."}'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            '<h3 style="font-family: sans-serif; color: #1E1E2F;">🥗 Your'
            ' Daily Core Nutrition Blueprint</h3>',
            unsafe_allow_html=True,
        )
        for m, f in DIET_DATABASE[dietary_restriction][assigned_group].items():
            st.markdown(
                '<div style="background:#FFFDE7; border-left: 6px solid'
                ' #FBC02D; padding: 15px; margin-bottom:10px; border-radius:'
                f' 6px;"><strong>{m}:</strong> {f}</div>',
                unsafe_allow_html=True,
            )
else:
    st.info(
        "👈 Configure your profile details in the sidebar and click **GENERATE"
        " MY OPTIMIZED PROTOCOL** to build your personalized plan."
    )
