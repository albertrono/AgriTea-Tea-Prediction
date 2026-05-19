import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import yaml
import bcrypt
from pathlib import Path

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriTea Predictor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SHARED CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #0f3d23; }
[data-testid="stSidebar"] * { color: #d1fae5 !important; }
[data-testid="stSidebar"] .stRadio label { color: #d1fae5 !important; font-size: 14px; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #6ee7b7 !important; }

/* ── Main ── */
.main { background-color: #f5f9f6; }

/* ── Auth card ── */
.auth-card {
    background: white;
    border-radius: 16px;
    padding: 40px 44px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    max-width: 460px;
    margin: 40px auto;
}
.auth-logo {
    text-align: center;
    font-size: 48px;
    margin-bottom: 4px;
}
.auth-title {
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    color: #0f3d23;
    margin-bottom: 4px;
}
.auth-subtitle {
    text-align: center;
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 28px;
}
.auth-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 20px 0;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #1a6b3a;
}

/* ── Section titles ── */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f3d23;
    margin: 1.2rem 0 0.6rem 0;
}

/* ── Badges ── */
.badge-recommended {
    background: #d1fae5;
    color: #065f46;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: 0.04em;
}

/* ── Model info box ── */
.model-info-box {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    color: #065f46;
    margin-bottom: 12px;
}

/* ── Prediction result card ── */
.pred-result-card {
    background: linear-gradient(135deg, #0f3d23 0%, #1a6b3a 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    text-align: center;
    margin: 1rem 0;
}
.pred-result-card .yield-value {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -1px;
}
.pred-result-card .yield-label {
    font-size: 16px;
    opacity: 0.8;
    margin-top: 4px;
}

/* ── Recommendation cards ── */
.rec-card {
    background: white;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    border-left: 4px solid #1a6b3a;
    font-size: 14px;
    color: #1f2937;
}

/* ── Performance table ── */
.perf-table th {
    background-color: #0f3d23;
    color: white;
    padding: 8px 14px;
    text-align: left;
    font-size: 13px;
}
.perf-table td {
    padding: 8px 14px;
    font-size: 13px;
    border-bottom: 1px solid #e5e7eb;
}
.perf-table tr:nth-child(even) td { background-color: #f9fafb; }
.perf-table tr.best-row td { background-color: #ecfdf5; font-weight: 600; color: #065f46; }

label { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).parent / "auth_config.yaml"

def load_config() -> dict:
    """Load credentials config, creating it if it doesn't exist."""
    if not CONFIG_PATH.exists():
        default = {
            "credentials": {"usernames": {}},
            "cookie": {
                "name": "agritea_auth",
                "key": "agritea_secret_key_change_in_prod",
                "expiry_days": 7,
            },
        }
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(default, f)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def register_user(username: str, name: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    config = load_config()
    users = config["credentials"]["usernames"]

    if not username or not name or not email or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if "@" not in email:
        return False, "Please enter a valid email address."
    if username in users:
        return False, "Username already exists. Please choose another."
    if any(u["email"] == email for u in users.values()):
        return False, "An account with that email already exists."

    users[username] = {
        "name": name,
        "email": email,
        "password": hash_password(password),
    }
    save_config(config)
    return True, "Account created successfully! You can now log in."


def login_user(username: str, password: str) -> tuple[bool, str, str]:
    """Attempt login. Returns (success, message, display_name)."""
    config = load_config()
    users = config["credentials"]["usernames"]

    if not username or not password:
        return False, "Please enter your username and password.", ""
    if username not in users:
        return False, "Invalid username or password.", ""
    if not verify_password(password, users[username]["password"]):
        return False, "Invalid username or password.", ""

    return True, "Login successful.", users[username]["name"]


# ══════════════════════════════════════════════════════════════════════════════
# AUTH UI
# ══════════════════════════════════════════════════════════════════════════════

def show_auth_page():
    """Render the login / registration page."""
    # Centre a narrow column
    _, col, _ = st.columns([1, 1.6, 1])

    with col:
        st.markdown('<div class="auth-logo">🍃</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">AgriTea Predictor</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-subtitle">AI-powered tea yield prediction for Kenyan farms</div>',
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        # ── LOGIN TAB ──────────────────────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Login", type="primary", use_container_width=True, key="login_btn"):
                if username and password:
                    success, msg, display_name = login_user(username.strip(), password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username.strip()
                        st.session_state["display_name"] = display_name
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please fill in both fields.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Don't have an account? Switch to the **Register** tab above.")

        # ── REGISTER TAB ──────────────────────────────────────────────────────
        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_name     = st.text_input("Full Name",        key="reg_name",     placeholder="e.g. Jane Wanjiku")
            reg_email    = st.text_input("Email Address",    key="reg_email",    placeholder="e.g. jane@example.com")
            reg_username = st.text_input("Username",         key="reg_username", placeholder="Choose a username (min 3 chars)")
            reg_password = st.text_input("Password",         type="password", key="reg_password", placeholder="At least 6 characters")
            reg_confirm  = st.text_input("Confirm Password", type="password", key="reg_confirm",  placeholder="Repeat your password")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", type="primary", use_container_width=True, key="register_btn"):
                if reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(
                        reg_username.strip(),
                        reg_name.strip(),
                        reg_email.strip(),
                        reg_password,
                    )
                    if success:
                        st.success(msg)
                        st.info("Head over to the **Login** tab to sign in.")
                    else:
                        st.error(msg)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Already have an account? Switch to the **Login** tab above.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "XGBoost (Tuned) ⭐": os.path.join(MODEL_DIR, "best_xgboost_model.joblib"),
        "XGBoost":            os.path.join(MODEL_DIR, "xgboost_model.joblib"),
        "Random Forest":      os.path.join(MODEL_DIR, "random_forest_model.joblib"),
        "Linear Regression":  os.path.join(MODEL_DIR, "linear_regression_model.joblib"),
    }
    for name, path in paths.items():
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return models


MODEL_METRICS = {
    "XGBoost (Tuned) ⭐": {"MAE": 7.746, "RMSE": 10.164, "R²": 0.723, "best": True},
    "XGBoost":            {"MAE": 7.470, "RMSE": 10.131, "R²": 0.725, "best": False},
    "Random Forest":      {"MAE": None,  "RMSE": None,   "R²": None,  "best": False},
    "Linear Regression":  {"MAE": 8.238, "RMSE": 10.988, "R²": 0.676, "best": False},
}

FEATURES = ["Rainfall_mm", "Rainfall_Lag1_mm", "Rainfall_Lag2_mm", "Avg_Temp_C", "Soil_pH"]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (shown after login)
# ══════════════════════════════════════════════════════════════════════════════

def show_main_app():
    models = load_models()

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🍃 AgriTea")
        st.markdown("### Tea Yield Prediction")
        st.markdown("---")

        # Logged-in user info
        display_name = st.session_state.get("display_name", "User")
        username     = st.session_state.get("username", "")
        st.markdown(
            f'<div style="background:#1a6b3a;border-radius:8px;padding:10px 14px;margin-bottom:12px;">'
            f'<div style="font-size:12px;opacity:0.7;">Logged in as</div>'
            f'<div style="font-size:15px;font-weight:700;">{display_name}</div>'
            f'<div style="font-size:11px;opacity:0.6;">@{username}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            ["Predict Yield", "Recommendations", "Model Performance"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("#### Select Model")
        model_choice = st.selectbox(
            "Choose model",
            list(models.keys()) if models else list(MODEL_METRICS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        if model_choice and ("⭐" in model_choice or "Tuned" in model_choice):
            st.markdown(
                '<span class="badge-recommended" style="background:#d1fae5;color:#065f46;'
                'padding:4px 10px;border-radius:99px;font-size:12px;font-weight:700;">'
                "✓ Recommended model</span>",
                unsafe_allow_html=True,
            )
        st.markdown("")
        m = MODEL_METRICS.get(model_choice, {})
        if m.get("R²"):
            st.metric("R² Score", f'{m["R²"]:.3f}')
            st.metric("MAE (KG)", f'{m["MAE"]:.3f}')

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["authenticated", "username", "display_name"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── PREDICT YIELD ──────────────────────────────────────────────────────────
    if page == "Predict Yield":
        st.markdown("## 🍃 Tea Yield Prediction")
        st.markdown(
            f'<div class="model-info-box">🤖 <b>Active model:</b> {model_choice} &nbsp;|&nbsp; '
            f'Enter field conditions below and click <b>Predict</b></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="section-title">🌧 Rainfall Conditions</div>', unsafe_allow_html=True)
            rainfall = st.number_input(
                "Current month rainfall (mm)",
                min_value=0.0, max_value=600.0, value=120.0, step=1.0,
                help="Total rainfall recorded this month in millimeters",
            )
            rainfall_lag1 = st.number_input(
                "Last month rainfall — Lag 1 (mm)",
                min_value=0.0, max_value=600.0, value=110.0, step=1.0,
                help="Rainfall from 1 month ago.",
            )
            rainfall_lag2 = st.number_input(
                "Two months ago rainfall — Lag 2 (mm)",
                min_value=0.0, max_value=600.0, value=95.0, step=1.0,
                help="Rainfall from 2 months ago.",
            )

        with col2:
            st.markdown('<div class="section-title">🌡 Soil & Climate</div>', unsafe_allow_html=True)
            avg_temp = st.number_input(
                "Average temperature (°C)",
                min_value=5.0, max_value=45.0, value=22.0, step=0.5,
                help="Average ambient temperature this month in Celsius",
            )
            soil_ph = st.slider(
                "Soil pH",
                min_value=3.0, max_value=9.0, value=5.8, step=0.1,
                help="Tea grows best at pH 4.5–6.0.",
            )
            st.caption(
                "🔬 Soil pH is the strongest predictor of yield "
                f"(XGBoost importance: **86.6%**). Optimal range for tea: **4.5 – 6.0**"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🌿 Predict Yield", type="primary", use_container_width=False)

        if predict_btn:
            selected_model = models.get(model_choice)
            if selected_model is None:
                st.error("Selected model could not be loaded. Please check the .joblib files.")
            else:
                input_df = pd.DataFrame(
                    [[rainfall, rainfall_lag1, rainfall_lag2, avg_temp, soil_ph]],
                    columns=FEATURES,
                )
                try:
                    prediction = float(selected_model.predict(input_df)[0])
                    prediction = max(0.0, prediction)

                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.markdown(
                            f"""
                            <div class="pred-result-card">
                                <div style="font-size:14px;opacity:0.7;margin-bottom:8px;">PREDICTED TEA YIELD</div>
                                <div class="yield-value">{prediction:.1f} <span style="font-size:28px;font-weight:400">KG</span></div>
                                <div class="yield-label">per acre · {model_choice}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        number={"suffix": " KG", "font": {"size": 28}},
                        delta={"reference": 200, "increasing": {"color": "#1a6b3a"}},
                        title={"text": "Yield Score vs Baseline (200 KG)"},
                        gauge={
                            "axis": {"range": [0, 500], "tickwidth": 1},
                            "bar": {"color": "#1a6b3a"},
                            "steps": [
                                {"range": [0, 150],  "color": "#fee2e2"},
                                {"range": [150, 280], "color": "#fef3c7"},
                                {"range": [280, 500], "color": "#d1fae5"},
                            ],
                            "threshold": {
                                "line": {"color": "#0f3d23", "width": 3},
                                "thickness": 0.75,
                                "value": prediction,
                            },
                        },
                    ))
                    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Interpretation")
                    if prediction < 150:
                        st.error("⚠️ Low yield predicted. Review soil pH and irrigation levels.")
                    elif prediction < 280:
                        st.warning("📊 Moderate yield. Consider fertiliser adjustments and verify rainfall data.")
                    else:
                        st.success("✅ Good yield expected. Conditions look favourable for this harvest cycle.")

                    with st.expander("View input summary"):
                        summary_df = pd.DataFrame({
                            "Feature": [
                                "Rainfall (mm)", "Rainfall Lag 1 (mm)", "Rainfall Lag 2 (mm)",
                                "Avg Temperature (°C)", "Soil pH",
                            ],
                            "Value": [rainfall, rainfall_lag1, rainfall_lag2, avg_temp, soil_ph],
                        })
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # ── RECOMMENDATIONS ────────────────────────────────────────────────────────
    elif page == "Recommendations":
        st.markdown("## 💡 Smart Recommendations")
        st.markdown("Evidence-based guidance for improving tea yield on your farm.")

        tabs = st.tabs(["Soil Management", "Water & Irrigation", "Harvesting", "General Agronomy"])

        rec_sets = [
            [
                ("✅", "#1a6b3a", "Maintain soil pH between **4.5 and 6.0** for optimal tea growth. This is the single most important yield factor (86% model importance)."),
                ("⚠️", "#f59e0b", "If pH is above 6.0, apply **agricultural sulphur or acidifying fertilisers** to lower it gradually."),
                ("⚠️", "#f59e0b", "If pH is below 4.5, apply **dolomitic lime** carefully — avoid over-liming, which can lock out micronutrients."),
                ("ℹ️", "#3b82f6", "Test soil pH at least **twice per year** — before the long rains and short rains seasons."),
                ("✅", "#1a6b3a", "Incorporate **organic matter (compost or green manure)** to improve soil structure and moisture retention."),
            ],
            [
                ("✅", "#1a6b3a", "Monitor **monthly rainfall carefully** — both current and lagged values affect yield prediction."),
                ("⚠️", "#f59e0b", "During dry spells (< 80mm/month), **increase irrigation frequency** to maintain soil moisture."),
                ("ℹ️", "#3b82f6", "Use drip irrigation where possible — it reduces water use by up to 40% compared to overhead methods."),
                ("✅", "#1a6b3a", "Keep detailed rainfall records for at least **3 months back** — these lag values directly feed the model."),
                ("ℹ️", "#3b82f6", "Install a **simple rain gauge** on the farm to get accurate local measurements rather than relying on regional averages."),
            ],
            [
                ("✅", "#1a6b3a", "Harvest using the **two-leaves-and-a-bud** standard for highest quality green leaf."),
                ("ℹ️", "#3b82f6", "Plucking rounds should be every **7–14 days** depending on flush growth rate."),
                ("⚠️", "#f59e0b", "Avoid harvesting during or immediately after heavy rain — wet leaf reduces factory acceptance quality."),
                ("✅", "#1a6b3a", "Track yield per plucking round to calibrate the predictor with your actual farm data over time."),
            ],
            [
                ("✅", "#1a6b3a", "Apply **nitrogen-rich fertiliser (CAN or urea)** during the flush period to boost leaf development."),
                ("ℹ️", "#3b82f6", "Follow a **split fertiliser application** schedule — 3–4 smaller doses per year rather than one large dose."),
                ("⚠️", "#f59e0b", "Watch for **blister blight and red spider mite** — common in humid conditions above 80% humidity."),
                ("✅", "#1a6b3a", "Maintain shade trees at optimal density — they regulate temperature and reduce moisture stress in dry months."),
                ("ℹ️", "#3b82f6", "Average temperatures between **18°C and 28°C** are ideal. Use this app to track how deviations affect your predicted yield."),
            ],
        ]

        headers = ["#### Soil Health", "#### Water & Rainfall Management",
                   "#### Harvesting Guidance", "#### General Agronomy"]

        for tab, header, recs in zip(tabs, headers, rec_sets):
            with tab:
                st.markdown(header)
                for icon, border, text in recs:
                    st.markdown(
                        f'<div class="rec-card" style="border-left-color:{border};">{icon} {text}</div>',
                        unsafe_allow_html=True,
                    )

    # ── MODEL PERFORMANCE ──────────────────────────────────────────────────────
    elif page == "Model Performance":
        st.markdown("## 📊 Model Performance Comparison")
        st.markdown(
            "All models were trained on the same dataset. "
            "**XGBoost (Tuned)** is recommended — it achieved the lowest error after hyperparameter optimisation via GridSearchCV."
        )

        st.markdown("#### Evaluation Metrics on Test Set")
        perf_rows = ""
        for name, m in MODEL_METRICS.items():
            mae  = f'{m["MAE"]:.3f}'  if m["MAE"]  else "—"
            rmse = f'{m["RMSE"]:.3f}' if m["RMSE"] else "—"
            r2   = f'{m["R²"]:.3f}'   if m["R²"]   else "—"
            row_class = "best-row" if m["best"] else ""
            star = " ⭐ Recommended" if m["best"] else ""
            perf_rows += (
                f'<tr class="{row_class}"><td>{name}{star}</td>'
                f'<td>{mae}</td><td>{rmse}</td><td>{r2}</td></tr>'
            )

        st.markdown(
            f"""
            <table class="perf-table" style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">
                <thead><tr>
                    <th>Model</th>
                    <th>MAE (↓ better)</th>
                    <th>RMSE (↓ better)</th>
                    <th>R² (↑ better)</th>
                </tr></thead>
                <tbody>{perf_rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Random Forest metrics not recorded in notebook output; model file is available for use.")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### XGBoost Feature Importance")
        fi_data = {
            "Soil_pH":          0.8656,
            "Rainfall_Lag1_mm": 0.0484,
            "Avg_Temp_C":       0.0432,
            "Rainfall_Lag2_mm": 0.0285,
            "Rainfall_mm":      0.0143,
        }
        fi_df = pd.DataFrame(fi_data.items(), columns=["Feature", "Importance"]).sort_values("Importance")

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker_color=["#0f3d23" if f == "Soil_pH" else "#6ee7b7" for f in fi_df["Feature"]],
            text=[f"{v:.1%}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300,
            xaxis_title="Importance score",
            yaxis_title=None,
            margin=dict(l=10, r=60, t=20, b=40),
            template="plotly_white",
            xaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "🔬 **Soil pH dominates at 86.6%** importance. "
            "Small changes in pH have an outsized effect on predicted yield — "
            "keeping it in the 4.5–6.0 range is the highest-leverage intervention a farmer can make."
        )

        st.markdown("#### Understanding the Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**MAE — Mean Absolute Error**")
            st.markdown(
                "Average kg difference between predicted and actual yield. "
                "A MAE of 7.7 means predictions are off by ~7.7 KG on average."
            )
        with c2:
            st.markdown("**RMSE — Root Mean Squared Error**")
            st.markdown(
                "Like MAE but penalises large errors more heavily. "
                "Useful for detecting outlier predictions."
            )
        with c3:
            st.markdown("**R² — Coefficient of Determination**")
            st.markdown(
                "How much yield variance the model explains. "
                "R² = 0.72 means the model captures 72% of the yield variation in the data."
            )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("authenticated"):
    show_auth_page()
else:
    show_main_app()
