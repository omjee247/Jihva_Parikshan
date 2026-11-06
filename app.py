"""
AI Jihva Pariksha — Ayurvedic Tongue Diagnostic System
Updated with all 4 new visual & UX refinements:
1. High-Contrast Readability: Lighter background veil (88% white-ivory wash over the Ayurvedic Rishi & herbs illustration),
   solid crisp white content cards (`#ffffff`), and deep high-contrast dark text (`#0f172a`) so every word is effortless to read.
2. PDF Download Only: Removed the PNG download option; kept only a compact PDF download button.
3. Compact, Properly Sized Buttons: Removed full-width stretched buttons (`use_container_width=True`) and styled buttons
   to a clean, compact pill/rounded rectangle width.
4. Redesigned Main Header Banner ("AI Jihva Pariksha — Ayurvedic Tongue Diagnostic System"): Replaced the heavy dark green block
   with a luminous Royal Saffron-Gold & Warm Ivory Ayurvedic Temple/Manuscript header with deep emerald-and-burgundy typography.
"""

import os
import hashlib
import base64
import importlib
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import ayurveda_quotes
importlib.reload(ayurveda_quotes)
from ayurveda_quotes import get_random_quote

import tongue_analyzer
importlib.reload(tongue_analyzer)
from tongue_analyzer import analyze_tongue_image

import report_generator
importlib.reload(report_generator)
from report_generator import generate_report_pdf_bytes

from feedback_store import (
    init_db,
    predict_with_feedback_calibration,
    save_patient_review,
    get_patient_reviews,
)


st.set_page_config(
    page_title="Jihva Pariksha — Ayurvedic Tongue Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

init_db()


@st.cache_data(show_spinner=False)
def get_background_image_data_uri() -> str:
    """
    Loads the Ayurvedic Rishi (Maharishi Sushruta / Vaidya) & Sacred Herbs background image
    (`ayurveda_bg.png`) as a base64 data URI.
    """
    bg_path = os.path.join(os.path.dirname(__file__), "ayurveda_bg.png")
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}", "cover"

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">
      <rect width="1200" height="800" fill="#fbfaf6"/>
    </svg>"""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}", "cover"


bg_uri, bg_size_mode = get_background_image_data_uri()

st.markdown(f"""
<style>
    /* Hide all Streamlit branding, top toolbar, sidebar toggle, footer, and developer profile/crown badges */
    #MainMenu {{visibility: hidden !important; display: none !important;}}
    footer {{visibility: hidden !important; display: none !important;}}
    header {{visibility: hidden !important; display: none !important;}}
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="manage-app-button"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    .stAppDeployButton,
    [class^="viewerBadge_"],
    [class*="viewerBadge_"],
    [class^="profilePreview_"],
    [class*="profilePreview_"],
    [class^="styles_viewerBadge__"],
    a[href*="streamlit.io/cloud"],
    a[href*="share.streamlit.io/user"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
    }}

    /* Ultra-Light Translucent Veil over the Ayurvedic Rishi & Herbs Background for Effortless Readability */
    .stApp {{
        background-image:
            linear-gradient(180deg, rgba(255, 255, 255, 0.87) 0%, rgba(253, 251, 245, 0.89) 100%),
            url("{bg_uri}");
        background-repeat: no-repeat;
        background-size: {bg_size_mode};
        background-position: center center;
        background-attachment: fixed;
        color: #0f172a !important;
    }}

    /* Force dark, crisp, readable typography across all Streamlit labels, headings & markdown */
    .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: #0f172a;
    }}

    /* Solid Crisp White Content Cards so background artwork never interferes with reading text */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #ffffff !important;
        border: 1.5px solid #d4c4a8 !important;
        border-radius: 14px !important;
        padding: 0.35rem !important;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07) !important;
    }}

    /* Make all buttons compact (natural width, never overly long) */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {{
        width: auto !important;
        min-width: 150px !important;
        max-width: 290px !important;
        padding: 0.45rem 1.25rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    }}

    /* Primary Button Styling (Warm Ayurvedic Emerald) */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"],
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #15803d 0%, #166534 100%) !important;
        color: #ffffff !important;
        border: 1px solid #14532d !important;
    }}
    .stButton > button[kind="primary"] *,
    .stFormSubmitButton > button[kind="primary"] *,
    .stDownloadButton > button * {{
        color: #ffffff !important;
    }}

    /* Reduce top page container padding so header sits higher and cleaner */
    .block-container {{
        padding-top: 1.0rem !important;
        padding-bottom: 1.5rem !important;
    }}

    /* Redesigned Compact Header Banner: Low padding, Saffron-Gold & Sandalwood Ivory Card */
    .main-header {{
        background: linear-gradient(135deg, #fffcf2 0%, #fef3c7 50%, #fde68a 100%);
        padding: 0.55rem 1.15rem;
        border-radius: 10px;
        border: 1.5px solid #d97706;
        border-left: 6px solid #b45309;
        margin-bottom: 0.65rem;
        box-shadow: 0 3px 10px rgba(180, 83, 9, 0.10);
    }}
    .main-header h2 {{
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1.32rem !important;
        line-height: 1.25 !important;
        color: #78350f !important;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    .main-header p {{
        margin: 2px 0 0 0 !important;
        padding: 0 !important;
        font-size: 0.88rem !important;
        line-height: 1.3 !important;
        color: #92400e !important;
        font-weight: 500;
    }}

    /* Sacred Shloka Card (at bottom of report) */
    .shloka-box {{
        background: #fffbeb;
        border: 1.5px solid #f59e0b;
        border-left: 6px solid #b45309;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        margin-top: 0.75rem;
        margin-bottom: 0.85rem;
        color: #1e293b;
        box-shadow: 0 3px 12px rgba(217, 119, 6, 0.08);
    }}

    /* Self-contained Report Card */
    .report-card {{
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        color: #0f172a;
    }}

    /* Fixed Diagnostic Parameter Table */
    .fixed-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.96rem;
    }}
    .fixed-table th {{
        background-color: #78350f;
        color: #ffffff !important;
        text-align: left;
        padding: 10px 14px;
    }}
    .fixed-table td {{
        padding: 10px 14px;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a !important;
        font-weight: 500;
    }}
    .fixed-table tr:nth-child(even) {{
        background-color: #f8fafc;
    }}

    /* Disclaimer Box */
    .disclaimer-box {{
        background-color: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.85rem 1.2rem;
        font-size: 0.88rem;
        color: #1e293b !important;
        margin-top: 0.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# Active DOM script to hide any Streamlit Cloud viewer/profile badges
components.html(
    """
    <script>
    (function() {
        function hideBadges() {
            try {
                const doc = window.parent.document;
                const selectors = [
                    '[class^="viewerBadge_"]',
                    '[class*="viewerBadge_"]',
                    '[class^="profilePreview_"]',
                    '[class*="profilePreview_"]',
                    '[data-testid="manage-app-button"]',
                    'a[href*="streamlit.io/cloud"]',
                    'a[href*="share.streamlit.io/user"]'
                ];
                selectors.forEach(sel => {
                    doc.querySelectorAll(sel).forEach(el => {
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                    });
                });
            } catch (e) {}
        }
        hideBadges();
        setInterval(hideBadges, 800);
    })();
    </script>
    """,
    height=0,
)


# --- SESSION STATE INITIALIZATION FOR PERSISTENT PATIENT DETAILS & REPORT ---
if "saved_name" not in st.session_state:
    st.session_state.saved_name = ""
if "saved_age" not in st.session_state:
    st.session_state.saved_age = 0
if "saved_gender" not in st.session_state:
    st.session_state.saved_gender = "Select Gender"
if "is_editing_details" not in st.session_state:
    st.session_state.is_editing_details = True
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "stored_img_bgr" not in st.session_state:
    st.session_state.stored_img_bgr = None
if "stored_image_seed" not in st.session_state:
    st.session_state.stored_image_seed = None
if "active_quote" not in st.session_state:
    st.session_state.active_quote = get_random_quote()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1


# --- POPUP CONFIRMATION DIALOG FOR "CLEAR ALL" ---
@st.dialog("⚠️ Confirm Clear All Details & Report")
def confirm_clear_all_dialog():
    st.markdown(
        "Are you sure you want to clear **all patient details**, the **uploaded tongue photograph**, and the **generated diagnostic report**?"
    )
    st.caption("If you click Cancel, you will return to the exact same page with all your details and report intact.")
    c_yes, c_cancel = st.columns(2)
    with c_yes:
        if st.button("🗑️ Yes, Clear All", type="primary"):
            st.session_state.saved_name = ""
            st.session_state.saved_age = 0
            st.session_state.saved_gender = "Select Gender"
            st.session_state.is_editing_details = True
            st.session_state.report_generated = False
            st.session_state.stored_img_bgr = None
            st.session_state.stored_image_seed = None
            st.session_state.uploader_key += 1
            st.rerun()
    with c_cancel:
        if st.button("↩️ Cancel"):
            st.rerun()


# --- MAIN HEADER (NEW ROYAL SAFFRON-GOLD & SANDALWOOD IVORY THEME) ---
st.markdown("""
<div class="main-header">
    <h2>🌿 AI Jihva Pariksha — Ayurvedic Tongue Diagnostic System</h2>
    <p>
        Classical <b>Ashtasthana Pariksha</b> Visual Tongue Analysis • Tridosha Balance, Agni Assessment & Pathya-Apathya Remedies
    </p>
</div>
""", unsafe_allow_html=True)


# --- TOP ACTION BAR: COMPACT EDIT DETAILS BUTTON & CLEAR ALL BUTTON ---
details_saved_valid = (
    bool(st.session_state.saved_name.strip())
    and int(st.session_state.saved_age) >= 1
    and st.session_state.saved_gender in ["Male", "Female", "Other"]
)

top_bar_left, top_bar_right = st.columns([4, 1.2])
with top_bar_left:
    if details_saved_valid and not st.session_state.is_editing_details:
        if st.button("✏️ Edit Patient Details"):
            st.session_state.is_editing_details = True
            st.rerun()
with top_bar_right:
    if details_saved_valid or st.session_state.stored_img_bgr is not None:
        if st.button("🗑️ Clear All"):
            confirm_clear_all_dialog()


# --- PATIENT DETAILS SECTION (SAVE / EDIT WITHOUT EVER LOSING THE GENERATED REPORT) ---
with st.container(border=True):
    if st.session_state.is_editing_details or not details_saved_valid:
        st.markdown("#### 👤 Enter / Edit Patient Details")
        with st.form("patient_details_form"):
            p_col1, p_col2, p_col3 = st.columns([1.6, 0.8, 1])
            with p_col1:
                input_name = st.text_input(
                    "Full Name *",
                    value=st.session_state.saved_name,
                    placeholder="Enter your full name"
                )
            with p_col2:
                input_age = st.number_input(
                    "Age (Years) *",
                    min_value=0,
                    max_value=115,
                    value=int(st.session_state.saved_age),
                    step=1
                )
            with p_col3:
                gender_opts = ["Select Gender", "Male", "Female", "Other"]
                g_idx = gender_opts.index(st.session_state.saved_gender) if st.session_state.saved_gender in gender_opts else 0
                input_gender = st.selectbox("Gender *", options=gender_opts, index=g_idx)

            f_btn1, f_btn2, _ = st.columns([1, 1, 3])
            with f_btn1:
                save_clicked = st.form_submit_button("💾 Save Details", type="primary")
            with f_btn2:
                cancel_edit = st.form_submit_button(
                    "Cancel",
                    disabled=not details_saved_valid
                )

            if save_clicked:
                if not input_name.strip() or int(input_age) < 1 or input_gender not in ["Male", "Female", "Other"]:
                    st.error("Please enter a valid Full Name, Age (≥ 1), and select your Gender before saving.")
                else:
                    st.session_state.saved_name = input_name.strip()
                    st.session_state.saved_age = int(input_age)
                    st.session_state.saved_gender = input_gender
                    st.session_state.is_editing_details = False
                    st.rerun()
            elif cancel_edit and details_saved_valid:
                st.session_state.is_editing_details = False
                st.rerun()
    else:
        st.markdown(
            f"#### 👤 Patient Details  \n"
            f"**Name:** `{st.session_state.saved_name}` &nbsp;|&nbsp; "
            f"**Age:** `{st.session_state.saved_age} Years` &nbsp;|&nbsp; "
            f"**Gender:** `{st.session_state.saved_gender}`"
        )


details_saved_valid = (
    bool(st.session_state.saved_name.strip())
    and int(st.session_state.saved_age) >= 1
    and st.session_state.saved_gender in ["Male", "Female", "Other"]
)

# --- UPLOAD PHOTO OPTION APPEARS ONCE PATIENT DETAILS ARE SAVED ---
if details_saved_valid:
    with st.container(border=True):
        up_col1, up_col2 = st.columns([2, 1])

        with up_col1:
            input_method = st.radio(
                "Select Tongue Photo Option:",
                options=["Upload Tongue Photo (JPG/PNG)", "Use Camera (Take Live Photo)"],
                horizontal=True,
                key=f"radio_src_{st.session_state.uploader_key}"
            )

            new_raw_bytes = None
            if input_method == "Upload Tongue Photo (JPG/PNG)":
                uploaded_file = st.file_uploader(
                    "Upload a clear close-up tongue photograph",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"file_up_{st.session_state.uploader_key}"
                )
                if uploaded_file is not None:
                    new_raw_bytes = uploaded_file.read()

            elif input_method == "Use Camera (Take Live Photo)":
                cam_file = st.camera_input(
                    "Protrude tongue gently in natural daylight (without flash)",
                    key=f"cam_up_{st.session_state.uploader_key}"
                )
                if cam_file is not None:
                    new_raw_bytes = cam_file.read()

            if new_raw_bytes is not None:
                new_seed = hashlib.md5(new_raw_bytes).hexdigest()[:10]
                if st.session_state.stored_image_seed != new_seed:
                    file_bytes = np.asarray(bytearray(new_raw_bytes), dtype=np.uint8)
                    decoded_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if decoded_bgr is not None:
                        st.session_state.stored_img_bgr = decoded_bgr
                        st.session_state.stored_image_seed = new_seed
                        st.session_state.report_generated = False
                        st.session_state.active_quote = get_random_quote()

        with up_col2:
            st.info(
                "**📋 Photo Guidelines**\n"
                "1. Take photo in **morning before brushing/eating**.\n"
                "2. Use **natural daylight** (no flash).\n"
                "3. Relax tongue naturally."
            )


# --- PREVIEW UPLOADED IMAGE & COMPACT SUBMIT BUTTON ---
if details_saved_valid and st.session_state.stored_img_bgr is not None:
    img_bgr = st.session_state.stored_img_bgr
    image_seed = st.session_state.stored_image_seed

    with st.container(border=True):
        st.markdown("#### 🖼️ Uploaded Tongue Image Preview")
        prev_col1, prev_col2 = st.columns([1, 2])
        with prev_col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Tongue Photograph", use_container_width=True)
        with prev_col2:
            st.markdown(
                f"**Patient:** `{st.session_state.saved_name}` • `{st.session_state.saved_age} Years` • `{st.session_state.saved_gender}`\n\n"
                "Click **Submit & Analyze Tongue** below to generate your **Jihva Pariksha Diagnostic Report**."
            )
            if st.button("🔍 Submit & Analyze Tongue", type="primary"):
                st.session_state.report_generated = True


# --- DISPLAY REPORT (STAYS VISIBLE EVEN WHEN EDITING/SAVING PATIENT DETAILS!) ---
if details_saved_valid and st.session_state.stored_img_bgr is not None and st.session_state.report_generated:
    img_bgr = st.session_state.stored_img_bgr
    image_seed = st.session_state.stored_image_seed
    clean_name = st.session_state.saved_name
    patient_age = int(st.session_state.saved_age)
    patient_gender = st.session_state.saved_gender

    report_id = f"JP-{image_seed[:6].upper()}-{patient_age}{patient_gender[0].upper()}"
    raw_analysis = analyze_tongue_image(img_bgr, age=patient_age, gender=patient_gender)

    if raw_analysis.get("is_rejected"):
        with st.container(border=True):
            st.error(
                f"🚫 **Invalid Image Rejected — Not a Human Tongue Photograph**\n\n"
                f"**Reason:** {raw_analysis.get('rejection_reason', 'The uploaded image does not appear to be a close-up human tongue photograph.')}\n\n"
                f"Please upload a **clear, close-up photograph of a human tongue** (not a full face/person portrait, book page, screenshot, wall, or random object) and click **Submit & Analyze Tongue** again."
            )
    else:
        calibration = predict_with_feedback_calibration(
            raw_analysis["feature_vector"],
            raw_analysis["dominant_dosha"],
            raw_analysis["dosha_scores"]
        )
        raw_analysis["dominant_dosha"] = calibration["calibrated_dosha"]
        raw_analysis["dosha_scores"] = calibration["calibrated_scores"]

        active_quote = st.session_state.active_quote

        if raw_analysis["quality"]["quality_warning"]:
            st.warning(f"⚠️ **Notice:** {raw_analysis['quality']['quality_warning']}")

        with st.container(border=True):
            st.subheader("🔬 Tongue Examination Visual Maps")
            v_col1, v_col2, v_col3, v_col4 = st.columns(4)
            with v_col1:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="1. Uploaded Tongue Photo", use_container_width=True)
            with v_col2:
                st.image(
                    cv2.cvtColor(raw_analysis["images"]["seg_overlay"], cv2.COLOR_BGR2RGB),
                    caption="2. Boundary & Tooth-Mark Map",
                    use_container_width=True
                )
            with v_col3:
                st.image(
                    cv2.cvtColor(raw_analysis["images"]["coating_overlay"], cv2.COLOR_BGR2RGB),
                    caption=f"3. Coating Map ({raw_analysis['parameters']['Coating_Area']})",
                    use_container_width=True
                )
            with v_col4:
                st.image(
                    cv2.cvtColor(raw_analysis["images"]["fissure_overlay"], cv2.COLOR_BGR2RGB),
                    caption=f"4. Fissure Map ({raw_analysis['parameters']['Fissure_Count']} grooves)",
                    use_container_width=True
                )

        # --- JIHVA PARIKSHA DIAGNOSTIC REPORT WITH LANGUAGE TOGGLE (ENGLISH vs HINDI) ---
        with st.container(border=True):
            hdr_col1, hdr_col2 = st.columns([1.5, 1])
            with hdr_col1:
                st.subheader("📑 Jihva Pariksha Diagnostic Report")
            with hdr_col2:
                report_lang = st.radio(
                    "🌐 View & Download Report Language (भाषा चुनें):",
                    options=["English", "Hindi (हिंदी)"],
                    horizontal=True
                )

            is_hi_mode = "Hindi" in report_lang

            rep_left, rep_right = st.columns([1.12, 1])
            params = raw_analysis["parameters"]

            with rep_left:
                if is_hi_mode:
                    gender_hi = {"Male": "पुरुष", "Female": "महिला", "Other": "अन्य"}.get(patient_gender, patient_gender)
                    st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between; border-bottom:2px solid #78350f; padding-bottom:8px; margin-bottom:12px;">
                            <div><b>रोगी (Patient):</b> {clean_name} ({patient_age} वर्ष, {gender_hi})</div>
                            <div><b>रिपोर्ट ID:</b> <code>{report_id}</code></div>
                        </div>
                        <table class="fixed-table">
                            <thead>
                                <tr>
                                    <th>मापदंड (Parameter)</th>
                                    <th>निरीक्षण (Observation)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td><b>रंग (Colour)</b></td><td>{params['Colour_Hi']}</td></tr>
                                <tr><td><b>परत (Coating)</b></td><td>{params['Coating_Hi']} ({params['Coating_Type_Hi']})</td></tr>
                                <tr><td><b>परत का क्षेत्र (Coating Area)</b></td><td>{params['Coating_Area']}</td></tr>
                                <tr><td><b>दरारें (Fissures)</b></td><td>{params['Fissures_Hi']}</td></tr>
                                <tr><td><b>आकार (Shape)</b></td><td>{params['Shape_Hi']}</td></tr>
                                <tr><td><b>दांतों के निशान (Tooth Marks)</b></td><td>{params['Tooth_Marks_Hi']}</td></tr>
                                <tr><td><b>नमी (Moisture)</b></td><td>{params['Moisture_Hi']}</td></tr>
                                <tr><td><b>प्रधान दोष (Dominant Dosha)</b></td><td><b style="color:#15803d;">{raw_analysis['dominant_dosha_hi']}</b></td></tr>
                                <tr><td><b>अग्नि / आम स्थिति (Agni / Ama)</b></td><td>{raw_analysis['ama_status_hi']}</td></tr>
                            </tbody>
                        </table>
                        <p style="font-size:0.9rem; color:#0f172a; margin-top:12px; margin-bottom:0;">
                            <b>आयु एवं लिंग विश्लेषण (वय परीक्षा):</b> {raw_analysis['age_gender_context_hi']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="report-card">
                        <div style="display:flex; justify-content:space-between; border-bottom:2px solid #78350f; padding-bottom:8px; margin-bottom:12px;">
                            <div><b>Patient:</b> {clean_name} ({patient_age} yrs, {patient_gender})</div>
                            <div><b>Report ID:</b> <code>{report_id}</code></div>
                        </div>
                        <table class="fixed-table">
                            <thead>
                                <tr>
                                    <th>Parameter</th>
                                    <th>Observation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td><b>Colour (Varna)</b></td><td>{params['Colour_Full']}</td></tr>
                                <tr><td><b>Coating (Lepa)</b></td><td>{params['Coating']} ({params['Coating_Type']})</td></tr>
                                <tr><td><b>Coating area</b></td><td>{params['Coating_Area']}</td></tr>
                                <tr><td><b>Fissures (Sphutita)</b></td><td>{params['Fissures']}</td></tr>
                                <tr><td><b>Shape (Akriti)</b></td><td>{params['Shape_State']}</td></tr>
                                <tr><td><b>Tooth marks</b></td><td>{params['Tooth_Marks']}</td></tr>
                                <tr><td><b>Moisture (Snigdhata)</b></td><td>{params['Moisture']}</td></tr>
                                <tr><td><b>Dominant Dosha</b></td><td><b style="color:#15803d;">{raw_analysis['dominant_dosha']}</b></td></tr>
                                <tr><td><b>Agni / Ama Status</b></td><td>{raw_analysis['ama_status']}</td></tr>
                            </tbody>
                        </table>
                        <p style="font-size:0.9rem; color:#0f172a; margin-top:12px; margin-bottom:0;">
                            <b>Age & Gender Assessment (Vaya Pariksha):</b> {raw_analysis['age_gender_context']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with rep_right:
                with st.container(border=True):
                    st.markdown(
                        "#### ⚖️ त्रिदोष संतुलन एवं जीभ के क्षेत्रीय अंग"
                        if is_hi_mode else
                        "#### ⚖️ Tridosha Distribution & Regional Organ Zones"
                    )
                    d_scores = raw_analysis["dosha_scores"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("वात (Vata)" if is_hi_mode else "Vata (Air/Ether)", f"{d_scores['Vata']}%")
                    m2.metric("पित्त (Pitta)" if is_hi_mode else "Pitta (Fire/Water)", f"{d_scores['Pitta']}%")
                    m3.metric("कफ (Kapha)" if is_hi_mode else "Kapha (Earth/Water)", f"{d_scores['Kapha']}%")

                    st.markdown("**जीभ के क्षेत्रीय अंग (Jihva Sthana):**" if is_hi_mode else "**Regional Tongue Organ Mapping (Jihva Sthana):**")
                    active_zones = raw_analysis["zones_hi"] if is_hi_mode else raw_analysis["zones"]
                    for zone_label, zone_obs in active_zones.items():
                        st.markdown(f"- **{zone_label}:** {zone_obs}")

                    st.markdown("---")
                    st.markdown(
                        "#### 🌿 आयुर्वेदिक उपचार, परामर्श एवं पथ्य-अपथ्य (आहार)"
                        if is_hi_mode else
                        "#### 🌿 Classical Ayurvedic Prescription, Remedies & Diet"
                    )
                    rem = raw_analysis["remedies_hi"] if is_hi_mode else raw_analysis["remedies"]
                    st.markdown("**1. दैनिक मुख एवं जिह्वा दिनचर्या (Dinacharya):**" if is_hi_mode else "**1. Daily Oral Regimen (Dinacharya):**")
                    for item in rem["dinacharya"]:
                        st.markdown(f"- {item}")
                    st.markdown("**2. आयुर्वेदिक हर्बल उपचार (Upachara):**" if is_hi_mode else "**2. Herbal Remedies & Formulations (Upachara):**")
                    for item in rem["herbal_remedies"]:
                        st.markdown(f"- {item}")
                    st.markdown("**3. पथ्य एवं अपथ्य आहार (Diet):**" if is_hi_mode else "**3. Wholesome Diet (Pathya) & Restrictions (Apathya):**")
                    for item in rem["pathya_diet"]:
                        st.markdown(f"- ✅ **{'क्या खाएं (Pathya)' if is_hi_mode else 'Eat (Pathya)'}:** {item}")
                    for item in rem["apathya_avoid"]:
                        st.markdown(f"- ❌ **{'परहेज करें (Apathya)' if is_hi_mode else 'Avoid (Apathya)'}:** {item}")

            # --- QUOTE AT THE BOTTOM OF THE REPORT JUST BEFORE DISCLAIMER (SANSKRIT + ENGLISH + EXPLANATION ONLY) ---
            disc_text = (
                "<b>अस्वीकरण (Disclaimer):</b> यह जिह्वा परीक्षा रिपोर्ट आयुर्वेदिक स्वास्थ्य जागरूकता एवं प्राथमिक मूल्यांकन के लिए है। किसी भी गंभीर स्वास्थ्य समस्या के लिए योग्य आयुर्वेदिक वैद्य से परामर्श अवश्य लें।"
                if is_hi_mode else
                "<b>Clinical Disclaimer:</b> This Jihva Pariksha report is generated for Ayurvedic educational and wellness assessment purposes. It does not replace pulse diagnosis (Nadi Pariksha) or direct consultation with a qualified Ayurvedic Vaidya."
            )

            st.markdown(f"""
            <div class="shloka-box">
                <div style="font-size:0.84rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:#92400e;">
                    🕉️ Ancient Ayurvedic Scripture — {active_quote['source']}
                </div>
                <div style="font-size:1.15rem; font-weight:700; margin-top:4px; color:#78350f;">
                    {active_quote['sanskrit']} <span style="font-size:0.9rem; font-weight:400; font-style:italic;">({active_quote['transliteration']})</span>
                </div>
                <div style="font-size:0.95rem; color:#1e293b; margin-top:6px;">
                    <b>English Meaning:</b> "{active_quote['english']}"<br>
                    <span style="font-size:0.92rem; color:#78350f;"><b>Ayurvedic Explanation:</b> {active_quote['explanation']}</span>
                </div>
            </div>
            <div class="disclaimer-box">
                {disc_text}
            </div>
            """, unsafe_allow_html=True)

            # --- COMPACT PDF-ONLY DOWNLOAD BUTTON ---
            st.markdown("#### " + ("📥 रिपोर्ट डाउनलोड करें (PDF)" if is_hi_mode else "📥 Download Official Report (PDF)"))
            pdf_bytes = generate_report_pdf_bytes(
                clean_name, patient_age, patient_gender, report_id, raw_analysis, active_quote, lang_mode=report_lang
            )
            lang_tag = "Hindi" if is_hi_mode else "English"
            st.download_button(
                label="📄 PDF रिपोर्ट डाउनलोड करें" if is_hi_mode else "📄 Download Report as PDF",
                data=pdf_bytes,
                file_name=f"Jihva_Pariksha_Report_{report_id}_{lang_tag}.pdf",
                mime="application/pdf"
            )

        # --- PATIENT REVIEWS ---
        with st.container(border=True):
            st.subheader("⭐ Patient Reviews")
            r_col1, r_col2 = st.columns([1, 1.2])
            with r_col1:
                with st.form("patient_review_form"):
                    rev_name = st.text_input("Patient Name", value=clean_name)
                    rev_rating = st.slider("Rating (Stars)", min_value=1, max_value=5, value=5)
                    rev_comment = st.text_area(
                        "Your Review",
                        placeholder="Write your review here..."
                    )
                    sub_rev = st.form_submit_button("🌟 Submit Review", type="primary")
                    if sub_rev:
                        ok = save_patient_review(
                            name=rev_name or clean_name,
                            age=patient_age,
                            gender=patient_gender,
                            rating=rev_rating,
                            dominant_dosha=raw_analysis["dominant_dosha"],
                            review_text=rev_comment
                        )
                        if ok:
                            st.success("Thank you! Your review has been published.")
                        else:
                            st.warning("Please write a short review before submitting.")

            with r_col2:
                st.markdown("#### 💬 Recent Reviews")
                reviews = get_patient_reviews(limit=6)
                if not reviews:
                    st.caption("No patient reviews yet. Be the first to add one!")
                for rv in reviews:
                    stars = "⭐" * int(rv["rating"])
                    st.markdown(
                        f"**{rv['name']}** ({rv['age']}y, {rv['gender']}) — {stars}  \n"
                        f"<span style='font-size:0.82rem; color:#475569;'>{rv['created_at']} • Dosha Indication: {rv['dominant_dosha']}</span>  \n"
                        f"{rv['review_text']}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
