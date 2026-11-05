"""
AI Jihva Pariksha — Ayurvedic Tongue Diagnosis Web Application
Built with Streamlit + OpenCV + Zero-Hallucination Rule Engine + Sanitized Feedback Calibration.
"""

import os
import hashlib
import urllib.parse
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from ayurveda_quotes import get_quote_for_report, AYURVEDIC_QUOTES
from tongue_analyzer import (
    analyze_tongue_image,
    ALLOWED_COLOURS,
    ALLOWED_COATING_STATES,
    ALLOWED_FISSURE_STATES,
    ALLOWED_DOSHAS,
)
from feedback_store import (
    init_db,
    save_training_feedback,
    predict_with_feedback_calibration,
    get_feedback_stats,
    save_patient_review,
    get_patient_reviews,
)
from report_generator import generate_report_png_bytes, generate_report_pdf_bytes


st.set_page_config(
    page_title="AI Jihva Pariksha — Ayurvedic Tongue Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite tables
init_db()

# Custom CSS for clean clinical Ayurvedic report styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #134e3a 0%, #0f3d2e 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .shloka-box {
        background-color: #fef9e7;
        border-left: 5px solid #d97706;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        color: #451a03;
    }
    .report-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .fixed-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.96rem;
    }
    .fixed-table th {
        background-color: #134e3a;
        color: #ffffff;
        text-align: left;
        padding: 10px 14px;
    }
    .fixed-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #e5e7eb;
        color: #111827;
    }
    .fixed-table tr:nth-child(even) {
        background-color: #f9fafb;
    }
    .badge-anti-hallucination {
        display: inline-block;
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def generate_synthetic_sample_tongue(preset: str = "Coated White (Kapha-Ama)") -> np.ndarray:
    """Creates a realistic test tongue image so anyone can test the app even before uploading a photo."""
    img = np.full((420, 520, 3), (195, 205, 225), dtype=np.uint8)
    # Oral cavity dark background
    cv2.ellipse(img, (260, 215), (175, 165), 0, 0, 360, (45, 35, 65), -1)

    if preset == "Coated White (Kapha-Ama)":
        body_col = (145, 135, 225)  # Pinkish BGR
        cv2.ellipse(img, (260, 225), (135, 155), 0, 0, 360, body_col, -1)
        # White coating in center
        coat_mask = np.zeros((420, 520), dtype=np.uint8)
        cv2.ellipse(coat_mask, (260, 215), (92, 118), 0, 0, 360, 255, -1)
        noise = np.random.randint(-12, 12, (420, 520, 3), dtype=np.int16)
        coated_bgr = np.clip(np.full_like(img, (218, 222, 232), dtype=np.int16) + noise, 0, 255).astype(np.uint8)
        img[coat_mask > 0] = cv2.addWeighted(img[coat_mask > 0], 0.25, coated_bgr[coat_mask > 0], 0.75, 0)
        # Mild central fissure
        cv2.line(img, (260, 125), (260, 310), (95, 85, 145), 3)

    elif preset == "Fissured & Dry (Vata Predominant)":
        body_col = (135, 120, 195)
        cv2.ellipse(img, (260, 225), (115, 160), 0, 0, 360, body_col, -1)
        # Multiple fissures
        cv2.line(img, (260, 110), (260, 335), (60, 45, 95), 4)
        for fy in range(145, 315, 28):
            cv2.line(img, (200, fy), (258, fy + 10), (65, 50, 100), 3)
            cv2.line(img, (262, fy + 8), (320, fy - 4), (65, 50, 100), 3)

    else:  # Red & Yellowish Coating (Pitta Predominant)
        body_col = (105, 95, 238)  # Deep red BGR
        cv2.ellipse(img, (260, 225), (130, 152), 0, 0, 360, body_col, -1)
        coat_mask = np.zeros((420, 520), dtype=np.uint8)
        cv2.ellipse(coat_mask, (260, 200), (80, 95), 0, 0, 360, 255, -1)
        img[coat_mask > 0] = (145, 205, 235)  # Yellowish tint in BGR

    return cv2.GaussianBlur(img, (5, 5), 0)


# --- SIDEBAR: PATIENT REGISTRATION & ANTI-HALLUCINATION STATUS ---
with st.sidebar:
    st.markdown("### 👤 1. Patient Details")
    st.caption("Age & gender calibrate Ayurvedic *Vaya Pariksha* (life-stage) analysis.")
    patient_name = st.text_input("Patient Name", value="", placeholder="e.g., Aarav Sharma")
    col_a, col_g = st.columns(2)
    with col_a:
        patient_age = st.number_input("Age (Years)", min_value=1, max_value=110, value=28, step=1)
    with col_g:
        patient_gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

    st.divider()
    st.markdown("### 🛡️ Anti-Hallucination Engine")
    st.markdown('<span class="badge-anti-hallucination">✓ 100% Deterministic Mode</span>', unsafe_allow_html=True)
    st.caption(
        "• **Purely Tongue-Focused**: Non-tongue images are automatically flagged.\n"
        "• **Fixed Diagnostic Schema**: All 7 parameters & Doshas use locked clinical enums.\n"
        "• **Verified Scriptures**: Shlokas are drawn strictly from Charaka, Sushruta, Ashtanga Hridaya & Yogaratnakara.\n"
        "• **Sanitized Feedback DB**: Spam, off-topic text & contradictory outliers are stripped before calibration."
    )

    fb_stats = get_feedback_stats()
    st.metric("Verified Training Feedback Samples", fb_stats["accepted"], delta=f"{fb_stats['rejected']} noisy/irrelevant filtered")

    st.divider()
    st.markdown("### 🕉️ Scripture Quote Selector")
    quote_mode = st.radio(
        "Report Quote Mode",
        options=["Auto-pick from Ancient Scriptures", "Choose Specific Shloka"],
        index=0
    )
    if quote_mode == "Choose Specific Shloka":
        selected_q_idx = st.selectbox(
            "Select Scripture",
            options=range(len(AYURVEDIC_QUOTES)),
            format_func=lambda i: f"{AYURVEDIC_QUOTES[i]['source']}"
        )
    else:
        selected_q_idx = None


# --- MAIN HEADER ---
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; color:white;">🌿 AI Jihva Pariksha — Ayurvedic Tongue Diagnostic System</h2>
    <p style="margin:6px 0 0 0; opacity:0.92; font-size:0.98rem;">
        Upload a tongue photograph for standardized <b>Ashtasthana Pariksha</b> analysis (Colour, Coating %, Fissures, Shape, Tooth Marks, Dosha & Prescription).
    </p>
</div>
""", unsafe_allow_html=True)


# --- STEP 1: UPLOAD OR CAPTURE TONGUE IMAGE ---
st.subheader("📸 Step 1: Upload or Capture Tongue Photograph")
up_col1, up_col2 = st.columns([2, 1])

with up_col1:
    input_method = st.radio(
        "Select Image Source:",
        options=["Upload Tongue Photo (JPG/PNG)", "Use Camera (Take Live Photo)", "Try Built-in Sample Tongue (Instant Demo)"],
        horizontal=True
    )

    img_bgr = None
    image_seed = "default_seed"

    if input_method == "Upload Tongue Photo (JPG/PNG)":
        uploaded_file = st.file_uploader("Upload clear close-up tongue photograph", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            image_seed = hashlib.md5(raw_bytes).hexdigest()[:10]
            file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif input_method == "Use Camera (Take Live Photo)":
        cam_file = st.camera_input("Protrude tongue gently in natural daylight (without flash)")
        if cam_file is not None:
            raw_bytes = cam_file.read()
            image_seed = hashlib.md5(raw_bytes).hexdigest()[:10]
            file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:
        sample_choice = st.selectbox(
            "Select Sample Tongue Condition:",
            options=[
                "Coated White (Kapha-Ama)",
                "Fissured & Dry (Vata Predominant)",
                "Red & Yellowish Coating (Pitta Predominant)"
            ]
        )
        img_bgr = generate_synthetic_sample_tongue(sample_choice)
        image_seed = hashlib.md5(sample_choice.encode("utf-8")).hexdigest()[:10]

with up_col2:
    st.info(
        "**📋 Standard Capture Guidelines**\n"
        "1. Take photo in **morning before brushing/eating**.\n"
        "2. Use **indirect daylight** (avoid coloured LED or harsh flash).\n"
        "3. Relax tongue naturally — do not over-stretch for more than 5 seconds."
    )

# --- PREVIEW UPLOADED IMAGE & SUBMIT BUTTON ---
if "analyzed_seed" not in st.session_state:
    st.session_state.analyzed_seed = None

if img_bgr is not None:
    current_state_key = f"{image_seed}_{patient_age}_{patient_gender}"
    st.markdown("#### 🖼️ Uploaded Tongue Image Preview")
    prev_col1, prev_col2 = st.columns([1, 2])
    with prev_col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded Tongue Photograph", use_container_width=True)
    with prev_col2:
        st.markdown(
            f"**Patient Details Ready:** `{patient_name or 'Walk-in Patient'}` • `{patient_age} Years` • `{patient_gender}`\n\n"
            "Click the **Submit & Analyze Tongue** button below to run the 7-module Ayurvedic *Jihva Pariksha* computer-vision engine and generate your diagnostic report."
        )
        if st.button("🔍 Submit & Analyze Tongue", type="primary", use_container_width=True):
            st.session_state.analyzed_seed = current_state_key

# --- RUN ANALYSIS & DISPLAY FIXED DIAGNOSTIC REPORT AFTER SUBMIT ---
if img_bgr is not None and st.session_state.analyzed_seed == f"{image_seed}_{patient_age}_{patient_gender}":
    report_id = f"JP-{image_seed[:6].upper()}-{patient_age}{patient_gender[0].upper()}"
    raw_analysis = analyze_tongue_image(img_bgr, age=int(patient_age), gender=patient_gender)

    # Calibrate Dosha prediction with accepted feedback from SQLite
    calibration = predict_with_feedback_calibration(
        raw_analysis["feature_vector"],
        raw_analysis["dominant_dosha"],
        raw_analysis["dosha_scores"]
    )
    raw_analysis["dominant_dosha"] = calibration["calibrated_dosha"]
    raw_analysis["dosha_scores"] = calibration["calibrated_scores"]

    # Pick Scripture Quote (Sanskrit + English)
    if selected_q_idx is not None:
        active_quote = AYURVEDIC_QUOTES[selected_q_idx]
    else:
        active_quote = get_quote_for_report(raw_analysis["dominant_dosha"], seed_str=report_id)

    # Quality warning if non-tongue or blurry image
    if raw_analysis["quality"]["quality_warning"]:
        st.warning(f"⚠️ **Image Quality / Gatekeeper Notice:** {raw_analysis['quality']['quality_warning']}")

    st.subheader("🔬 Computer Vision Segmentation & Feature Maps")
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

    st.divider()

    # --- STEP 2: FIXED-FORMAT CLINICAL JIHVA PARIKSHA REPORT ---
    st.subheader("📑 Standardized Jihva Pariksha Diagnostic Report (Fixed Format)")

    # Ancient Ayurvedic Quote Banner (Sanskrit + English)
    st.markdown(f"""
    <div class="shloka-box">
        <div style="font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; color:#92400e;">
            🕉️ Ancient Ayurvedic Scripture — {active_quote['source']}
        </div>
        <div style="font-size:1.18rem; font-weight:700; margin-top:6px; color:#78350f;">
            {active_quote['sanskrit']}
        </div>
        <div style="font-size:0.92rem; font-style:italic; color:#92400e; margin-bottom:6px;">
            {active_quote['transliteration']}
        </div>
        <div style="font-size:0.98rem; color:#451a03;">
            <b>English Meaning:</b> "{active_quote['english']}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fixed Patient Header + Fixed 7-Row Parameter Table (Matches screenshot layout)
    rep_left, rep_right = st.columns([1.15, 1])

    params = raw_analysis["parameters"]
    display_name = (patient_name or "Walk-in Patient").strip()

    with rep_left:
        st.markdown(f"""
        <div class="report-card">
            <div style="display:flex; justify-content:space-between; border-bottom:2px solid #134e3a; padding-bottom:8px; margin-bottom:12px;">
                <div><b>Patient:</b> {display_name} ({patient_age} yrs, {patient_gender})</div>
                <div><b>Report ID:</b> <code>{report_id}</code></div>
            </div>
            <table class="fixed-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>AI Observation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><b>Colour</b></td><td>{params['Colour_Full']}</td></tr>
                    <tr><td><b>Coating</b></td><td>{params['Coating']} ({params['Coating_Type']})</td></tr>
                    <tr><td><b>Coating area</b></td><td>{params['Coating_Area']}</td></tr>
                    <tr><td><b>Fissures</b></td><td>{params['Fissures']}</td></tr>
                    <tr><td><b>Shape</b></td><td>{params['Shape_State']}</td></tr>
                    <tr><td><b>Tooth marks</b></td><td>{params['Tooth_Marks']}</td></tr>
                    <tr><td><b>Moisture</b></td><td>{params['Moisture']}</td></tr>
                    <tr><td><b>Dominant Dosha</b></td><td><b style="color:#134e3a;">{raw_analysis['dominant_dosha']}</b></td></tr>
                    <tr><td><b>Agni / Ama Status</b></td><td>{raw_analysis['ama_status']}</td></tr>
                </tbody>
            </table>
            <p style="font-size:0.84rem; color:#4b5563; margin-top:10px; margin-bottom:0;">
                <b>Age & Gender Assessment (Vaya Pariksha):</b> {raw_analysis['age_gender_context']}<br>
                <b>Model Calibration Status:</b> {calibration['calibration_note']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with rep_right:
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown("#### ⚖️ Tridosha Distribution & Regional Organ Zones")
        d_scores = raw_analysis["dosha_scores"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Vata (Air/Ether)", f"{d_scores['Vata']}%")
        m2.metric("Pitta (Fire/Water)", f"{d_scores['Pitta']}%")
        m3.metric("Kapha (Earth/Water)", f"{d_scores['Kapha']}%")

        st.markdown("**Regional Tongue Organ Mapping (Jihva Sthana):**")
        for zone_label, zone_obs in raw_analysis["zones"].items():
            st.markdown(f"- **{zone_label}:** {zone_obs}")

        st.markdown("---")
        st.markdown("#### 🌿 Classical Ayurvedic Prescription, Remedies & Diet")
        rem = raw_analysis["remedies"]
        st.markdown("**1. Daily Oral Regimen (Dinacharya):**")
        for item in rem["dinacharya"]:
            st.markdown(f"- {item}")
        st.markdown("**2. Herbal Remedies & Formulations (Upachara):**")
        for item in rem["herbal_remedies"]:
            st.markdown(f"- {item}")
        st.markdown("**3. Wholesome Diet (Pathya) & Restrictions (Apathya):**")
        for item in rem["pathya_diet"]:
            st.markdown(f"- ✅ **Eat:** {item}")
        for item in rem["apathya_avoid"]:
            st.markdown(f"- ❌ **Avoid:** {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STEP 3: DOWNLOAD REPORT (PDF & PNG IMAGE) + SHARE OPTIONS ---
    st.subheader("📥 Step 3: Download Official Report (PDF / Image) & Share")
    png_bytes = generate_report_png_bytes(display_name, int(patient_age), patient_gender, report_id, raw_analysis, active_quote)
    pdf_bytes = generate_report_pdf_bytes(display_name, int(patient_age), patient_gender, report_id, raw_analysis, active_quote)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1.2])
    with dl_col1:
        st.download_button(
            label="📄 Download Full Report + Prescription (PDF)",
            data=pdf_bytes,
            file_name=f"Jihva_Pariksha_Report_{report_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="🖼️ Download Report Card (PNG Image)",
            data=png_bytes,
            file_name=f"Jihva_Pariksha_Report_{report_id}.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col3:
        share_summary = (
            f"🌿 *AI Jihva Pariksha Report ({report_id})*\n"
            f"👤 Patient: {display_name} ({patient_age}y, {patient_gender})\n"
            f"🔍 Colour: {params['Colour']} | Coating: {params['Coating_Area']} | Fissures: {params['Fissures']}\n"
            f"⚖️ Dosha Indication: {raw_analysis['dominant_dosha']} ({raw_analysis['ama_status']})\n"
            f"🕉️ Scripture: \"{active_quote['sanskrit']}\" — {active_quote['english']}"
        )
        wa_url = f"https://wa.me/?text={urllib.parse.quote(share_summary)}"
        mail_url = f"mailto:?subject={urllib.parse.quote('Ayurvedic Tongue Analysis Report ' + report_id)}&body={urllib.parse.quote(share_summary)}"
        st.markdown(
            f'<div style="display:flex; gap:10px;">'
            f'<a href="{wa_url}" target="_blank" style="flex:1; text-align:center; background:#25D366; color:white; padding:9px 12px; border-radius:8px; text-decoration:none; font-weight:600;">📲 Share on WhatsApp</a>'
            f'<a href="{mail_url}" style="flex:1; text-align:center; background:#1d4ed8; color:white; padding:9px 12px; border-radius:8px; text-decoration:none; font-weight:600;">✉️ Share via Email</a>'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander("📋 Copy Shareable Text Summary or Preview Printable Report Card"):
        st.code(share_summary, language="markdown")
        st.image(png_bytes, caption="Printable Fixed-Format Report Card Preview", width=540)

    st.divider()

    # --- STEP 4: FEEDBACK & TRAINING CALIBRATION (WITH BACKEND IRRELEVANT DATA FILTER) ---
    fb_tab, rev_tab = st.tabs([
        "🧠 Step 4: Correct / Train Model (Feedback & Anti-Noise Sanitizer)",
        "⭐ Step 5: Patient Reviews & Experience"
    ])

    with fb_tab:
        st.markdown(
            "If an Ayurvedic Vaidya or practitioner observes a different reading on this tongue image, submit the correction below. "
            "**How the backend keeps training clean:**\n"
            "1. **Locked Dropdown Labels**: Corrections use strictly standardized Ayurvedic classes.\n"
            "2. **Biometric Outlier Check**: Rejects trolls/errors that contradict extreme objective pixel values (e.g. claiming 0% coating when >55% is coated).\n"
            "3. **NLP Keyword Sanitizer**: Strips URLs, phone numbers, and non-medical chatter from notes before storing in SQLite."
        )
        with st.form("vaidya_feedback_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                corr_dosha = st.selectbox(
                    "Corrected Dominant Dosha",
                    options=ALLOWED_DOSHAS,
                    index=ALLOWED_DOSHAS.index(raw_analysis["dominant_dosha"]) if raw_analysis["dominant_dosha"] in ALLOWED_DOSHAS else 0
                )
                corr_colour = st.selectbox(
                    "Corrected Tongue Colour",
                    options=ALLOWED_COLOURS,
                    index=ALLOWED_COLOURS.index(params["Colour_Full"]) if params["Colour_Full"] in ALLOWED_COLOURS else 0
                )
            with fc2:
                corr_coating = st.selectbox(
                    "Corrected Coating State",
                    options=ALLOWED_COATING_STATES,
                    index=ALLOWED_COATING_STATES.index(params["Coating_Type"]) if params["Coating_Type"] in ALLOWED_COATING_STATES else 0
                )
                corr_fissures = st.selectbox(
                    "Corrected Fissure Severity",
                    options=ALLOWED_FISSURE_STATES,
                    index=ALLOWED_FISSURE_STATES.index(params["Fissure_State"]) if params["Fissure_State"] in ALLOWED_FISSURE_STATES else 0
                )

            raw_clinical_notes = st.text_area(
                "Vaidya / Practitioner Clinical Notes (Optional — irrelevant sentences are automatically stripped in backend)",
                placeholder="e.g., Patient has white Kapha coating at root with mild Vata fissures along central groove."
            )
            submit_fb = st.form_submit_button("✅ Submit Verified Feedback to Calibrate Model")

            if submit_fb:
                res = save_training_feedback(
                    age=int(patient_age),
                    gender=patient_gender,
                    features=raw_analysis["feature_vector"],
                    predicted_dosha=raw_analysis["dominant_dosha"],
                    corrected_dosha=corr_dosha,
                    corrected_colour=corr_colour,
                    corrected_coating=corr_coating,
                    corrected_fissures=corr_fissures,
                    raw_notes=raw_clinical_notes
                )
                if res["accepted"]:
                    st.success(
                        f"**Feedback Accepted!** (Relevance Score: `{res['relevance_score']}`) — {res['message']}\n\n"
                        f"**Sanitized Backend Notes Stored:** `{res['cleaned_notes'] or 'Structured labels only'}`"
                    )
                else:
                    st.error(f"**Feedback Filtered Out by Backend Gatekeeper:** {res['message']}")

        # Show recent feedback audit table
        stats_now = get_feedback_stats()
        if stats_now["recent"]:
            with st.expander(f"🔍 View Backend Training Store Audit Log ({stats_now['accepted']} accepted, {stats_now['rejected']} rejected)"):
                for r in stats_now["recent"]:
                    status_icon = "✅ Accepted" if r[5] == 1 else "❌ Filtered Out"
                    st.markdown(
                        f"- `{r[0]}` | **{status_icon}** (Relevance `{r[4]}`) | Predicted: *{r[1]}* → Corrected: **{r[2]}** | "
                        f"Sanitized Notes: `{r[3] or '—'}` | _{r[6]}_"
                    )

    with rev_tab:
        st.markdown("Share your experience as a patient or student testing the Jihva Pariksha report.")
        r_col1, r_col2 = st.columns([1, 1.2])
        with r_col1:
            with st.form("patient_review_form"):
                rev_name = st.text_input("Your Name", value=patient_name or "")
                rev_rating = st.slider("Accuracy & Clarity Rating (Stars)", min_value=1, max_value=5, value=5)
                rev_comment = st.text_area(
                    "Your Review",
                    placeholder="The tongue coating map and Ayurvedic dietary recommendations were very clear and matched my Vaidya's advice."
                )
                sub_rev = st.form_submit_button("🌟 Post Patient Review")
                if sub_rev:
                    ok = save_patient_review(
                        name=rev_name,
                        age=int(patient_age),
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
            st.markdown("#### 💬 Recent Patient Reviews")
            reviews = get_patient_reviews(limit=6)
            if not reviews:
                st.caption("No patient reviews yet. Be the first to add one!")
            for rv in reviews:
                stars = "⭐" * int(rv["rating"])
                st.markdown(
                    f"**{rv['name']}** ({rv['age']}y, {rv['gender']}) — {stars}  \n"
                    f"<span style='font-size:0.82rem; color:#6b7280;'>{rv['created_at']} • Dosha Indication: {rv['dominant_dosha']}</span>  \n"
                    f"{rv['review_text']}",
                    unsafe_allow_html=True
                )
                st.markdown("---")
elif img_bgr is not None:
    st.info("👆 Tongue image uploaded! Click **🔍 Submit & Analyze Tongue** above to generate your Ayurvedic diagnostic report.")
else:
    st.info("👆 Please upload a tongue photograph, take a live camera snapshot, or select the **Built-in Sample Tongue** option above.")

