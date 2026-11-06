"""
Native Vector PDF Report Generator using fpdf2 + uharfbuzz + Google NotoSansDevanagari.
Fixes all Hindi Devanagari matras, half-consonant conjuncts (संयुक्ताक्षर), and alignment in the downloaded PDF!
Why PIL had issues before:
- Pillow's default ImageDraw lacks `libraqm`/`harfbuzz`, which causes Hindi matras (ि, े, ो, ्र, ्) to split or misalign.
- By using `fpdf2` with `pdf.set_text_shaping(True)` powered by `uharfbuzz` and `NotoSansDevanagari`,
  every Hindi and Sanskrit word in the PDF is shaped with 100% typographical accuracy, matching the browser!
"""

import os
import io
import tempfile
import datetime
import cv2
from PIL import Image
from fpdf import FPDF
from typing import Dict, Any


FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
REG_FONT_PATH = os.path.join(FONT_DIR, "KohinoorDevanagari-Regular.ttf")
BOLD_FONT_PATH = os.path.join(FONT_DIR, "KohinoorDevanagari-Bold.ttf")


def _ensure_fonts_exist():
    """Ensures KohinoorDevanagari (or Hind Latin+Devanagari) Regular & Bold TTF fonts exist locally or on cloud."""
    os.makedirs(FONT_DIR, exist_ok=True)
    if (not os.path.exists(REG_FONT_PATH) or not os.path.exists(BOLD_FONT_PATH)) and os.path.exists("/System/Library/Fonts/Kohinoor.ttc"):
        try:
            from fontTools.ttLib import TTCollection
            ttc = TTCollection("/System/Library/Fonts/Kohinoor.ttc")
            ttc.fonts[0].save(REG_FONT_PATH)
            ttc.fonts[3].save(BOLD_FONT_PATH)
        except Exception:
            pass
    if not os.path.exists(REG_FONT_PATH) or not os.path.exists(BOLD_FONT_PATH):
        try:
            import urllib.request
            if not os.path.exists(REG_FONT_PATH):
                urllib.request.urlretrieve(
                    "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Regular.ttf",
                    REG_FONT_PATH
                )
            if not os.path.exists(BOLD_FONT_PATH):
                urllib.request.urlretrieve(
                    "https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf",
                    BOLD_FONT_PATH
                )
        except Exception:
            pass


def _bgr_to_temp_png(img_bgr, target_size=(340, 210)) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize(target_size)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


def generate_report_pdf_bytes(
    patient_name: str,
    age: int,
    gender: str,
    report_id: str,
    analysis: Dict[str, Any],
    quote: Dict[str, str],
    lang_mode: str = "English"
) -> bytes:
    """
    Generates a crisp, single-page A4 Vector PDF Report with HarfBuzz complex-script shaping enabled
    (`pdf.set_text_shaping(True)`), ensuring flawless Hindi (हिंदी) and Sanskrit Devanagari typography & alignment.
    """
    _ensure_fonts_exist()
    is_hi = ("Hindi" in lang_mode) or ("हिंदी" in lang_mode)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Register Unicode Devanagari + Latin font with HarfBuzz text shaping
    pdf.add_font("Noto", "", REG_FONT_PATH)
    pdf.add_font("Noto", "B", BOLD_FONT_PATH)
    pdf.set_text_shaping(True)

    # Soft ivory page background
    pdf.set_fill_color(253, 252, 248)
    pdf.rect(0, 0, 210, 297, style="F")

    # 1. Top Header Banner (Compact Royal Saffron-Gold & Sandalwood Header)
    pdf.set_fill_color(254, 243, 199)
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.5)
    pdf.rect(8, 7, 194, 19, style="DF")

    pdf.set_xy(12, 9)
    pdf.set_text_color(120, 53, 15)
    pdf.set_font("Noto", "B", 13)
    if is_hi:
        pdf.cell(125, 6, "जिह्वा परीक्षा निदान रिपोर्ट (Jihva Pariksha Report)", new_x="RIGHT", new_y="TOP")
    else:
        pdf.cell(125, 6, "JIHVA PARIKSHA DIAGNOSTIC REPORT", new_x="RIGHT", new_y="TOP")

    now_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    pdf.set_font("Noto", "B", 8.5)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(65, 6, f"ID: {report_id}  |  {now_str}", align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(12)
    pdf.set_font("Noto", "", 8.5)
    pdf.set_text_color(146, 64, 14)
    if is_hi:
        pdf.cell(186, 5, "अष्टस्थान परीक्षा — आयुर्वेदिक जीभ परीक्षण, त्रिदोष संतुलन एवं उपचार", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(186, 5, "Classical Ashtasthana Pariksha — Ayurvedic Tongue Analysis, Tridosha Balance & Remedies", new_x="LMARGIN", new_y="NEXT")

    # 2. Patient Details Strip
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(8, 28, 194, 17, style="DF")

    pdf.set_xy(12, 29.5)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Noto", "B", 9.5)
    p_name = patient_name.strip()
    if is_hi:
        gender_hi = {"Male": "पुरुष", "Female": "महिला", "Other": "अन्य"}.get(gender, gender)
        info_line = f"रोगी का नाम: {p_name}     |     आयु: {age} वर्ष     |     लिंग: {gender_hi}     |     प्रधान दोष: {analysis['dominant_dosha_hi']}"
        ctx_line = f"वय परीक्षा संदर्भ: {analysis['age_gender_context_hi']}"
    else:
        info_line = f"Patient Name: {p_name}     |     Age: {age} Years     |     Gender: {gender}     |     Dominant Dosha: {analysis['dominant_dosha']}"
        ctx_line = f"Age/Gender Context: {analysis['age_gender_context']}"

    pdf.cell(186, 5.5, info_line, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(12)
    pdf.set_font("Noto", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(186, 4.2, ctx_line)

    # 3. Left Column: Tongue Visual Maps  |  Right Column: Fixed 8-Row Parameter Table
    y_mid = 47
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(8, y_mid, 78, 78, style="DF")
    pdf.rect(88, y_mid, 114, 78, style="DF")

    # Save temporary images and embed in Left Box
    t_orig = _bgr_to_temp_png(analysis["images"]["original_wb"], (320, 195))
    t_coat = _bgr_to_temp_png(analysis["images"]["coating_overlay"], (160, 110))
    t_fiss = _bgr_to_temp_png(analysis["images"]["fissure_overlay"], (160, 110))
    try:
        pdf.set_xy(11, y_mid + 2)
        pdf.set_font("Noto", "B", 9)
        pdf.set_text_color(21, 128, 61)
        pdf.cell(72, 5, "जीभ का चित्र एवं मैप" if is_hi else "UPLOADED TONGUE & FEATURE MAPS")
        pdf.image(t_orig, x=11, y=y_mid + 8, w=72, h=40)
        pdf.image(t_coat, x=11, y=y_mid + 50, w=35, h=21)
        pdf.image(t_fiss, x=48, y=y_mid + 50, w=35, h=21)
        pdf.set_xy(11, y_mid + 72)
        pdf.set_font("Noto", "", 7)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(72, 4, "Left: Coating Map (Cyan)  |  Right: Fissure Map (Red)")
    finally:
        for tp in [t_orig, t_coat, t_fiss]:
            try:
                os.remove(tp)
            except Exception:
                pass

    # Right Box: Aligned Fixed Parameter Table
    pdf.set_xy(91, y_mid + 2)
    pdf.set_font("Noto", "B", 9.5)
    pdf.set_text_color(21, 128, 61)
    pdf.cell(108, 5.5, "जिह्वा परीक्षा के मुख्य अवलोकन" if is_hi else "JIHVA PARIKSHA OBSERVATIONS", new_x="LMARGIN", new_y="NEXT")

    # Table Header
    th_y = y_mid + 8.5
    pdf.set_fill_color(120, 53, 15)
    pdf.rect(91, th_y, 108, 7, style="F")
    pdf.set_xy(93, th_y + 1)
    pdf.set_font("Noto", "B", 8.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(42, 5, "मापदंड (Parameter)" if is_hi else "Parameter")
    pdf.cell(64, 5, "निरीक्षण (Observation)" if is_hi else "Observation")

    params = analysis["parameters"]
    if is_hi:
        rows = [
            ("रंग (Colour)", params["Colour_Hi"]),
            ("परत (Coating)", params["Coating_Hi"]),
            ("परत का क्षेत्र (Area)", f"{params['Coating_Area']} ({params['Coating_Type_Hi']})"),
            ("दरारें (Fissures)", params["Fissures_Hi"]),
            ("आकार (Shape)", params["Shape_Hi"]),
            ("दांतों के निशान", params["Tooth_Marks_Hi"]),
            ("नमी (Moisture)", params["Moisture_Hi"]),
            ("प्रधान दोष (Dosha)", analysis["dominant_dosha_hi"]),
        ]
    else:
        rows = [
            ("Colour (Varna)", params["Colour_Full"]),
            ("Coating (Lepa)", params["Coating"]),
            ("Coating area", f"{params['Coating_Area']} ({params['Coating_Type']})"),
            ("Fissures (Sphutita)", params["Fissures"]),
            ("Shape (Akriti)", params["Shape_State"]),
            ("Tooth marks", params["Tooth_Marks"]),
            ("Moisture (Snigdhata)", params["Moisture"]),
            ("Dominant Dosha", analysis["dominant_dosha"]),
        ]

    row_h = 7.4
    for idx, (k_lbl, v_val) in enumerate(rows):
        ry = th_y + 7 + idx * row_h
        if idx % 2 == 1:
            pdf.set_fill_color(248, 250, 252)
            pdf.rect(91, ry, 108, row_h, style="F")
        pdf.set_draw_color(226, 232, 240)
        pdf.line(91, ry + row_h, 199, ry + row_h)

        pdf.set_xy(93, ry + 1.2)
        pdf.set_font("Noto", "B", 8.2)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(42, 5, k_lbl)

        pdf.set_xy(135, ry + 1.2)
        pdf.set_font("Noto", "B" if idx == 7 else "", 8.2)
        if idx == 7:
            pdf.set_text_color(21, 128, 61)
        else:
            pdf.set_text_color(15, 23, 42)
        pdf.cell(63, 5, str(v_val)[:42])

    # 4. Tridosha Balance & Regional Organ Map Box
    y_zone = 127.5
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(8, y_zone, 194, 27, style="DF")

    pdf.set_xy(12, y_zone + 2)
    pdf.set_font("Noto", "B", 8.2)
    pdf.set_text_color(21, 128, 61)
    d_scores = analysis["dosha_scores"]
    if is_hi:
        pdf.cell(
            186, 5,
            f"त्रिदोष संतुलन: वात (Vata): {d_scores['Vata']}% | पित्त (Pitta): {d_scores['Pitta']}% | कफ (Kapha): {d_scores['Kapha']}%  |  अग्नि/आम: {analysis['ama_status_hi']}",
            new_x="LMARGIN", new_y="NEXT"
        )
    else:
        pdf.cell(
            186, 5,
            f"Tridosha Balance: Vata: {d_scores['Vata']}% | Pitta: {d_scores['Pitta']}% | Kapha: {d_scores['Kapha']}%  |  Agni/Ama: {analysis['ama_status']}",
            new_x="LMARGIN", new_y="NEXT"
        )

    active_zones = analysis["zones_hi"] if is_hi else analysis["zones"]
    pdf.set_font("Noto", "", 8.2)
    pdf.set_text_color(30, 41, 59)
    for z_k, z_v in active_zones.items():
        pdf.set_x(12)
        pdf.cell(186, 5.2, f"• {z_k}: {z_v}", new_x="LMARGIN", new_y="NEXT")

    # 5. Ayurvedic Prescription, Herbal Remedies & Pathya-Apathya Box
    y_rem = 156.5
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(21, 128, 61)
    pdf.rect(8, y_rem, 194, 83, style="DF")

    pdf.set_fill_color(21, 128, 61)
    pdf.rect(8, y_rem, 194, 7.5, style="F")
    pdf.set_xy(12, y_rem + 1.2)
    pdf.set_font("Noto", "B", 9.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(
        186, 5,
        "आयुर्वेदिक उपचार, परामर्श एवं पथ्य-अपथ्य (आहार)" if is_hi else "AYURVEDIC PRESCRIPTION, REMEDIES & PATHYA-APATHYA (DIET)",
        new_x="LMARGIN", new_y="NEXT"
    )

    rem = analysis["remedies_hi"] if is_hi else analysis["remedies"]
    if is_hi:
        sec_blocks = [
            ("1. दैनिक मुख एवं जिह्वा दिनचर्या (Dinacharya):", rem["dinacharya"], (21, 128, 61)),
            ("2. आयुर्वेदिक हर्बल उपचार (Upachara):", rem["herbal_remedies"], (146, 64, 14)),
            ("3. पथ्य आहार — क्या खाएं (Eat):", rem["pathya_diet"], (21, 128, 61)),
            ("4. अपथ्य — परहेज करें (Avoid):", rem["apathya_avoid"], (185, 28, 28)),
        ]
    else:
        sec_blocks = [
            ("1. Daily Oral Regimen (Dinacharya):", rem["dinacharya"], (21, 128, 61)),
            ("2. Herbal Remedies & Formulations (Upachara):", rem["herbal_remedies"], (146, 64, 14)),
            ("3. Wholesome Diet (Pathya — Recommended):", rem["pathya_diet"], (21, 128, 61)),
            ("4. Diet & Habits to Avoid (Apathya — Restrict):", rem["apathya_avoid"], (185, 28, 28)),
        ]

    pdf.set_y(y_rem + 9)
    for sec_title, items, col_rgb in sec_blocks:
        pdf.set_x(12)
        pdf.set_font("Noto", "B", 8.5)
        pdf.set_text_color(*col_rgb)
        pdf.cell(186, 4.6, sec_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Noto", "", 8.0)
        pdf.set_text_color(15, 23, 42)
        for item in items:
            pdf.set_x(14)
            pdf.multi_cell(182, 4.1, f"• {item}")

    # 6. Ancient Ayurvedic Scripture Quote Box (Strictly Sanskrit + English + Explanation right before Disclaimer!)
    y_q = 241.5
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(217, 119, 6)
    pdf.rect(8, y_q, 194, 34, style="DF")

    pdf.set_xy(12, y_q + 2)
    pdf.set_font("Noto", "B", 8.5)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(186, 4.5, f"ANCIENT AYURVEDIC SCRIPTURE — {quote['source']}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(12)
    pdf.set_font("Noto", "B", 9.0)
    pdf.set_text_color(120, 53, 15)
    pdf.multi_cell(186, 4.5, f"{quote['sanskrit']}  ({quote['transliteration']})")

    pdf.set_x(12)
    pdf.set_font("Noto", "", 7.8)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(186, 4.0, f"English Meaning: \"{quote['english']}\"")

    pdf.set_x(12)
    pdf.set_font("Noto", "", 7.6)
    pdf.set_text_color(120, 53, 15)
    pdf.multi_cell(186, 3.9, f"Ayurvedic Explanation: {quote.get('explanation', '')}")

    # 7. Footer Clinical Disclaimer
    y_disc = 277.5
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(8, y_disc, 194, 14, style="DF")
    pdf.set_xy(12, y_disc + 2)
    pdf.set_font("Noto", "", 7.5)
    pdf.set_text_color(71, 85, 105)
    if is_hi:
        disc_str = (
            "अस्वीकरण (Disclaimer): यह जिह्वा परीक्षा रिपोर्ट आयुर्वेदिक स्वास्थ्य जागरूकता एवं प्राथमिक मूल्यांकन के लिए है। "
            "किसी भी गंभीर स्वास्थ्य समस्या के लिए योग्य आयुर्वेदिक वैद्य से परामर्श अवश्य लें।"
        )
    else:
        disc_str = (
            "CLINICAL DISCLAIMER: This Jihva Pariksha report is generated for Ayurvedic educational and wellness assessment purposes. "
            "It does not replace pulse diagnosis (Nadi Pariksha) or direct consultation with a qualified Ayurvedic Vaidya."
        )
    pdf.multi_cell(186, 4.0, disc_str)

    return bytes(pdf.output())
