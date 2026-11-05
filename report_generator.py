"""
Fixed-Format Diagnostic Report Generator (PNG Image & PDF Export).
Produces a standardized hospital/clinical style Jihva Pariksha report containing:
1. Header & Report ID / Timestamp
2. Patient Demographics (Name, Age, Gender, Vaya-Avastha Stage)
3. Ancient Ayurvedic Shloka (Sanskrit Devanagari / Transliteration + English Meaning)
4. Uploaded Tongue Photo + Segmented / Coating Visual Overlays
5. Fixed 7-Row Parameter Table (Colour, Coating, Coating Area, Fissures, Shape, Tooth Marks, Moisture)
6. Dosha & Ama Assessment + Regional Organ Map
7. Classical Prescription / Dinacharya / Pathya-Apathya Remedies
8. Educational Disclaimer
"""

import io
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any


def _load_fonts():
    """Loads clean system TrueType fonts on macOS/Linux with safe fallback."""
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    deva_candidates = [
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/System/Library/Fonts/KohinoorDevanagari.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ]

    def pick_font(paths, size):
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()

    return {
        "title": pick_font(bold_candidates, 28),
        "subtitle": pick_font(font_candidates, 16),
        "h2": pick_font(bold_candidates, 20),
        "body_bold": pick_font(bold_candidates, 16),
        "body": pick_font(font_candidates, 16),
        "small": pick_font(font_candidates, 13),
        "deva": pick_font(deva_candidates + font_candidates, 18),
    }


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font, max_width: int) -> list:
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


def render_report_image(
    patient_name: str,
    age: int,
    gender: str,
    report_id: str,
    analysis: Dict[str, Any],
    quote: Dict[str, str]
) -> Image.Image:
    """
    Renders a high-resolution, fixed-format diagnostic report card as a PIL RGB Image.
    Can be downloaded directly as PNG or converted into a single-page clinical PDF.
    """
    W, H = 1080, 1620
    canvas = Image.new("RGB", (W, H), (250, 248, 243))
    draw = ImageDraw.Draw(canvas)
    fonts = _load_fonts()

    # 1. Top Header Banner (Deep Ayurvedic Emerald)
    draw.rectangle([(0, 0), (W, 118)], fill=(19, 78, 58))
    draw.text((40, 24), "AI JIHVA PARIKSHA — AYURVEDIC TONGUE DIAGNOSTIC REPORT", fill=(255, 255, 255), font=fonts["title"])
    draw.text(
        (40, 68),
        "Standardized Ashtasthana Pariksha Visual Assessment • Deterministic OpenCV Engine",
        fill=(209, 250, 229),
        font=fonts["subtitle"]
    )
    now_str = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    draw.text((W - 330, 32), f"Report ID: {report_id}", fill=(254, 243, 199), font=fonts["body_bold"])
    draw.text((W - 330, 62), f"Date: {now_str}", fill=(229, 231, 235), font=fonts["small"])

    # 2. Ancient Ayurvedic Scripture Quote Box (Sanskrit + Transliteration + English)
    y = 134
    draw.rounded_rectangle([(35, y), (W - 35, y + 124)], radius=10, fill=(254, 249, 231), outline=(217, 119, 6), width=2)
    draw.text((55, y + 12), f"ANCIENT AYURVEDIC SCRIPTURE — {quote['source']}", fill=(146, 64, 14), font=fonts["body_bold"])
    draw.text((55, y + 38), f"Sanskrit: {quote['sanskrit']}  ({quote['transliteration']})", fill=(120, 53, 15), font=fonts["deva"])
    eng_lines = _wrap_text(f"Meaning: \"{quote['english']}\"", draw, fonts["body"], W - 110)
    for idx, line in enumerate(eng_lines[:2]):
        draw.text((55, y + 70 + idx * 22), line, fill=(69, 26, 3), font=fonts["body"])

    # 3. Fixed Patient Demographics Strip
    y = 274
    draw.rounded_rectangle([(35, y), (W - 35, y + 92)], radius=8, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
    p_name = (patient_name or "Walk-in Patient").strip()
    draw.text((55, y + 14), f"Patient Name: {p_name}", fill=(17, 24, 39), font=fonts["body_bold"])
    draw.text((430, y + 14), f"Age: {age} Years", fill=(17, 24, 39), font=fonts["body_bold"])
    draw.text((640, y + 14), f"Gender: {gender}", fill=(17, 24, 39), font=fonts["body_bold"])
    draw.text((830, y + 14), f"Ama State: {analysis['ama_status'].split('(')[0].strip()}", fill=(180, 83, 9), font=fonts["body_bold"])

    ctx_lines = _wrap_text(f"Age/Gender Clinical Context: {analysis['age_gender_context']}", draw, fonts["small"], W - 110)
    for idx, line in enumerate(ctx_lines[:2]):
        draw.text((55, y + 46 + idx * 18), line, fill=(75, 85, 99), font=fonts["small"])

    # 4. Tongue Visual Panel (Left) + Fixed 7-Row Parameter Table (Right)
    y = 382
    # Left Card: Tongue Images
    draw.rounded_rectangle([(35, y), (455, y + 460)], radius=10, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
    draw.text((55, y + 14), "CAPTURED TONGUE & AI OVERLAYS", fill=(19, 78, 58), font=fonts["h2"])

    orig_rgb = cv2.cvtColor(analysis["images"]["original_wb"], cv2.COLOR_BGR2RGB)
    coat_rgb = cv2.cvtColor(analysis["images"]["coating_overlay"], cv2.COLOR_BGR2RGB)
    fiss_rgb = cv2.cvtColor(analysis["images"]["fissure_overlay"], cv2.COLOR_BGR2RGB)

    pil_orig = Image.fromarray(orig_rgb).resize((380, 230))
    pil_coat = Image.fromarray(coat_rgb).resize((184, 130))
    pil_fiss = Image.fromarray(fiss_rgb).resize((184, 130))

    canvas.paste(pil_orig, (55, y + 48))
    canvas.paste(pil_coat, (55, y + 294))
    canvas.paste(pil_fiss, (251, y + 294))
    draw.text((55, y + 430), "Left: Coating Map (Cyan)   |   Right: Fissure Map (Red)", fill=(107, 114, 128), font=fonts["small"])

    # Right Card: Fixed Diagnostic Parameter Table (matches screenshot layout)
    draw.rounded_rectangle([(475, y), (W - 35, y + 460)], radius=10, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
    draw.text((495, y + 14), "STANDARDIZED JIHVA PARIKSHA OBSERVATIONS", fill=(19, 78, 58), font=fonts["h2"])

    table_y = y + 52
    draw.rectangle([(495, table_y), (W - 55, table_y + 38)], fill=(243, 244, 246))
    draw.text((510, table_y + 9), "Parameter", fill=(17, 24, 39), font=fonts["body_bold"])
    draw.text((730, table_y + 9), "AI Observation", fill=(17, 24, 39), font=fonts["body_bold"])

    params = analysis["parameters"]
    fixed_rows = [
        ("Colour", params["Colour_Full"]),
        ("Coating (Lepa)", params["Coating"]),
        ("Coating area", f"{params['Coating_Area']} ({params['Coating_Type']})"),
        ("Fissures (Sphutita)", params["Fissures"]),
        ("Shape (Akriti)", params["Shape_State"]),
        ("Tooth marks", params["Tooth_Marks"]),
        ("Moisture (Snigdhata)", params["Moisture"]),
        ("Dominant Dosha", analysis["dominant_dosha"]),
    ]

    row_h = 44
    for i, (k_label, v_val) in enumerate(fixed_rows):
        ry = table_y + 38 + i * row_h
        if i % 2 == 1:
            draw.rectangle([(495, ry), (W - 55, ry + row_h)], fill=(249, 250, 251))
        draw.line([(495, ry + row_h), (W - 55, ry + row_h)], fill=(229, 231, 235), width=1)
        draw.text((510, ry + 11), k_label, fill=(55, 65, 81), font=fonts["body_bold"])
        val_color = (19, 78, 58) if k_label == "Dominant Dosha" else (17, 24, 39)
        draw.text((730, ry + 11), str(v_val)[:36], fill=val_color, font=fonts["body_bold" if k_label == "Dominant Dosha" else "body"])

    # 5. Tridosha Balance & Regional Organ Map Strip
    y = 858
    draw.rounded_rectangle([(35, y), (W - 35, y + 148)], radius=10, fill=(255, 255, 255), outline=(209, 213, 219), width=2)
    draw.text((55, y + 14), "TRIDOSHA QUANTIFICATION & REGIONAL TONGUE ORGAN ZONES", fill=(19, 78, 58), font=fonts["h2"])

    d_scores = analysis["dosha_scores"]
    dosha_str = f"Vata: {d_scores['Vata']}%   |   Pitta: {d_scores['Pitta']}%   |   Kapha: {d_scores['Kapha']}%   -->   Conclusion: {analysis['dominant_dosha']}"
    draw.text((55, y + 48), dosha_str, fill=(180, 83, 9), font=fonts["body_bold"])

    z_y = y + 80
    for z_name, z_val in analysis["zones"].items():
        draw.text((55, z_y), f"• {z_name}: {z_val}", fill=(55, 65, 81), font=fonts["body"])
        z_y += 22

    # 6. Ayurvedic Suggestions, Prescription & Pathya-Apathya Remedies
    y = 1022
    draw.rounded_rectangle([(35, y), (W - 35, y + 495)], radius=10, fill=(255, 255, 255), outline=(19, 78, 58), width=2)
    draw.rectangle([(35, y), (W - 35, y + 42)], fill=(19, 78, 58))
    draw.text((55, y + 10), "AYURVEDIC SUGGESTIONS, CLASSICAL REMEDIES & PATHYA-APATHYA (DIET)", fill=(255, 255, 255), font=fonts["h2"])

    rem = analysis["remedies"]
    curr_y = y + 56

    sections = [
        ("1. Daily Tongue & Oral Regimen (Dinacharya):", rem["dinacharya"], (19, 78, 58)),
        ("2. Classical Domestic Herbal Formulations (Aushadha / Upachara):", rem["herbal_remedies"], (146, 64, 14)),
        ("3. Wholesome Diet & Lifestyle (Pathya — Recommended):", rem["pathya_diet"], (21, 128, 61)),
        ("4. Diet & Habits to Avoid (Apathya — Restrict):", rem["apathya_avoid"], (185, 28, 28)),
    ]

    for sec_title, items, col in sections:
        draw.text((55, curr_y), sec_title, fill=col, font=fonts["body_bold"])
        curr_y += 24
        for item in items:
            wrapped = _wrap_text(f"• {item}", draw, fonts["body"], W - 115)
            for line in wrapped:
                draw.text((70, curr_y), line, fill=(31, 41, 55), font=fonts["body"])
                curr_y += 21
        curr_y += 8

    # 7. Footer Disclaimer
    y = 1532
    draw.rectangle([(0, y), (W, H)], fill=(243, 244, 246))
    disc = (
        "CLINICAL DISCLAIMER: This Jihva Pariksha report is generated by a deterministic rule-based computer vision system for educational "
        "and wellness screening purposes. It does not replace pulse examination (Nadi Pariksha) or consultation with a registered Ayurvedic Vaidya."
    )
    for i, line in enumerate(_wrap_text(disc, draw, fonts["small"], W - 80)):
        draw.text((40, y + 18 + i * 18), line, fill=(107, 114, 128), font=fonts["small"])

    return canvas


def generate_report_png_bytes(
    patient_name: str,
    age: int,
    gender: str,
    report_id: str,
    analysis: Dict[str, Any],
    quote: Dict[str, str]
) -> bytes:
    img = render_report_image(patient_name, age, gender, report_id, analysis, quote)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_report_pdf_bytes(
    patient_name: str,
    age: int,
    gender: str,
    report_id: str,
    analysis: Dict[str, Any],
    quote: Dict[str, str]
) -> bytes:
    """
    Exports the exact fixed-format diagnostic report card as a crisp A4 PDF document
    using PIL's built-in PDF writer (zero external font-encoding issues, preserves Devanagari Sanskrit & images).
    """
    img = render_report_image(patient_name, age, gender, report_id, analysis, quote)
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=150.0, title=f"Jihva_Pariksha_Report_{report_id}")
    return buf.getvalue()
