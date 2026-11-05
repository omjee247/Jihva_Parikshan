"""
Deterministic Image Processing & Ayurvedic Tongue Analysis Engine (Zero-Hallucination).
Modules:
- Quality & Tongue Presence Gatekeeper (rejects non-tongue / unreadable images)
- Gray-World White Balance & HSV/LAB Tongue Segmentation
- Module 1: Colour Analysis (Pale, Pinkish, Red, Dark Red/Purple, Bluish)
- Module 2: Coating (Lepa / Ama) Percentage & Colour (Absent, Thin White, Thick White, Yellowish, Brownish)
- Module 3: Fissure & Groove Detection (Central crack, multiple fissures, count & severity)
- Module 4: Shape Geometry (Oval, Long-Thin, Broad-Swollen, Pointed)
- Module 5: Tooth-Mark (Lateral Indentation) Detection
- Module 6: Moisture Estimation (Specular gloss analysis)
- Module 7: Classical Ayurvedic Regional Organ Map (Tip, Center, Root, Lateral Borders)
- Age & Gender Vaya-Pariksha Contextualizer
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


ALLOWED_COLOURS = ["Pinkish (Prakrita / Normal)", "Pale (Pandu / Kapha-Rakta Kshaya)", "Red (Rakta-Pitaja)", "Dark Red / Purplish", "Bluish / Cyanotic"]
ALLOWED_COATING_STATES = ["Absent / Clear (<8%)", "Thin White (Alpa Sama)", "Thick White (Kapha-Ama)", "Yellowish (Pitta-Ama)", "Brownish / Greyish (Vata-Ama)"]
ALLOWED_FISSURE_STATES = ["Not Detected", "Mild Central Groove", "Moderate Fissures Detected", "Deep Multiple Fissures (Sphutita)"]
ALLOWED_SHAPES = ["Oval (Normal / Balanced)", "Broad & Swollen (Kapha-predominant)", "Long & Thin (Vata-predominant)", "Pointed / Narrow Tip (Pitta-Vata)"]
ALLOWED_TOOTH_MARKS = ["Not Detected", "Mild Lateral Indentations", "Clear Tooth Marks Detected (Spleen/Agni Mandya)"]
ALLOWED_MOISTURE = ["Normal Moist (Prakrita)", "Excessively Wet / Salivary (Ardra - Kapha)", "Dry / Rough (Ruksha - Vata)", "Cannot reliably determine"]
ALLOWED_DOSHAS = ["Tridosha Balanced (Sama)", "Vata Predominant", "Pitta Predominant", "Kapha Predominant", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha"]


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Applies Gray-World colour constancy to reduce ambient lighting bias."""
    result = img_bgr.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b > 1e-3 and avg_g > 1e-3 and avg_r > 1e-3:
        result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return result.astype(np.uint8)


def segment_tongue(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Segments the tongue region using central ROI weighting + LAB/HSV mucosal colour priors + morphological cleanup.
    Returns:
        mask (uint8 0/255), contour_overlay (BGR), quality_meta (dict)
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Check blur (Laplacian variance) and brightness
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    # Tongue mucosa & coated tongue dorsal surface typically have higher 'a' channel in CIE LAB
    # OR pink/red/warm-white hues located centrally
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    # Pink/reddish mucosal tissue mask (A > 134 in OpenCV LAB where 128 is neutral)
    red_pink_mask = (A > 133) & (V > 45) & (V < 250)

    # Coated tongue mask (whitish/yellowish central region attached to tongue body)
    coated_mask = (A > 124) & (S < 95) & (V > 120)

    combined = np.where(red_pink_mask | coated_mask, 255, 0).astype(np.uint8)

    # Suppress outer 8% border artifacts (background/clothing)
    border_mask = np.zeros_like(combined)
    pad_y, pad_x = int(h * 0.06), int(w * 0.06)
    border_mask[pad_y:h - pad_y, pad_x:w - pad_x] = 255
    combined = cv2.bitwise_and(combined, border_mask)

    # Morphological closing and opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    best_contour = None

    if contours:
        # Score contours by area and proximity to image center
        img_cx, img_cy = w / 2.0, h / 2.0
        best_score = -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.04):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.hypot((cx - img_cx) / w, (cy - img_cy) / h)
            score = area * max(0.2, 1.0 - dist * 1.4)
            if score > best_score:
                best_score = score
                best_contour = cnt

    if best_contour is not None:
        hull = cv2.convexHull(best_contour)
        # Blend contour with convex hull to preserve lateral indentation info while filling internal holes
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        # Fill internal holes
        cnt_area = cv2.contourArea(best_contour)
        area_ratio = float(cnt_area) / float(h * w)
        # Verify mucosal redness inside contour so random faces/objects are flagged
        mean_a_inside = float(np.mean(A[mask > 0])) if np.any(mask > 0) else 128.0
        is_valid_tongue = (area_ratio >= 0.05) and (mean_a_inside >= 131.5)
    else:
        area_ratio = 0.0
        mean_a_inside = 128.0
        is_valid_tongue = False

    # Fallback elliptical mask in center if user uploaded a very close-up cropped tongue
    if not is_valid_tongue and float(np.mean(A[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)])) >= 135.0:
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.34), int(h * 0.38)), 0, 0, 360, 255, -1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = cnts[0] if cnts else None
        area_ratio = float(np.sum(mask > 0)) / float(h * w)
        mean_a_inside = float(np.mean(A[mask > 0]))
        is_valid_tongue = True

    overlay = img_bgr.copy()
    if best_contour is not None:
        cv2.drawContours(overlay, [best_contour], -1, (0, 230, 118), 3)

    quality_meta = {
        "is_valid_tongue": bool(is_valid_tongue),
        "blur_score": round(blur_score, 1),
        "mean_brightness": round(mean_brightness, 1),
        "area_ratio": round(area_ratio * 100, 1),
        "mean_a_channel": round(mean_a_inside, 1),
        "quality_warning": None
    }
    if not is_valid_tongue:
        quality_meta["quality_warning"] = (
            "Low tongue visibility or non-tongue image detected. Please upload a clear close-up photo of the extruded tongue in natural light."
        )
    elif blur_score < 25.0:
        quality_meta["quality_warning"] = "Image appears slightly blurry; fine fissure detection may have reduced sensitivity."
    elif mean_brightness < 55 or mean_brightness > 235:
        quality_meta["quality_warning"] = "Lighting is suboptimal (too dark or overexposed). White-balance correction was applied."

    return mask, overlay, quality_meta, best_contour


def analyze_tongue_image(img_bgr: np.ndarray, age: int = 30, gender: str = "Other") -> Dict[str, Any]:
    """
    Runs the complete 7-module deterministic OpenCV pipeline on a BGR image.
    Returns fixed-schema numerical metrics, visual overlays, and Ayurvedic interpretations.
    """
    # Standardize resolution while preserving aspect ratio
    target_w = 520
    h0, w0 = img_bgr.shape[:2]
    scale = target_w / float(max(w0, 1))
    img_resized = cv2.resize(img_bgr, (target_w, max(100, int(h0 * scale))), interpolation=cv2.INTER_AREA)

    wb_img = gray_world_white_balance(img_resized)
    mask, seg_overlay, quality_meta, contour = segment_tongue(wb_img)

    h, w = wb_img.shape[:2]
    if np.sum(mask > 0) < 200:
        # Safe center mask so metrics don't crash even if image is invalid
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.3), int(h * 0.35)), 0, 0, 360, 255, -1)

    lab = cv2.cvtColor(wb_img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(wb_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(wb_img, cv2.COLOR_BGR2GRAY)

    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    tongue_pixels = mask > 0
    total_tongue_px = max(1, int(np.sum(tongue_pixels)))

    # --- MODULE 2: COATING (LEPA) DETECTION ---
    # Coating pixels inside tongue mask have lower redness (A) or lower saturation (S) with higher lightness (L)
    med_L = float(np.median(L[tongue_pixels]))
    med_A = float(np.median(A[tongue_pixels]))
    med_S = float(np.median(S[tongue_pixels]))

    coating_px_mask = (
        tongue_pixels
        & (L > max(125, med_L * 0.88))
        & ((S < 92) | (A < max(142, med_A - 4)))
        & (V > 110)
    )
    # Erode mask border slightly so specular lip edges aren't counted as coating
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner_mask = cv2.erode(mask, inner_kernel, iterations=1) > 0
    coating_px_mask = coating_px_mask & inner_mask

    coating_area_pct = round((float(np.sum(coating_px_mask)) / float(max(1, np.sum(inner_mask)))) * 100.0, 1)
    coating_area_pct = float(np.clip(coating_area_pct, 0.0, 95.0))

    # Coating overlay visualization
    coating_overlay = wb_img.copy()
    coating_overlay[coating_px_mask] = (
        0.45 * coating_overlay[coating_px_mask] + 0.55 * np.array([255, 220, 60])
    ).astype(np.uint8)

    # Determine coating colour & state
    if coating_area_pct < 8.0:
        coating_present = "Absent"
        coating_state = ALLOWED_COATING_STATES[0]
    else:
        coating_present = "Present"
        mean_b_coat = float(np.mean(B[coating_px_mask]))
        mean_l_coat = float(np.mean(L[coating_px_mask]))
        if mean_l_coat < 135:
            coating_state = ALLOWED_COATING_STATES[4]  # Brownish / Greyish (Vata-Ama)
        elif mean_b_coat > 144:
            coating_state = ALLOWED_COATING_STATES[3]  # Yellowish (Pitta-Ama)
        elif coating_area_pct >= 32.0:
            coating_state = ALLOWED_COATING_STATES[2]  # Thick White (Kapha-Ama)
        else:
            coating_state = ALLOWED_COATING_STATES[1]  # Thin White (Alpa Sama)

    # --- MODULE 1: TONGUE BODY COLOUR ANALYSIS ---
    uncoated_px = inner_mask & (~coating_px_mask)
    if np.sum(uncoated_px) < 100:
        uncoated_px = inner_mask

    mean_L_body = float(np.mean(L[uncoated_px]))
    mean_A_body = float(np.mean(A[uncoated_px]))
    mean_B_body = float(np.mean(B[uncoated_px]))
    mean_S_body = float(np.mean(S[uncoated_px]))

    if mean_A_body < 138.0 and mean_L_body >= 150.0:
        body_colour = ALLOWED_COLOURS[1]  # Pale
        colour_short = "Pale"
    elif mean_A_body > 158.0 and mean_L_body >= 120.0:
        body_colour = ALLOWED_COLOURS[2]  # Red
        colour_short = "Red"
    elif mean_A_body > 152.0 and mean_L_body < 120.0:
        body_colour = ALLOWED_COLOURS[3]  # Dark Red / Purplish
        colour_short = "Dark Red / Purplish"
    elif mean_B_body < 118.0 and mean_A_body < 142.0:
        body_colour = ALLOWED_COLOURS[4]  # Bluish
        colour_short = "Bluish"
    else:
        body_colour = ALLOWED_COLOURS[0]  # Pinkish
        colour_short = "Pinkish"

    # --- MODULE 3: FISSURE & GROOVE DETECTION ---
    # Morphological Black-Hat transform highlights dark grooves/cracks inside tongue surface
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    deep_erode = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    bh_masked = cv2.bitwise_and(blackhat, blackhat, mask=deep_erode)

    _, fissure_bin = cv2.threshold(bh_masked, 22, 255, cv2.THRESH_BINARY)
    # Filter connected components that are elongated grooves
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fissure_bin, connectivity=8)
    fissure_clean = np.zeros_like(fissure_bin)
    fissure_count = 0
    for i in range(1, num_labels):
        area_c = stats[i, cv2.CC_STAT_AREA]
        w_c = stats[i, cv2.CC_STAT_WIDTH]
        h_c = stats[i, cv2.CC_STAT_HEIGHT]
        if area_c >= 28 and (max(w_c, h_c) >= 12):
            fissure_clean[labels == i] = 255
            fissure_count += 1

    fissure_density = round((float(np.sum(fissure_clean > 0)) / float(max(1, np.sum(deep_erode > 0)))) * 100.0, 2)
    fissure_overlay = wb_img.copy()
    fissure_overlay[fissure_clean > 0] = [0, 0, 255]

    if fissure_count == 0 or fissure_density < 0.35:
        fissures_short = "Not detected"
        fissure_state = ALLOWED_FISSURE_STATES[0]
    elif fissure_count <= 2 and fissure_density < 1.5:
        fissures_short = "Detected (Mild)"
        fissure_state = ALLOWED_FISSURE_STATES[1]
    elif fissure_density < 3.5:
        fissures_short = f"Detected ({fissure_count} grooves)"
        fissure_state = ALLOWED_FISSURE_STATES[2]
    else:
        fissures_short = f"Detected (Deep, {fissure_count} grooves)"
        fissure_state = ALLOWED_FISSURE_STATES[3]

    # --- MODULE 4 & 5: SHAPE & TOOTH-MARK DETECTION ---
    tooth_marks_short = "Not detected"
    tooth_marks_state = ALLOWED_TOOTH_MARKS[0]
    shape_short = "Oval"
    shape_state = ALLOWED_SHAPES[0]
    indentation_count = 0
    aspect_ratio = 1.25

    if contour is not None and len(contour) >= 5:
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = round(float(bh) / float(max(bw, 1)), 2)
        hull = cv2.convexHull(contour, returnPoints=True)
        hull_area = max(1.0, cv2.contourArea(hull))
        solidity = float(cv2.contourArea(contour)) / hull_area

        if aspect_ratio > 1.38:
            shape_short = "Long & Thin"
            shape_state = ALLOWED_SHAPES[2]
        elif aspect_ratio < 0.96 or solidity > 0.95:
            shape_short = "Broad / Swollen"
            shape_state = ALLOWED_SHAPES[1]
        else:
            shape_short = "Oval"
            shape_state = ALLOWED_SHAPES[0]

        # Convexity defects along lateral borders for tooth marks
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull_idx)
                if defects is not None:
                    defects_flat = defects.reshape(-1, 4)
                    for i in range(defects_flat.shape[0]):
                        s, e, f, d = defects_flat[i]
                        far_pt = tuple(np.asarray(contour[f]).reshape(-1)[:2])
                        depth_px = d / 256.0
                        # Lateral borders: left 30% or right 30% of bounding box, middle 65% vertically
                        is_lateral = (far_pt[0] < x + bw * 0.32) or (far_pt[0] > x + bw * 0.68)
                        is_mid_y = (y + bh * 0.18) < far_pt[1] < (y + bh * 0.85)
                        if is_lateral and is_mid_y and 4.5 <= depth_px <= (bw * 0.14):
                            indentation_count += 1
                            cv2.circle(seg_overlay, (int(far_pt[0]), int(far_pt[1])), 5, (255, 80, 0), -1)
            except cv2.error:
                pass

        if indentation_count >= 3:
            tooth_marks_short = f"Detected ({indentation_count} indentations)"
            tooth_marks_state = ALLOWED_TOOTH_MARKS[2]
        elif indentation_count >= 1:
            tooth_marks_short = "Mild lateral marks"
            tooth_marks_state = ALLOWED_TOOTH_MARKS[1]

    # --- MODULE 6: MOISTURE ESTIMATION ---
    # Specular reflection ratio inside tongue mask
    specular_px = inner_mask & (V > 225) & (S < 45)
    spec_ratio = (float(np.sum(specular_px)) / float(max(1, np.sum(inner_mask)))) * 100.0
    if quality_meta["blur_score"] < 20.0 or not quality_meta["is_valid_tongue"]:
        moisture_short = "Cannot reliably determine"
        moisture_state = ALLOWED_MOISTURE[3]
    elif spec_ratio > 3.2:
        moisture_short = "Excessively Moist (Ardra)"
        moisture_state = ALLOWED_MOISTURE[1]
    elif spec_ratio < 0.18 and fissure_density > 1.2:
        moisture_short = "Dry / Rough (Ruksha)"
        moisture_state = ALLOWED_MOISTURE[2]
    else:
        moisture_short = "Normal Moist"
        moisture_state = ALLOWED_MOISTURE[0]

    # --- MODULE 7: REGIONAL ORGAN ZONE ANALYSIS (Tip, Middle, Root) ---
    ys, xs = np.where(tongue_pixels)
    if len(ys) > 20:
        min_y, max_y = int(np.min(ys)), int(np.max(ys))
        span_y = max(10, max_y - min_y)
        root_zone = tongue_pixels & (np.arange(h)[:, None] < (min_y + span_y * 0.33))
        mid_zone = tongue_pixels & (np.arange(h)[:, None] >= (min_y + span_y * 0.33)) & (np.arange(h)[:, None] < (min_y + span_y * 0.68))
        tip_zone = tongue_pixels & (np.arange(h)[:, None] >= (min_y + span_y * 0.68))

        root_coat = round(float(np.sum(coating_px_mask & root_zone)) / float(max(1, np.sum(root_zone))) * 100.0, 1)
        mid_coat = round(float(np.sum(coating_px_mask & mid_zone)) / float(max(1, np.sum(mid_zone))) * 100.0, 1)
        tip_redness = round(float(np.mean(A[tip_zone])), 1) if np.any(tip_zone) else round(mean_A_body, 1)
    else:
        root_coat, mid_coat, tip_redness = coating_area_pct, coating_area_pct, round(mean_A_body, 1)

    # --- DETERMINISTIC DOSHA SCORING ENGINE ---
    vata_pts = 15.0
    pitta_pts = 15.0
    kapha_pts = 15.0

    # Fissures -> strong Vata indicator (Sphutita Jihva)
    if fissure_density >= 3.5 or fissure_count >= 5:
        vata_pts += 35.0
    elif fissure_count >= 2:
        vata_pts += 22.0
    elif fissure_count == 1:
        vata_pts += 10.0

    # Body colour
    if colour_short == "Red":
        pitta_pts += 32.0
    elif colour_short == "Dark Red / Purplish":
        pitta_pts += 22.0
        vata_pts += 15.0
    elif colour_short == "Pale":
        kapha_pts += 28.0
    elif colour_short == "Bluish":
        vata_pts += 28.0

    # Coating
    if coating_state == ALLOWED_COATING_STATES[2]:  # Thick White
        kapha_pts += 32.0
    elif coating_state == ALLOWED_COATING_STATES[3]:  # Yellowish
        pitta_pts += 30.0
    elif coating_state == ALLOWED_COATING_STATES[4]:  # Brownish/Greyish
        vata_pts += 28.0
    elif coating_state == ALLOWED_COATING_STATES[1]:  # Thin White
        kapha_pts += 10.0

    # Shape & Tooth marks
    if shape_short == "Broad / Swollen":
        kapha_pts += 20.0
    elif shape_short == "Long & Thin":
        vata_pts += 20.0
    if indentation_count >= 2:
        kapha_pts += 14.0
        vata_pts += 10.0

    # Age & Gender contextual adjustment (Vaya Pariksha in Ayurveda:
    # Balya/Youth <16 -> Kapha natural stage; Madhya 16-60 -> Pitta stage; Vriddha >60 -> Vata stage)
    age_stage_note = ""
    if age < 16:
        age_stage_note = f"Age ({age} yrs — Balya Avastha): Natural Kapha physiological phase; mild white coating or fuller tongue contour is common in youth."
    elif age <= 58:
        age_stage_note = f"Age ({age} yrs — Madhya Avastha): Natural Pitta metabolic phase; heat/redness markers on the tongue warrant extra cooling attention."
    else:
        age_stage_note = f"Age ({age} yrs — Vriddha Avastha): Natural Vata physiological phase; mild dryness and surface grooves increase naturally with age."

    gender_note = ""
    if gender.lower() == "female" and colour_short == "Pale":
        gender_note = "In female physiology, a pale tongue body (Pandu) often correlates with Rakta Dhatu / iron nourishment needs alongside Kapha."
    elif gender.lower() == "male" and colour_short == "Red":
        gender_note = "Redness with coating in adult physiology frequently reflects dietary Pitta (spicy/fermented foods or late meals)."

    total_pts = max(1.0, vata_pts + pitta_pts + kapha_pts)
    vata_pct = round((vata_pts / total_pts) * 100.0, 1)
    pitta_pct = round((pitta_pts / total_pts) * 100.0, 1)
    kapha_pct = round(100.0 - vata_pct - pitta_pct, 1)

    scores = {"Vata": vata_pct, "Pitta": pitta_pct, "Kapha": kapha_pct}
    sorted_doshas = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_d, top_val = sorted_doshas[0]
    sec_d, sec_val = sorted_doshas[1]

    if top_val < 38.0 and (top_val - sorted_doshas[2][1]) < 12.0:
        dominant_dosha = "Tridosha Balanced (Sama)"
    elif (top_val - sec_val) <= 7.5 and top_val >= 36.0:
        pair = {top_d, sec_d}
        if pair == {"Vata", "Pitta"}:
            dominant_dosha = "Vata-Pitta"
        elif pair == {"Pitta", "Kapha"}:
            dominant_dosha = "Pitta-Kapha"
        else:
            dominant_dosha = "Vata-Kapha"
    else:
        dominant_dosha = f"{top_d} Predominant"

    # Ama (digestive toxin) status
    if coating_area_pct < 10.0:
        ama_status = "Nirama (Clear Agni / Minimal Toxin Accumulation)"
    elif coating_area_pct < 30.0:
        ama_status = "Alpa Sama (Mild Digestive Coating / Early Ama)"
    else:
        ama_status = "Sama Jihva (Moderate-to-High Ama / Sluggish Agni)"

    # Fixed-template Ayurvedic Suggestions, Diet (Pathya) & Classical Remedies
    remedies = get_curated_ayurvedic_protocol(dominant_dosha, coating_area_pct, fissure_count)

    # Feature vector for adaptive KNN feedback model
    feature_vector = [
        round(mean_L_body, 2),
        round(mean_A_body, 2),
        round(mean_B_body, 2),
        round(mean_S_body, 2),
        round(coating_area_pct, 2),
        round(fissure_density, 2),
        float(fissure_count),
        round(aspect_ratio, 2),
        float(indentation_count),
        float(age)
    ]

    return {
        "quality": quality_meta,
        "parameters": {
            "Colour": colour_short,
            "Colour_Full": body_colour,
            "Coating": coating_present,
            "Coating_Area": f"{coating_area_pct}%",
            "Coating_Area_Num": coating_area_pct,
            "Coating_Type": coating_state,
            "Fissures": fissures_short,
            "Fissure_State": fissure_state,
            "Fissure_Count": fissure_count,
            "Shape": shape_short,
            "Shape_State": shape_state,
            "Tooth_Marks": tooth_marks_short,
            "Tooth_Marks_State": tooth_marks_state,
            "Moisture": moisture_short,
            "Moisture_State": moisture_state,
        },
        "zones": {
            "Tip (Heart / Lungs - Kapha/Pitta)": "Elevated redness (Heat)" if tip_redness > 152 else "Balanced",
            "Middle (Stomach / Agni - Pitta/Kapha)": f"{mid_coat}% coated ({'Ama present' if mid_coat >= 22 else 'Clear'})",
            "Root (Colon / Pakvashaya - Vata)": f"{root_coat}% coated ({'Apana Vata / Ama' if root_coat >= 25 else 'Clear'})",
        },
        "dosha_scores": scores,
        "dominant_dosha": dominant_dosha,
        "ama_status": ama_status,
        "age_gender_context": f"{age_stage_note} {gender_note}".strip(),
        "remedies": remedies,
        "feature_vector": feature_vector,
        "images": {
            "original_wb": wb_img,
            "seg_overlay": seg_overlay,
            "coating_overlay": coating_overlay,
            "fissure_overlay": fissure_overlay,
        }
    }


def get_curated_ayurvedic_protocol(dominant_dosha: str, coating_pct: float, fissure_count: int) -> Dict[str, list]:
    """
    Returns strictly verified, non-hallucinated classical Ayurvedic lifestyle, dietary (Pathya-Apathya),
    and gentle domestic herbal regimens based on Jihva Pariksha findings.
    """
    dinacharya = [
        "Jihva Nirlekhana: Gently scrape the tongue from root to tip 5–7 strokes every morning using a copper or silver scraper (Charaka Samhita Su. 5.74).",
        "Gandusha / Kavala (Oil Pulling): Swish 1 tablespoon of warm sesame oil (for Vata/Kapha) or coconut oil (for Pitta) for 3–5 minutes, then rinse with warm water."
    ]
    herbs = []
    diet_pathya = []
    diet_apathya = []

    if "Vata" in dominant_dosha or fissure_count >= 2:
        herbs.append("Warm Dashamoola or Mulethi (Licorice / Yashtimadhu) infusion — 1/2 tsp steeped in warm water after meals to soothe dryness and fissures.")
        herbs.append("Triphala Churna — 1/2 tsp with lukewarm water and 4 drops of cow's ghee at bedtime to pacify Apana Vata.")
        diet_pathya.append("Warm, freshly cooked, unctuous meals (Moong dal khichdi with A2 ghee, stewed seasonal vegetables, soaked almonds).")
        diet_apathya.append("Avoid cold/refrigerated drinks, dry crackers, excessive raw salads, late-night waking, and skipping meals.")

    if "Pitta" in dominant_dosha:
        herbs.append("Dhanyaka Hima (Coriander + Fennel seed cold/lukewarm infusion) — 1 tsp soaked in water to cool digestive Pitta.")
        herbs.append("Amalaki (Amla) or Avipattikar Churna — 1/2 tsp before meals with room-temperature water to balance Pachaka Pitta.")
        diet_pathya.append("Cooling, sweet-bitter-astringent foods: pomegranate, sweet gourd (Lauki), mint-coriander chutney, and rose-infused water.")
        diet_apathya.append("Avoid sour/fermented foods, excess red chillies, deep-fried snacks, vinegar, and excessive coffee/tea on an empty stomach.")

    if "Kapha" in dominant_dosha or coating_pct >= 25.0:
        herbs.append("Trikatu or Dry Ginger (Shunthi) + Black Pepper + Tulsi warm decoction — sipped in the morning to kindle Agni and clear Ama coating.")
        herbs.append("Warm turmeric + rock salt gargle (Kavalagraha) to clear Kapha-Ama coating from the root of the tongue.")
        diet_pathya.append("Light, warm, easily digestible meals: barley (Yava), old rice, green gram soup seasoned with cumin, ginger, and black pepper.")
        diet_apathya.append("Avoid cold dairy/curd at night, heavy sweets, refined flour (Maida), daytime sleeping (Divasvapna), and overeating before the previous meal digests.")

    if not herbs:
        herbs.append("Cumin-Coriander-Fennel (CCF) tea after meals to maintain balanced Agni (Sama Agni).")
        diet_pathya.append("Eat freshly prepared seasonal meals aligned with hunger (Agnibala) and sip warm water throughout the day.")
        diet_apathya.append("Avoid incompatible food combinations (Viruddha Ahara) such as milk with sour fruits or fish.")

    return {
        "dinacharya": dinacharya,
        "herbal_remedies": herbs[:3],
        "pathya_diet": diet_pathya[:2],
        "apathya_avoid": diet_apathya[:2]
    }
