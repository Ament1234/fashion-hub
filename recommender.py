import cv2
import numpy as np
import pandas as pd
import random
import re
from pathlib import Path
from PIL import ImageColor
from django.conf import settings
from sklearn.cluster import KMeans

# ---------- CONFIG / PATHS ----------
STYLES_CSV = Path(settings.BASE_DIR) / "yourapp" / "data" / "styles.csv"
IMAGES_CSV = Path(settings.BASE_DIR) / "yourapp" / "data" / "images.csv"

# Folder where your image files (1234.jpg etc.) are stored inside MEDIA_ROOT
# e.g. MEDIA_ROOT / "images" / "1234.jpg"
IMAGE_FOLDER_NAME = "images"

# cached dataframe
PRODUCTS_DF = None


# ---------- UTIL HELPERS ----------
def _safe_title(s):
    """Normalize color/master names to Title Case and strip whitespace (handles NaN)."""
    if pd.isna(s):
        return ""
    return str(s).strip().title()


def _safe_lower(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


def _extract_id_from_filename(fn: str):
    """
    Try to extract an integer id from a filename like '12345.jpg' or 'img_12345.jpeg'.
    Return None if not found.
    """
    if pd.isna(fn):
        return None
    fname = str(fn)
    m = re.search(r"(\d+)", fname)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _build_image_url(row):
    """
    Build a usable image URL.
    Priority:
      1) If CSV already has a full link -> use it
      2) Otherwise, build from MEDIA_URL/IMAGE_FOLDER_NAME/<id>.jpg
    """
    link = row.get("image_link", "")
    if isinstance(link, str) and link.strip():
        # Already a valid URL or path
        return link.strip()

    # No link in CSV, try to build from id
    _id = row.get("id", None)
    if pd.isna(_id) or _id is None:
        return ""

    try:
        _id = int(_id)
    except Exception:
        return ""

    # Example: /media/images/1234.jpg
    return f"{settings.MEDIA_URL.rstrip('/')}/{IMAGE_FOLDER_NAME}/{_id}.jpg"


# ---------- 1. LOAD & CLEAN DATASET ONCE ----------
def load_products_df():
    """
    Load and clean the fashion dataset (styles + images).
    Caches loaded DATAFRAME in PRODUCTS_DF.
    """
    global PRODUCTS_DF
    if PRODUCTS_DF is not None:
        return PRODUCTS_DF

    # Read CSVs defensively
    try:
        styles = pd.read_csv(STYLES_CSV, on_bad_lines="skip", dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read styles CSV: {e}")

    try:
        images = pd.read_csv(IMAGES_CSV, on_bad_lines="skip", dtype=str)
    except Exception:
        # If images file is missing, continue with styles only
        images = pd.DataFrame(columns=["filename", "link"])

    # Normalize column names
    styles.columns = [c.strip().lower() for c in styles.columns]
    images.columns = [c.strip().lower() for c in images.columns]

    # Keep relevant columns if present
    desired = ["id", "gender", "mastercategory", "subcategory", "articletype", "basecolour", "season", "usage"]
    present = [c for c in desired if c in styles.columns]
    data = styles[present].copy()

    # Normalize text columns
    for col in ["gender", "mastercategory", "subcategory", "basecolour", "usage"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # Drop rows with essential missing fields
    if "gender" in data.columns and "mastercategory" in data.columns and "basecolour" in data.columns:
        data = data.dropna(subset=["gender", "mastercategory", "basecolour"])

    # Normalize category/gender values
    if "gender" in data.columns:
        data["gender"] = data["gender"].str.title()
    if "mastercategory" in data.columns:
        data["mastercategory"] = data["mastercategory"].str.title()
    if "subcategory" in data.columns:
        data["subcategory"] = data["subcategory"].str.title()
    if "basecolour" in data.columns:
        data["basecolour"] = data["basecolour"].str.title()
    if "usage" in data.columns:
        data["usage"] = data["usage"].replace({"nan": ""})
        data["usage"] = data["usage"].fillna("").astype(str).str.strip().str.title()

    # Filter: keep Men/Women (be tolerant)
    if "gender" in data.columns:
        ok_genders = {"Men", "Women"}
        data = data[data["gender"].isin(ok_genders)]

    # Filter mastercategory
    if "mastercategory" in data.columns:
        data = data[data["mastercategory"].isin(["Apparel", "Footwear"])]

    # Exclude unwanted subcategories (case normalized)
    if "subcategory" in data.columns:
        exclude_sub = {"Innerwear", "Loungewear And Nightwear"}
        data = data[~data["subcategory"].isin(exclude_sub)]

    # ---------- IMAGES MERGE ----------
    if "filename" not in images.columns:
        images["filename"] = ""
    images["filename"] = images["filename"].astype(str)
    images["id"] = images["filename"].apply(_extract_id_from_filename)

    # ensure styles id column exists as int-like where possible
    if "id" in data.columns:
        data["id_raw"] = data["id"]
        try:
            data["id"] = data["id"].astype(int)
        except Exception:
            data["id"] = data["id_raw"].apply(_extract_id_from_filename)

    # Merge on id (only where id matched)
    if "id" in data.columns and "id" in images.columns:
        # Some datasets use 'link', some 'image_link'
        link_col = "link" if "link" in images.columns else "image_link"
        images[link_col] = images.get(link_col, "").astype(str)

        merged = data.merge(
            images[["id", link_col]],
            on="id",
            how="left"
        )
        merged.rename(columns={link_col: "image_link"}, inplace=True)
    else:
        data["image_link"] = ""
        merged = data

    # Build final, non-empty image_link for each row
    merged["image_link"] = merged["image_link"].fillna("").astype(str)
    merged["image_link"] = merged.apply(_build_image_url, axis=1)

    PRODUCTS_DF = merged.reset_index(drop=True)
    return PRODUCTS_DF


# ---------- 2. SKIN TONE & (DISABLED) GENDER DETECTION ----------
skin_tones = {
    "#373028": "Deepest Skin",
    "#422811": "Very Deep",
    "#513B2E": "Deep Brown",
    "#6F503C": "Medium Brown",
    "#81654F": "Tan",
    "#9D7A54": "Light Tan",
    "#BEA07E": "Medium Fair",
    "#E5C8A6": "Light Fair",
    "#E7C1B8": "Warm Fair",
    "#F3DAD6": "Very Fair",
    "#FBF2F3": "Pale",
}


def _read_image_from_django_file(django_file):
    """
    Safely read UploadedFile into OpenCV image.
    """
    try:
        django_file.seek(0)
        file_bytes = np.frombuffer(django_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _dominant_color_kmeans(img_rgb, k=3):
    """
    Use KMeans to find dominant cluster color in RGB image (img_rgb: HxWx3 in RGB).
    Returns RGB tuple.
    """
    h, w, _ = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3).astype(float)
    if len(pixels) > 10000:
        idx = np.random.choice(len(pixels), 10000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    try:
        km = KMeans(n_clusters=k, random_state=0, n_init=4)
        km.fit(sample)
        counts = np.bincount(km.labels_)
        center = km.cluster_centers_[np.argmax(counts)]
        return tuple(int(x) for x in center)
    except Exception:
        avg = pixels.mean(axis=0)
        return tuple(int(x) for x in avg)


def detect_skin_tone_from_image_file(django_file):
    """
    Reads image, detects face, computes dominant skin-like color.
    Returns (hex_code, name) or (None, None).
    """
    img = _read_image_from_django_file(django_file)
    if img is None:
        return None, None

    # face detection (Haar) - fallback to whole image if no face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        y0 = max(0, y + int(h * 0.2))
        y1 = min(img.shape[0], y + h + pad_y)
        x0 = max(0, x + pad_x)
        x1 = min(img.shape[1], x + w - pad_x)
        img_face = img[y0:y1, x0:x1]
        if img_face.size == 0:
            img_face = img[y:y + h, x:x + w]
    else:
        img_face = img

    try:
        img_face = cv2.resize(img_face, (200, 200))
    except Exception:
        pass

    img_rgb = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)

    dom_rgb = _dominant_color_kmeans(img_rgb, k=3)
    avg_color_hex = "#{:02x}{:02x}{:02x}".format(dom_rgb[0], dom_rgb[1], dom_rgb[2])

    avg_color_rgb = np.array(ImageColor.getrgb(avg_color_hex))

    def dist(hex_code):
        return np.linalg.norm(avg_color_rgb - np.array(ImageColor.getrgb(hex_code)))

    closest_tone_hex = min(skin_tones.keys(), key=dist)
    closest_tone_name = skin_tones[closest_tone_hex]
    return closest_tone_hex, closest_tone_name


def detect_gender_from_image_file(django_file):
    """
    GENDER DETECTION DISABLED:
    This function now always returns None.
    Pass gender explicitly from the page/form instead
    (e.g. 'Men' or 'Women') when calling recommend_outfits.
    """
    return None


# ---------- 3. SKIN TONE → COLOR MAPPING ----------
skin_tone_to_color_mapping = {
    "#373028": ["Navy Blue", "Black", "Charcoal", "Burgundy", "Maroon", "Olive", "Rust", "Gold", "Cream", "Peach"],
    "#422811": ["Navy Blue", "Brown", "Khaki", "Olive", "Maroon", "Mustard", "Teal", "Tan", "Rust", "Burgundy"],
    "#513B2E": ["Cream", "Beige", "Olive", "Burgundy", "Red", "Orange", "Mustard", "Bronze", "Teal", "Peach"],
    "#6F503C": ["Beige", "Brown", "Green", "Khaki", "Cream", "Peach", "Lime Green", "Olive", "Maroon", "Rust", "Mustard"],
    "#81654F": ["Beige", "Off White", "Sea Green", "Cream", "Lavender", "Mauve", "Burgundy", "Yellow", "Lime Green"],
    "#9D7A54": ["Olive", "Khaki", "Yellow", "Sea Green", "Turquoise Blue", "Coral", "White", "Gold", "Peach"],
    "#BEA07E": ["Coral", "Sea Green", "Turquoise Blue", "Pink", "Lavender", "Rose", "White", "Peach", "Teal", "Fluorescent Green"],
    "#E5C8A6": ["Turquoise Blue", "Peach", "Teal", "Pink", "Red", "Rose", "Off White", "White", "Cream", "Gold", "Yellow"],
    "#E7C1B8": ["Pink", "Rose", "Peach", "White", "Off White", "Beige", "Lavender", "Teal", "Fluorescent Green"],
    "#F3DAD6": ["White", "Cream", "Peach", "Pink", "Rose", "Lavender", "Mustard", "Lime Green", "Light Blue", "Fluorescent Green"],
    "#FBF2F3": ["Soft Pastels (Peach, Lavender, Pink)", "White", "Off White", "Rose", "Light Blue", "Sea Green", "Fluorescent Green", "Silver", "Cream", "Tan"],
}

NEUTRALS = ["Black", "White", "Beige", "Cream", "Off White", "Grey", "Charcoal"]


# ---------- 4. OUTFIT GENERATION ----------
def _get_complementary(color, palette):
    others = [c for c in palette if c != color]
    return random.choice(others) if others else color


def _get_analogous(color, palette):
    others = [c for c in palette if c != color]
    return random.choice(others) if others else color


def _get_neutral(palette):
    neutrals_in_palette = [c for c in palette if c in NEUTRALS]
    if neutrals_in_palette:
        return random.choice(neutrals_in_palette)
    return random.choice(NEUTRALS)


def _normalize_color_name(name):
    """Make matching tolerant (Title case and strip)"""
    return _safe_title(name)


def recommend_outfits(gender, usage, tone_hex, max_outfits=5):
    """
    Main function: given gender (string or None), usage (string or None),
    and skin tone hex, returns a list of outfit dicts with image links.
    Strategy:
      - strict: gender + recommended color + usage
      - if empty: relax usage requirement
      - if still empty: relax color filter (allow any neutral + recommend colors)
    """
    df = load_products_df()

    if df is None or df.empty:
        return []

    # Normalize inputs
    gender = _safe_title(gender) if gender else ""
    usage = _safe_title(usage) if usage else ""
    recommended_colors = [_safe_title(c) for c in skin_tone_to_color_mapping.get(tone_hex, [])]

    # ensure neutrals in recommended palette
    for n in NEUTRALS:
        if n not in recommended_colors:
            recommended_colors.append(n)

    # Prepare df for matching
    df_local = df.copy()
    if "gender" in df_local.columns:
        df_local["gender"] = df_local["gender"].astype(str).str.title()
    if "basecolour" in df_local.columns:
        df_local["basecolour"] = df_local["basecolour"].astype(str).str.title()
    if "usage" in df_local.columns:
        df_local["usage"] = df_local["usage"].fillna("").astype(str).str.title()
    if "subcategory" in df_local.columns:
        df_local["subcategory"] = df_local["subcategory"].fillna("").astype(str).str.title()
    df_local["image_link"] = df_local["image_link"].fillna("").astype(str)

    # Start filters
    filtered = df_local
    if gender:
        # Only filter by gender if we actually know it
        filtered = filtered[filtered["gender"] == gender]

    # helper to select wearable groups
    def build_outfits(filtered_df):
        # Only items that actually have an image_link
        filtered_df = filtered_df[filtered_df["image_link"].str.strip() != ""]

        top_wear = filtered_df[filtered_df["subcategory"] == "Topwear"]
        bottom_wear = filtered_df[filtered_df["subcategory"] == "Bottomwear"]
        footwear = filtered_df[filtered_df["mastercategory"] == "Footwear"]

        if top_wear.empty or bottom_wear.empty or footwear.empty:
            return []

        combos = []
        top_iter = top_wear.head(20).itertuples(index=False)
        bottom_iter_list = list(bottom_wear.head(80).itertuples(index=False))
        footwear_list = list(footwear.head(80).itertuples(index=False))

        for top in top_iter:
            top_color = _safe_title(getattr(top, "basecolour", ""))
            for bottom in bottom_iter_list:
                bottom_color = _get_complementary(top_color, recommended_colors)
                for foot in footwear_list:
                    foot_color = random.choice(
                        [_get_analogous(bottom_color, recommended_colors), _get_neutral(recommended_colors)]
                    )

                    top_image = getattr(top, "image_link", "") or ""
                    bottom_image = getattr(bottom, "image_link", "") or ""
                    foot_image = getattr(foot, "image_link", "") or ""

                    # Skip if any image is missing
                    if not (top_image and bottom_image and foot_image):
                        continue

                    combos.append({
                        "top_id": int(getattr(top, "id", -1)) if hasattr(top, "id") and pd.notna(getattr(top, "id")) else None,
                        "bottom_id": int(getattr(bottom, "id", -1)) if hasattr(bottom, "id") and pd.notna(getattr(bottom, "id")) else None,
                        "foot_id": int(getattr(foot, "id", -1)) if hasattr(foot, "id") and pd.notna(getattr(foot, "id")) else None,
                        "top_color": top_color,
                        "bottom_color": bottom_color,
                        "foot_color": foot_color,
                        "top_image": top_image,
                        "bottom_image": bottom_image,
                        "foot_image": foot_image,
                    })
        random.shuffle(combos)
        return combos

    # Strategy 1: strict color + strict usage (if provided)
    df_colors = filtered[filtered["basecolour"].isin(recommended_colors)]
    if usage:
        df_colors_usage = df_colors[df_colors["usage"].str.contains(usage, na=False)]
    else:
        df_colors_usage = df_colors

    outfits = build_outfits(df_colors_usage)
    if outfits:
        return outfits[:max_outfits]

    # Strategy 2: relax usage (ignore usage filter)
    outfits = build_outfits(df_colors)
    if outfits:
        return outfits[:max_outfits]

    # Strategy 3: relax color filter (allow any color but prefer recommended in sampling)
    outfits = build_outfits(filtered)
    if outfits:
        return outfits[:max_outfits]

    return []
