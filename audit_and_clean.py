# audit_and_clean.py
# Usage: python audit_and_clean.py
# Place this in your project root (same BASE_DIR as settings.BASE_DIR).

import csv
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# <-- set this to your CSV file inside yourapp\data (update filename if different)
CSV_IN = os.path.join(BASE_DIR, "yourapp", "data", "styles.csv")

# outputs (cleaned / report) will be created next to your csv inside yourapp\data
CSV_OUT = os.path.join(BASE_DIR, "yourapp", "data", "styles_cleaned.csv")
CSV_PROBLEMS = os.path.join(BASE_DIR, "yourapp", "data", "styles_problem_rows.csv")
REPORT = os.path.join(BASE_DIR, "yourapp", "data", "styles_audit_report.txt")

# image lookup dirs (script will try these to find images)
MEDIA_DIRS = [
    os.path.join(BASE_DIR, "yourapp", "media", "fashion", "images"),
    os.path.join(BASE_DIR, "yourapp", "media"),
    os.path.join(BASE_DIR, "media"),
]


# === Normalization maps (edit/add items as you need) ===
COLOR_MAP = {
    # common variants -> canonical
    "navy blue": "blue",
    "navy": "blue",
    "royal blue": "blue",
    "off white": "white",
    "ivory": "white",
    "blk": "black",
    "blk.": "black",
    "lt blue": "light blue",
    "dk blue": "blue",
    "grey": "gray",
    "charcoal": "gray",
    "maroon": "red",
    "burgundy": "red",
}

GENDER_MAP = {
    "m": "men",
    "male": "men",
    "man": "men",
    "men": "men",
    "f": "women",
    "female": "women",
    "woman": "women",
    "women": "women",
}

SEASON_MAP = {
    "summer": "Summer",
    "winter": "Winter",
    "all": "All",
    "spring": "Spring",
    "autumn": "Autumn",
    "fall": "Autumn"
}

# canonical header names we will output
CANONICAL_HEADERS = [
    "id",
    "productDisplayName",
    "articleType",
    "baseColour",
    "gender",
    "usage",
    "season",
    "image_url",
    "price",
    "discount"
]

# helpers
def canonical_header(h):
    h = (h or "").strip()
    h_low = h.lower()
    if h_low in ("id", "itemid", "productid", "product_id"):
        return "id"
    if h_low in ("productdisplayname", "product_display_name", "name", "title"):
        return "productDisplayName"
    if h_low in ("articletype", "article_type", "article", "type"):
        return "articleType"
    if h_low in ("basecolour", "base_colour", "color", "colour", "basecolor"):
        return "baseColour"
    if h_low in ("gender", "sex"):
        return "gender"
    if h_low in ("usage","category","use"):
        return "usage"
    if h_low in ("season",):
        return "season"
    if h_low in ("image_url","image","img","imageurl"):
        return "image_url"
    if h_low in ("price", "mrp", "cost"):
        return "price"
    if h_low in ("discount","badge"):
        return "discount"
    # default: return original trimmed header as fallback
    return h

def normalize_color(c):
    if not c:
        return ""
    s = c.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    if s in COLOR_MAP:
        return COLOR_MAP[s]
    # try strip punctuation
    s2 = re.sub(r'[^a-z0-9 ]+', '', s)
    if s2 in COLOR_MAP:
        return COLOR_MAP[s2]
    # common short maps
    if s in ("blk", "black"):
        return "black"
    if s in ("w", "white"):
        return "white"
    return s

def normalize_gender(g):
    if not g:
        return ""
    s = g.strip().lower()
    return GENDER_MAP.get(s, s)

def guess_image_path(row):
    # try a few common patterns
    idv = row.get("id") or ""
    candidates = []
    if idv:
        candidates.append(os.path.join("media", "men", f"{idv}.jpg"))
        candidates.append(os.path.join("media", "men", f"{idv}.jpeg"))
        candidates.append(os.path.join("media", f"{idv}.jpg"))
    # if csv had image url
    if row.get("image_url"):
        candidates.append(row.get("image_url"))
    return candidates

def file_exists_any(candidates):
    for c in candidates:
        # if it's an absolute or URL starting with http, count as exists (we won't check remote)
        if isinstance(c, str) and (c.startswith("http://") or c.startswith("https://")):
            return True, c
        path = os.path.join(BASE_DIR, c) if not os.path.isabs(c) else c
        if os.path.exists(path):
            return True, c
    return False, None

# === Main processing ===
if not os.path.exists(CSV_IN):
    print("ERROR: dataset/styles.csv not found at", CSV_IN)
    raise SystemExit(1)

with open(CSV_IN, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    raw_rows = list(reader)

# Build mapped headers
original_headers = reader.fieldnames if reader.fieldnames else []
mapped_headers = [canonical_header(h) for h in original_headers]

# Counters and reports
bad_rows = []
duplicates = []
seen_signatures = set()
article_counter = Counter()
color_counter = Counter()
gender_counter = Counter()
missing_image_rows = []

clean_rows = []
for i, raw in enumerate(raw_rows, start=1):
    # build normalized row dict
    row = {}
    # map columns
    for orig_h in original_headers:
        can_h = canonical_header(orig_h)
        row[can_h] = (raw.get(orig_h) or "").strip()

    # Normalize frequently problematic fields
    row['baseColour'] = normalize_color(row.get('baseColour',''))
    row['gender'] = normalize_gender(row.get('gender',''))
    # Title-case season
    if row.get('season'):
        row['season'] = SEASON_MAP.get(row['season'].strip().lower(), row['season'].strip().title())
    # Fill id if missing using hash of name (not ideal, but helps)
    if not row.get('id'):
        if row.get('productDisplayName'):
            row['id'] = re.sub(r'[^0-9a-z]+', '-', row['productDisplayName'].strip().lower())[:40]
        else:
            row['id'] = f"row{i}"

    # Track counters
    article_counter[row.get('articleType','').lower()] += 1
    color_counter[row.get('baseColour','').lower()] += 1
    gender_counter[row.get('gender','').lower()] += 1

    # detect duplicates by (name + articleType + baseColour)
    signature = (row.get('productDisplayName','').strip().lower(),
                 row.get('articleType','').strip().lower(),
                 row.get('baseColour','').strip().lower())
    if signature in seen_signatures:
        duplicates.append((i, row))
    else:
        seen_signatures.add(signature)

    # image checks
    candidates = guess_image_path(row)
    exists, path = file_exists_any(candidates)
    if not exists:
        missing_image_rows.append((i, row, candidates))

    clean_rows.append(row)

# Write cleaned CSV
with open(CSV_OUT, "w", newline='', encoding="utf-8") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=CANONICAL_HEADERS)
    writer.writeheader()
    for r in clean_rows:
        out = {k: r.get(k,'') for k in CANONICAL_HEADERS}
        writer.writerow(out)

# Write problem rows
with open(CSV_PROBLEMS, "w", newline='', encoding="utf-8") as p_f:
    writer = csv.DictWriter(p_f, fieldnames=list(raw_rows[0].keys()) if raw_rows else CANONICAL_HEADERS)
    writer.writeheader()
    for idx, row, candidates in missing_image_rows:
        # merge candidate list into row for visibility
        outrow = row.copy()
        outrow['_missing_image_candidates'] = ";".join(candidates)
        writer.writerow({k: outrow.get(k,'') for k in writer.fieldnames})

# Write report
with open(REPORT, "w", encoding="utf-8") as rep:
    rep.write("STYLES.CSV AUDIT REPORT\n")
    rep.write("=======================\n\n")
    rep.write(f"Input file: {CSV_IN}\n")
    rep.write(f"Rows read: {len(raw_rows)}\n")
    rep.write(f"Cleaned output: {CSV_OUT}\n")
    rep.write(f"Problem rows (missing images): {len(missing_image_rows)} -> {CSV_PROBLEMS}\n")
    rep.write(f"Duplicate signatures detected: {len(duplicates)}\n\n")

    rep.write("Top articleType values (sample):\n")
    for k, v in article_counter.most_common(20):
        rep.write(f"  {k or '[empty]'}: {v}\n")
    rep.write("\nTop baseColour values (sample):\n")
    for k, v in color_counter.most_common(50):
        rep.write(f"  {k or '[empty]'}: {v}\n")
    rep.write("\nGender counts:\n")
    for k, v in gender_counter.most_common():
        rep.write(f"  {k or '[empty]'}: {v}\n")

    rep.write("\nDuplicates (first 20):\n")
    for idx, row in duplicates[:20]:
        rep.write(f"  Row {idx}: {row.get('productDisplayName')} | {row.get('articleType')} | {row.get('baseColour')}\n")

    rep.write("\nMissing image rows (first 50):\n")
    for idx, row, candidates in missing_image_rows[:50]:
        rep.write(f"  Row {idx}: id={row.get('id')} name={row.get('productDisplayName')} candidates={candidates}\n")

print("Audit complete.")
print("Report:", REPORT)
print("Clean CSV:", CSV_OUT)
print("Problem rows:", CSV_PROBLEMS)
