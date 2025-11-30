import csv
import os
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.conf import settings
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt  # use only for quick local testing
from django.contrib import messages

# Import recommender functions (adjust import path if your module is in a different package)
# Make sure fashion/recommender.py is importable (in your PYTHONPATH or app folder)
# Try a few likely import paths for the recommender module so imports work
# whether recommender.py is in a package named `fashion` or inside this app.
try:
    # original expected package
    from fashion.recommender import (
        detect_skin_tone_from_image_file,
        detect_gender_from_image_file,
        recommend_outfits,
    )
except Exception:
    try:
        # if recommender.py lives at yourapp/recommender.py
        from yourapp.recommender import (
            detect_skin_tone_from_image_file,
            detect_gender_from_image_file,
            recommend_outfits,
        )
    except Exception:
        try:
            # if recommender.py is in project root or same folder
            from recommender import (
                detect_skin_tone_from_image_file,
                detect_gender_from_image_file,
                recommend_outfits,
            )
        except Exception as e:
            # final fallback: provide a helpful error early (so runserver fails with clearer message)
            raise ImportError(
                "Cannot import recommender module. Put recommender.py in one of these locations:\n"
                " - a package named 'fashion' (fashion/recommender.py)\n"
                " - inside this app (yourapp/recommender.py)\n"
                " - project root (recommender.py)\n"
                f"Original error: {e}"
            ) from e


# ---- Helper: read CSV and normalize rows ----
def load_styles_csv(only_gender=None):
    """
    Read styles.csv from the project. Prefer yourapp/data/styles.csv but fall back
    to BASE_DIR/dataset/styles.csv if needed. Returns list of normalized dicts.
    If only_gender is provided (e.g. "men", "women"), only rows with that
    gender (or synonyms like "male", "boys", "girls") are returned.
    """
    rows = []

    primary_path = os.path.join(settings.BASE_DIR, "yourapp", "data", "styles.csv")
    fallback_path = os.path.join(settings.BASE_DIR, "dataset", "styles.csv")
    csv_path = primary_path if os.path.exists(primary_path) else (
        fallback_path if os.path.exists(fallback_path) else None
    )

    if not csv_path:
        return rows

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gender = (row.get("gender") or row.get("Gender") or "").strip().lower()

            # filter by gender if requested
            if only_gender:
                wanted = only_gender.lower()
                allowed = {wanted, wanted + "s"}

                if wanted == "men":
                    allowed.update({"man", "male", "m"})
                elif wanted == "women":
                    allowed.update({"woman", "female", "f"})
                    

                if gender not in allowed:
                    continue

            item_id = (
                row.get("id")
                or row.get("itemid")
                or row.get("product_id")
                or ""
            ).strip()
            name = (
                row.get("productDisplayName")
                or row.get("product_name")
                or row.get("title")
                or row.get("name")
                or ""
            ).strip()
            articleType = (
                row.get("articleType")
                or row.get("article_type")
                or row.get("article")
                or ""
            ).strip()
            baseColour = (
                row.get("baseColour")
                or row.get("base_colour")
                or row.get("color")
                or ""
            ).strip()
            usage = (row.get("usage") or row.get("category") or "").strip()
            season = (row.get("season") or "").strip()

            # try common image column names
            image_url = (
                row.get("image_url")
                or row.get("image")
                or row.get("image_link")
                or ""
            ).strip()
            if not image_url and item_id:
                # fallback to expected media location
                image_url = f"/media/fashion/images/{item_id}.jpg"

            badge = (row.get("discount") or row.get("badge") or None)
            price = (row.get("price") or None)

            rows.append(
                {
                    "id": item_id,
                    "productDisplayName": name,
                    "articleType": articleType,
                    "baseColour": baseColour,
                    "usage": usage,
                    "season": season,
                    "image_url": image_url,
                    "badge": badge,
                    "price": price,
                    "gender": gender,
                    "_raw": row,
                }
            )
    return rows


# ---- Views ----
def home(request):
    """
    Renders the main homepage (home.html). Loads a small set of featured items from CSV.
    """
    featured_items = []
    men_count = 0
    women_count = 0

    rows = load_styles_csv()  # load all available rows
    for row in rows:
        g = row.get("gender", "").lower()
        if g in ("men", "man", "male", "m"):
            men_count += 1
        elif g in ("women", "woman", "female", "f"):
            women_count += 1

        # show up to 15 featured items on home
        if len(featured_items) < 15:
            featured_items.append(
                {
                    "productDisplayName": row.get("productDisplayName"),
                    "articleType": row.get("articleType"),
                    "baseColour": row.get("baseColour"),
                    "gender": row.get("gender"),
                    "season": row.get("season"),
                    "usage": row.get("usage"),
                    "image_url": row.get("image_url") or "/static/img/placeholder.png",
                    "price": row.get("price") or "₹999",
                    "discount": row.get("badge") or row.get("discount"),
                }
            )

    return render(
        request,
        "home.html",
        {
            "featured_items": featured_items,
            "men_count": men_count,
            "women_count": women_count,
        },
    )


def men_page(request):
    return render(request, "men_dataset.html")


def women_page(request):
    return render(request, "women.html")

def suggestion_page(request):
    return render(request, "Suggestion.html")

def cart_page(request):
    cart = request.session.get('cart', {})
    cart_items = []
    subtotal = 0

    for item_id, item_data in cart.items():
        # Get item details from CSV
        products = load_styles_csv()
        product = next((p for p in products if p['id'] == item_id), None)
        if product:
            quantity = item_data.get('quantity', 1)
            price = float(product.get('price', 999)) if product.get('price') else 999
            total = price * quantity
            subtotal += total

            cart_items.append({
                'id': item_id,
                'name': product.get('productDisplayName', ''),
                'articleType': product.get('articleType', ''),
                'baseColour': product.get('baseColour', ''),
                'gender': product.get('gender', ''),
                'image_url': product.get('image_url', ''),
                'price': price,
                'quantity': quantity,
                'total': total,
                'size': item_data.get('size', 'M'),
            })

    shipping = 99
    tax = subtotal * 0.1
    total = subtotal + shipping + tax

    return render(request, 'cart.html', {
        'cart_items': cart_items,
        'subtotal': subtotal,
        'shipping': shipping,
        'tax': tax,
        'total': total,
    })


# --- Helper for pagination ---
def paginate(request, items, default_page_size=50, max_page_size=50):
    """
    Small helper to paginate any list safely.
    Uses default_page_size if client does not send page_size,
    and never allows more than max_page_size.
    """
    try:
        page = int(request.GET.get("page", 1))
    except ValueError:
        page = 1
    try:
        page_size = int(request.GET.get("page_size", default_page_size))
    except ValueError:
        page_size = default_page_size

    page_size = max(1, min(max_page_size, page_size))
    total_items = len(items)
    total_pages = (total_items + page_size - 1) // page_size
    if total_pages <= 0:
        total_pages = 1
    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]

    return page_items, page, page_size, total_items, total_pages


# --- API: Men Products ---
def api_men_products(request):
    try:
        products = load_styles_csv(only_gender="men")

        # filters
        category = (request.GET.get("category") or "").strip().lower()      # usage
        article_type = (
            request.GET.get("articleType")
            or request.GET.get("article_type")
            or ""
        ).strip().lower()
        base_colour = (
            request.GET.get("baseColour")
            or request.GET.get("base_colour")
            or ""
        ).strip().lower()
        search_q = (request.GET.get("q") or "").strip().lower()

        # apply filters
        filtered = []
        for p in products:
            usage = (p.get("usage") or "").strip().lower()
            art = (p.get("articleType") or "").strip().lower()
            colour = (p.get("baseColour") or "").strip().lower()
            name = (p.get("productDisplayName") or "").strip().lower()

            # category -> allow partial match to be more flexible
            if category and category not in usage:
                continue
            if article_type and art != article_type:
                continue
            if base_colour and colour != base_colour:
                continue
            if search_q and search_q not in name:
                continue
            filtered.append(p)

        # pagination (max 20 items per response)
        page_products, page, page_size, total_products, total_pages = paginate(
            request, filtered, default_page_size=15, max_page_size=50
        )

        out = []
        for p in page_products:
            out.append(
                {
                    "id": p.get("id"),
                    "name": p.get("productDisplayName"),
                    "articleType": p.get("articleType"),
                    "article_type": p.get("articleType"),
                    "category": p.get("usage"),
                    "baseColour": p.get("baseColour"),
                    "base_colour": p.get("baseColour"),
                    "usage": p.get("usage"),
                    "season": p.get("season"),
                    "image": p.get("image_url"),
                    "badge": p.get("badge"),
                    "price": p.get("price"),
                }
            )

        return JsonResponse(
            {
                "products": out,
                "page": page,
                "page_size": page_size,
                "total_products": total_products,
                "total_pages": total_pages,
            }
        )
    except Exception as e:
        return JsonResponse({"products": [], "error": str(e)})


# --- API: Women Products ---
def api_women_products(request):
    try:
        products = load_styles_csv(only_gender="women")

        # filters (same idea as men)
        category = (request.GET.get("category") or "").strip().lower()
        article_type = (
            request.GET.get("articleType")
            or request.GET.get("article_type")
            or ""
        ).strip().lower()
        base_colour = (
            request.GET.get("baseColour")
            or request.GET.get("base_colour")
            or ""
        ).strip().lower()
        search_q = (request.GET.get("q") or "").strip().lower()

        filtered = []
        for p in products:
            usage = (p.get("usage") or "").strip().lower()
            art = (p.get("articleType") or "").strip().lower()
            colour = (p.get("baseColour") or "").strip().lower()
            name = (p.get("productDisplayName") or "").strip().lower()

            if category and category not in usage:
                continue
            if article_type and art != article_type:
                continue
            if base_colour and colour != base_colour:
                continue
            if search_q and search_q not in name:
                continue
            filtered.append(p)

        # pagination (max 20 items per response)
        page_products, page, page_size, total_products, total_pages = paginate(
            request, filtered, default_page_size=15, max_page_size=50
        )

        out = []
        for p in page_products:
            out.append(
                {
                    "id": p.get("id"),
                    "name": p.get("productDisplayName"),
                    "articleType": p.get("articleType"),
                    "article_type": p.get("articleType"),
                    "category": p.get("usage"),
                    "baseColour": p.get("baseColour"),
                    "base_colour": p.get("baseColour"),
                    "usage": p.get("usage"),
                    "season": p.get("season"),
                    "image": p.get("image_url"),
                    "badge": p.get("badge"),
                    "price": p.get("price"),
                }
            )

        return JsonResponse(
            {
                "products": out,
                "page": page,
                "page_size": page_size,
                "total_products": total_products,
                "total_pages": total_pages,
            }
        )
    except Exception as e:
        return JsonResponse({"products": [], "error": str(e)})


# ---- Recommender API ----
# NOTE: If you get CSRF 403 while testing, temporarily decorate this view with @csrf_exempt.
@require_POST
def recommend_api(request):
    """
    POST /api/recommend/
    Expects multipart/form-data:
      - image: file
      - usage: optional string
    Returns:
    {
      "gender": "Men" | "Women" ",
      "tone_hex": "#E5C8A6",
      "tone_name": "Light Fair",
      "recommended": [ { top_image, bottom_image, foot_image, top_color, bottom_color, foot_color, ... }, ... ]
    }
    """
    image = request.FILES.get("image")
    usage = (request.POST.get("usage") or "").strip()

    if image is None:
        return HttpResponseBadRequest(
            json.dumps({"error": "No image file provided."}),
            content_type="application/json",
        )

    # Detect skin tone (function may consume the file; we attempt to rewind before reusing)
    tone_hex, tone_name = None, None
    try:
        tone_hex, tone_name = detect_skin_tone_from_image_file(image)
    except Exception:
        tone_hex, tone_name = None, None

    # Rewind file for next read
    try:
        image.seek(0)
    except Exception:
        pass

    # Detect gender (placeholder)
    try:
        gender_detected = detect_gender_from_image_file(image)
    except Exception:
        gender_detected = None

    # Normalize gender fallback
    if gender_detected not in ("Men", "Women"):
        # try to use usage or default to Men if uncertain
        gender_detected = "Men"

    # Use recommender to fetch outfit combos
    tone_for_recommender = tone_hex or ""
    try:
        recommended = recommend_outfits(
            gender_detected, usage, tone_for_recommender, max_outfits=8
        )
    except Exception as e:
        return HttpResponseBadRequest(
            json.dumps({"error": f"Recommender error: {str(e)}"}),
            content_type="application/json",
        )

    # Build response items, converting relative image paths to absolute URLs when possible
    def make_absolute(url):
        if not url or (isinstance(url, float) and url != url):  # handle NaN (NaN != NaN)
            return None
        if isinstance(url, str) and (url.startswith("http://") or url.startswith("https://")):
            return url
        if isinstance(url, str):
            # ensure leading slash
            if not url.startswith("/"):
                url = "/" + url
            return request.build_absolute_uri(url)
        return None

    out_list = []
    for out in recommended:
        top_img = (
            out.get("top_image")
            or out.get("image_link")
            or out.get("top_image_url")
            or None
        )
        bottom_img = out.get("bottom_image") or out.get("bottom_image_url") or None
        foot_img = out.get("foot_image") or out.get("foot_image_url") or None

        out_list.append(
            {
                "top_id": out.get("top_id"),
                "bottom_id": out.get("bottom_id"),
                "foot_id": out.get("foot_id"),
                "top_image": make_absolute(top_img),
                "bottom_image": make_absolute(bottom_img),
                "foot_image": make_absolute(foot_img),
                "top_color": out.get("top_color") or out.get("basecolour") or "",
                "bottom_color": out.get("bottom_color") or "",
                "foot_color": out.get("foot_color") or "",
                "price": out.get("price"),
                "size": out.get("size", "M"),
            }
        )

    response = {
        "gender": gender_detected,
        "tone_hex": tone_hex,
        "tone_name": tone_name,
        "recommended": out_list,
    }
    return JsonResponse(response)
def add_to_cart(request):
    """AJAX endpoint: add item to cart stored in session."""
    product_id = request.POST.get("product_id")
    name       = request.POST.get("name", "")
    image_url  = request.POST.get("image_url", "")
    gender     = request.POST.get("gender", "")
    color      = request.POST.get("color", "")
    size       = request.POST.get("size", "")
    try:
        quantity = int(request.POST.get("quantity", "1"))
    except ValueError:
        quantity = 1
    try:
        price = float(request.POST.get("price", "999"))
    except ValueError:
        price = 999.0

    if not product_id:
        return JsonResponse({"status": "error", "message": "missing product_id"}, status=400)

    cart = request.session.get("cart", [])

    # merge with existing same product + size
    found = False
    for item in cart:
        if item["product_id"] == product_id and item.get("size", "") == size:
            item["quantity"] += quantity
            found = True
            break

    if not found:
        cart.append({
            "id": f"{product_id}-{size}",
            "product_id": product_id,
            "name": name,
            "image_url": image_url,
            "gender": gender,
            "color": color,
            "size": size,
            "quantity": quantity,
            "price": price,
        })

    request.session["cart"] = cart
    request.session.modified = True

    total_count = sum(i["quantity"] for i in cart)
    return JsonResponse({"status": "success", "count": total_count})


def cart_view(request):
    """Render cart.html using items from session cart."""
    cart = request.session.get("cart", [])

    for item in cart:
        item["id"] = item.get("id") or f'{item.get("product_id")}-{item.get("size","")}'
        item["total_price"] = float(item.get("price", 0)) * int(item.get("quantity", 1))

    cart_subtotal = sum(i["total_price"] for i in cart)
    shipping_amount = 0 if cart_subtotal == 0 else 99
    tax_amount = cart_subtotal * 0.10
    cart_total = cart_subtotal + shipping_amount + tax_amount

    context = {
        "cart_items": cart,
        "cart_subtotal": int(cart_subtotal),
        "shipping_amount": int(shipping_amount),
        "tax_amount": int(tax_amount),
        "cart_total": int(cart_total),
    }
    return render(request, "cart.html", context)

