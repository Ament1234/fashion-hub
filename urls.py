# yourapp/urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("", views.home, name="home"),            # ROOT → home view
    path("men/", views.men_page, name="men_page"),
    path("men-collection/", views.men_page, name="men_collection"),
    path("man/", views.men_page, name="man_html"),
    path("women/", views.women_page, name="women_page"),
    path("suggestion/", views.suggestion_page, name="suggestion_page"),
    path("api/men-products/", views.api_men_products, name="api_men_products"),
    path("api/women-products/", views.api_women_products, name="api_women_products"),
    path("api/recommend/", views.recommend_api, name="api_recommend"),
    path('cart/', views.cart_page, name='cart_page'),  
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
