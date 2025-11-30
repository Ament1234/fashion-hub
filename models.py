from django.db import models


class ClothingItem(models.Model):
    GENDER_CHOICES = [
        ("Men", "Men"),
        ("Women", "Women"),
        ("Girls", "Girls"),
        ("Boys", "Boys"),
        ("Unisex", "Unisex"),
    ]

    kaggle_id = models.IntegerField(unique=True)      # "id" column in styles.csv
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)
    master_category = models.CharField(max_length=50)
    sub_category = models.CharField(max_length=50)
    article_type = models.CharField(max_length=100)
    base_colour = models.CharField(max_length=50)
    season = models.CharField(max_length=20)
    year = models.IntegerField(null=True, blank=True)
    usage = models.CharField(max_length=50)
    product_display_name = models.CharField(max_length=255)

    # path relative to MEDIA_ROOT, e.g. "fashion/images/1234.jpg"
    image_path = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.product_display_name} ({self.base_colour})"

    @property
    def image_url(self):
        from django.conf import settings
        return settings.MEDIA_URL + self.image_path
