from django.urls import re_path
from .views import all_views, picture_view, predict

urlpatterns = [
    re_path('getdata', all_views.as_view()),
    re_path('getimage/', picture_view.as_view()),
    re_path('predict/', predict.as_view())
]
