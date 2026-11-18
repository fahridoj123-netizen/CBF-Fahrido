from django.urls import path
from . import views

app_name = 'recommender'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('movie/<int:movie_id>/', views.movie_detail_view, name='movie_detail'),
    path('methodology/', views.methodology_view, name='methodology'),
    path('api/recommend/<int:movie_id>/', views.api_recommend, name='api_recommend'),
    path('api/search/', views.api_search, name='api_search'),
]
