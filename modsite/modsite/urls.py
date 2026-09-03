"""URLهای اصلی سایت."""
from django.urls import path

from core import views

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.single_test, name='test'),
    path('batch/', views.batch_test, name='batch'),
    path('batch/download/<str:fname>/', views.batch_download, name='batch_download'),
    path('batch/sample/', views.batch_sample, name='batch_sample'),
    path('train/', views.train_page, name='train'),
    path('train/upload-data/', views.train_upload_data, name='train_upload'),
    path('train/start/', views.train_start, name='train_start'),
    path('status/', views.status_json, name='status'),
]
