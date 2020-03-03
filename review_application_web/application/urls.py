from django.urls import path
from .views import SRDViewSet, FEADViewSet


urlpatterns = [
    path('srd/', SRDViewSet.as_view()),
    path('fead/', FEADViewSet.as_view())
]