from django.urls import path

from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    # ex: /polls/5/
    path('train/', views.train, name='train'),
    path('query/', views.query, name='query'),

]
