from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # one_neuron_perceptron
    path('one_neuron_perceptron_train/', views.one_neuron_perceptron_train, name='train'),
    path('one_neuron_perceptron_query/', views.one_neuron_perceptron_query, name='query'),
    # hopfield
    path('hopfield_train/', views.hopfield_train, name='train'),
    path('hopfield_detrain/', views.hopfield_detrain, name='detrain'),
    path('hopfield_query/', views.hopfield_query, name='query'),
    path('hopfield_restart/', views.hopfield_restart, name='restart')
]
