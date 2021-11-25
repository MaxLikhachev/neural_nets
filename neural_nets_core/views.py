import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas
import numpy
from neural_nets_core.models import ImageData, ModelData, DataFrame, OneNeuronPerceptron, perceptron, hopefield
from neural_nets.settings import PREDICTED_SYMBOLS
# Create your views here.

# TODO: Check training
# TODO: Add updating to dataframe
# TODO: Generate dataframe & train


def index(request):
    # print(request.POST['img'])
    model_data = ModelData()
    print('get_image', perceptron.query(input_list=model_data.data))

    context = {'answer': 'Это ' +
               PREDICTED_SYMBOLS[perceptron.query(input_list=model_data.data)] + ' ?'}
    return render(request, 'neural_nets_core/canvas.html', context)


@csrf_exempt
def one_neuron_perceptron_train(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    perceptron.train(input_list=model_data.data.ravel(), target=1 if perceptron.query(
        input_list=model_data.data.ravel()) == 0 else 0)
    return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})


@csrf_exempt
def one_neuron_perceptron_query(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})


@csrf_exempt
def hopfield_train(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    image_data_array=hopefield.train(input_list=model_data.data)
    model_data.set_decoded_image_data_to_file(image_data_array=image_data_array)
    print('hopfield_train',image_data_array) 
    with open('neural_nets_core/data/generated_image.png', 'rb') as f:
        contents = f.read()
    return HttpResponse(contents)


@csrf_exempt
def hopfield_query(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    image_data_array=hopefield.query(input_list=model_data.data)
    model_data.set_decoded_image_data_to_file(image_data_array=image_data_array)
    print('hopfield_query', image_data_array)
    with open('neural_nets_core/data/generated_image.png', 'rb') as f:
        contents = f.read()
    return HttpResponse(contents)

# data, labels = [], []


@csrf_exempt
def test(request):
    if request.method == "POST":
        model_data = ModelData(
            image_data_base64=json.loads(request.body)['img'])
        return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})
