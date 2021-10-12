import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas
import numpy
from one_neuron_perceptron.models import ImageData, ModelData, DataFrame, OneNeuronPerceptron, perceptron, get_prediction, get_train
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
    return render(request, 'one_neuron_perceptron/canvas.html', context)
    # image_data = ImageData()
    # image_data_array = image_data.get_encoded_image_data_from_file()
    # print('index', image_data_array.shape)
    # image_data.set_decoded_image_data_to_file(image_data_array=image_data_array)

    # dataframe = DataFrame(filenames=['one_neuron_perceptron/data/test_image.png'], extra_columns=[[0]], convert='L', grid_size=20, extra_columns_labels=['label'])

    # print(perceptron.train(input_list = dataframe.dataframe.to_numpy()[0][1:], target=0))
    # print(perceptron.query(input_list = dataframe.dataframe.to_numpy()[0][1:]))
    # matplotlib.pyplot.imshow(arr, cmap='Greys', interpolation='None')


@csrf_exempt
def train(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    perceptron.train(input_list=model_data.data.ravel(), target=1 if perceptron.query(
        input_list=model_data.data.ravel()) == 0 else 0)
    return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})


@csrf_exempt
def query(request):
    model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
    return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})

# data, labels = [], []


@csrf_exempt
def test(request):
    if request.method == "POST":
        # model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
        """ dataframe = DataFrame(filename='one_neuron_perceptron/data/test.csv')
        # dataframe.drop(['index', 'label'], axis=1, inplace=True)
        print(dataframe)
        rows, labels_temp = dataframe.convert()
        print(rows, labels_temp)
        for index, row in enumerate(rows):
            perceptron.train(input_list=numpy.array(row), target=labels_temp[index]) """

        # image_data.save_to_file(image_data_base64=img)

        """ print('save-mode', json.loads(request.body)['save-mode'], data, labels)
        if json.loads(request.body)['save-mode'] != 'no-save':
            model_data = ModelData(image_data_base64=json.loads(request.body)['img'])
            data.append(model_data.data.ravel())
            labels.append(1) if json.loads(request.body)['save-mode'] == "save-1" else labels.append(0)
            dataframe = DataFrame(data=data,  extra_columns=labels, extra_columns_labels=['label'])
            dataframe.save_to_file() """
        model_data = ModelData(
            image_data_base64=json.loads(request.body)['img'])
        return JsonResponse({"answer": perceptron.query(input_list=model_data.data.ravel())})
