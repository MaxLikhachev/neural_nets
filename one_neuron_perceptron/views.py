import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from one_neuron_perceptron.models import ImageData, ModelData, DataFrame, perceptron
from neural_nets.settings import PREDICTED_SYMBOLS
# Create your views here.

# TODO: Handle html-buttons for answers
# TODO: Add updating to dataframe
# TODO: Generate dataframe & train

def index(request):
    # print(request.POST['img'])
    model_data = ModelData()

    # image_data = ImageData()
    # image_data_array = image_data.get_encoded_image_data_from_file()
    # print('index', image_data_array.shape)
    # image_data.set_decoded_image_data_to_file(image_data_array=image_data_array)

    # dataframe = DataFrame(filenames=['one_neuron_perceptron/data/test_image.png'], extra_columns=[[0]], convert='L', grid_size=20, extra_columns_labels=['label'])

    
    # print(perceptron.train(input_list = dataframe.dataframe.to_numpy()[0][1:], target=0))
    # print(perceptron.query(input_list = dataframe.dataframe.to_numpy()[0][1:]))
    # matplotlib.pyplot.imshow(arr, cmap='Greys', interpolation='None')
    context = {'answer' : 'Это ' + PREDICTED_SYMBOLS[perceptron.query(input_list=model_data.data)] + ' ?'}
    return render(request, 'one_neuron_perceptron/canvas.html', context)

def train(request):
    return HttpResponse("You're looking at question %s.")

def query(request):
    return HttpResponse("You're looking at question %s.")

def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)

def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)

@csrf_exempt
def get_image(request):
    if request.method == "POST":
        image_data = ImageData()
        image_data.save_to_file(image_data_base64=json.loads(request.body)['img'])
        model_data = ModelData()
        print(perceptron.query(input_list=model_data.data))
    

        
    return JsonResponse({"answer":perceptron.query(input_list=model_data.data)})