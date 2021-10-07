from array import array
import base64
from django.db import models
import base64
import math
import random
import numpy
import pandas

from PIL import Image


# Create your models here.

class ImageData:
    def save_to_file(self, filename='one_neuron_perceptron/data/test_image.png', image_data_base64="", image_data_array=numpy.array([])):
        if image_data_base64 != "":
            with open(filename, 'wb') as file_to_save:
                file_to_save.write(base64.decodebytes(
                    image_data_base64.split(',')[1].encode('utf-8')))
        elif image_data_array.size != 0:
            #print('save_to_file', image_data_array.shape)
            Image.fromarray(image_data_array).save(
                'one_neuron_perceptron/data/converted_image.png')

    def read_from_file(self, filename='one_neuron_perceptron/data/test_image.png', convert='L'):
        return numpy.array(Image.open(filename).convert(convert))

    def encode_image_data_array(self, image_data_array=numpy.array([])):
        # image_data_array = numpy.array(image_data_array)
        if image_data_array.size != 0:
            # print('encode_image_data_array', image_data_array.shape)
            return numpy.array([[0 if element == 255 else 1 for element in row] for row in image_data_array])

    def decode_image_data_array(self, image_data_array=[]):
        if image_data_array.size != 0:
            # print('decode_image_data_array', image_data_array.shape)
            return numpy.array([[((1 - element) * 255).astype(numpy.uint8) for element in row] for row in image_data_array])

    def get_encoded_image_data_from_file(self, filename='one_neuron_perceptron/data/test_image.png', convert='L'):
        # return self.encode_image_data_array(image_data_array = self.read_from_file( filename=filename, convert=convert))
        # print('get_encoded_image_data_from_file')
        return numpy.array(self.encode_image_data_array(self.read_from_file(filename=filename, convert=convert)))

    def set_decoded_image_data_to_file(self, filename='one_neuron_perceptron/data/test_image.png', image_data_array=numpy.array([])):
        if image_data_array.size != 0:
            # print('set_decoded_image_data_to_file', image_data_array.shape)
            self.save_to_file(filename=filename, image_data_array=self.decode_image_data_array(
                image_data_array=image_data_array))


class ModelData(ImageData):
    def __init__(self, filename='one_neuron_perceptron/data/test_image.png', convert='L', grid_size=20) -> None:
        super().__init__()
        self.data = self.get_encoded_image_data_from_file(
            filename=filename, convert=convert)
        if not self.is_empty():
            self.data = self.crop()
            self.data = self.compress(grid_size=grid_size)
            self.set_decoded_image_data_to_file(image_data_array=self.data)

    def is_empty(self):
        return numpy.sum(self.data == 1) == 0 

    def crop(self):
        # print('clean', numpy.sum(self.data==0),  numpy.sum(self.data==1))
        fill_rows, fill_columns = numpy.where(self.data == 1)
        if fill_rows.size !=0 and fill_columns.size != 0:
            fill_row_start, fill_row_end, fill_column_start, fill_column_end = numpy.min(
                fill_rows), numpy.max(fill_rows), numpy.min(fill_columns), numpy.max(fill_columns)
            image_data_array_cropped = self.data[fill_row_start:fill_row_end,
                                                fill_column_start:fill_column_end]
            
            return image_data_array_cropped

    def compress(self, grid_size=20):
        return self.data[::self.data.shape[0]//grid_size if self.data.shape[0]//grid_size > 0 else 1, ::self.data.shape[1]//grid_size if self.data.shape[1]//grid_size > 0 else 1]


class DataFrame():
    def __init__(self, filenames=[], extra_columns=[], convert='L', grid_size=20, data=[], columns=[], extra_columns_labels=[]) -> None:
        if data == [] and  columns == []:
            for filename in filenames:
                data.append(numpy.array(ModelData(filename=filename, convert=convert,
                            grid_size=grid_size).data).ravel()[:grid_size*grid_size])
            for i in range(grid_size):
                for j in range(grid_size):
                    columns.append(str(i)+"x" + str(j))

        self.dataframe = pandas.DataFrame(data, columns=columns)

        if extra_columns_labels != []:
            for index, label in enumerate(extra_columns_labels):
                self.dataframe[label] = extra_columns[index]
        print(self.dataframe)

        # self.save_to_file()
        # self.read_from_file()

    def save_to_file(self, filename='one_neuron_perceptron/data/test.csv'):
        self.dataframe.to_csv(filename)

    def read_from_file(self, filename='one_neuron_perceptron/data/test.csv'):
        self.dataframe = pandas.read_csv(filename)
        print(self.dataframe)

class NeuralNetwork:
    def __init__(self, input_nodes=0, hidden_nodes=0, output_nodes=0, learning_rate=0.5) -> None:
        self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate = input_nodes, hidden_nodes, output_nodes, learning_rate

    def train(self,inputs_list, targets_list) -> None:
        pass

    def query(self) -> None:
        pass


class OneNeuronPerceptron(NeuralNetwork):
    def __init__(self, learning_rate = 0.5, size = 400) -> None:
        super().__init__(input_nodes=1, hidden_nodes=0, output_nodes=0, learning_rate=learning_rate)
        self.init_random_weights = lambda size=1, range = (-0.3, 0.3): range[0] + numpy.random.random_sample((size)) * range[1]
        self.size = size
        self.weights = self.init_random_weights(size=size)
        self.activation_function = lambda output: 0 if output < 0 else 1
        self.get_error = lambda target, output: target - output
        # self.query = lambda input_list: self.activation_function(numpy.sum(numpy.multiply(self.weights, numpy.array(input_list).ravel())))
        
    def train(self, input_list, target=0) -> None:
        self.weights += self.learning_rate * self.get_error(target, self.query(input_list)) * input_list
        print(numpy.mean(self.weights))

    def query(self, input_list=[]) -> None:
        return self.activation_function(numpy.sum(numpy.multiply(self.weights, input_list.ravel()[:self.size])))


    def save_weights_to_file(self, filename='one_neuron_perceptron/data/weights.csv'):
        dataframe = DataFrame(data=[self.weights])
        dataframe.save_to_file(filename)

    def read_weights_to_file(self, filename='one_neuron_perceptron/data/weights.csv'):
        dataframe = DataFrame(data=[self.weights])
        dataframe.read_from_file(filename)
        return dataframe.dataframe.to_numpy()[0]

perceptron = OneNeuronPerceptron()