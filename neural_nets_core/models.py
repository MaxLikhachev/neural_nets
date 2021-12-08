from array import array
import base64
from django.db import models
import base64
import numpy
import pandas
import random
import math

from PIL import Image


# Create your models here.

class ImageData:
    def __init__(self, filename='neural_nets_core/data/test_image.png', image_data_base64="", image_data_array=numpy.array([])) -> None:
        self.save_to_file(
            filename=filename, image_data_base64=image_data_base64, image_data_array=image_data_array)
        self.filename, self.image_data_base64, self.image_data_array = filename, image_data_base64, image_data_array if image_data_array.size != 0 else self.read_from_file(
            filename=filename)

    def save_to_file(self, filename='neural_nets_core/data/test_image.png', image_data_base64="", image_data_array=numpy.array([])):
        if image_data_base64 != "":
            with open(filename, 'wb') as file_to_save:
                file_to_save.write(base64.decodebytes(
                    image_data_base64.split(',')[1].encode('utf-8')))
        elif image_data_array.size != 0:
            #print('save_to_file', image_data_array.shape)
            Image.fromarray(image_data_array).save(filename)

    def read_from_file(self, filename='neural_nets_core/data/test_image.png', convert='L'):
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

    def get_encoded_image_data_from_file(self, filename='neural_nets_core/data/test_image.png', convert='L'):
        # return self.encode_image_data_array(image_data_array = self.read_from_file( filename=filename, convert=convert))
        # print('get_encoded_image_data_from_file')
        return numpy.array(self.encode_image_data_array(self.read_from_file(filename=filename, convert=convert)))

    def set_decoded_image_data_to_file(self, filename='neural_nets_core/data/generated_image.png', image_data_array=numpy.array([])):
        if image_data_array.size != 0:
            # print('set_decoded_image_data_to_file', image_data_array.shape)
            self.save_to_file(filename=filename, image_data_array=self.decode_image_data_array(
                image_data_array=image_data_array))


class ModelData(ImageData):
    def __init__(self, filename='neural_nets_core/data/test_image.png', convert='L', grid_size=20, image_data_base64="", image_data_array=numpy.array([])) -> None:
        super().__init__(filename=filename, image_data_base64=image_data_base64,
                         image_data_array=image_data_array)

        self.data = self.encode_image_data_array(
            image_data_array=self.image_data_array)

        self.compress = lambda grid_size=20: self.data[::self.data.shape[0]//grid_size if self.data.shape[0] //
                                                       grid_size > 0 else 1, ::self.data.shape[1]//grid_size if self.data.shape[1]//grid_size > 0 else 1]
        self.is_empty = lambda: numpy.sum(self.data == 1) == 0
        self.bipolar = lambda data=[], min_value = -1, max_value = 1: numpy.array(
            [min_value if element <= 0 else max_value for element in numpy.array(data).ravel()]).reshape(numpy.array(data).shape)

        if not self.is_empty():
            # self.data = self.crop()
            self.data = self.compress(grid_size=grid_size)
            self.data = self.data[:grid_size, :grid_size]
            self.set_decoded_image_data_to_file(image_data_array=self.data)
        # print('ModelData data')
        # print(self.data)

    def crop(self):
        # print('clean', numpy.sum(self.data==0),  numpy.sum(self.data==1))
        fill_rows, fill_columns = numpy.where(self.data == 1)
        if fill_rows.size != 0 and fill_columns.size != 0:
            fill_row_start, fill_row_end, fill_column_start, fill_column_end = numpy.min(
                fill_rows), numpy.max(fill_rows), numpy.min(fill_columns), numpy.max(fill_columns)
            image_data_array_cropped = self.data[fill_row_start:fill_row_end,
                                                 fill_column_start:fill_column_end]

            return image_data_array_cropped


class DataFrame():
    def __init__(self, filename='', filenames=[], extra_columns=[], convert='L', grid_size=20, data=[], columns=[], extra_columns_labels=[]) -> None:

        if data == [] and filenames.__len__() != 0:
            for filename in filenames:
                data.append(numpy.array(ModelData(filename=filename, convert=convert,
                            grid_size=grid_size).data).ravel()[:grid_size*grid_size])
        if columns == []:
            for i in range(grid_size):
                for j in range(grid_size):
                    columns.append(str(i)+"x" + str(j))

        self.dataframe = pandas.DataFrame(
            data, columns=columns) if filename == '' else self.read_from_file(filename=filename)

        if extra_columns_labels != []:
            for index, label in enumerate(extra_columns_labels):
                self.dataframe[label] = extra_columns[index]
        # print(self.dataframe)

        # self.save_to_file()
        # self.read_from_file()

    def save_to_file(self, filename='neural_nets_core/data/test.csv'):
        self.dataframe.to_csv(filename)

    def add_to_file(self, filename='neural_nets_core/data/test.csv'):
        # print(self.read_from_file())
        pandas.concat([self.read_from_file(), self.dataframe]).to_csv(filename)

    def read_from_file(self, filename='neural_nets_core/data/test.csv'):
        return pandas.read_csv(filename)

    def convert(self, grid_size=20):
        rows_temp, labels_temp = [], []
        for index, row in self.dataframe.iterrows():
            row_temp = []
            for i in range(grid_size):
                for j in range(grid_size):
                    row_temp.append(row[str(i)+'x'+str(j)])
            rows_temp.append(row_temp)
            labels_temp.append(row['label'])
        return(rows_temp, labels_temp)


class NeuralNetwork:
    def __init__(self, input_nodes=0, hidden_nodes=0, output_nodes=0, learning_rate=0.5) -> None:
        self.input_nodes, self.hidden_nodes, self.output_nodes, self.learning_rate = input_nodes, hidden_nodes, output_nodes, learning_rate

    def train(self, inputs_list, targets_list) -> None:
        pass

    def query(self) -> None:
        pass


class OneNeuronPerceptron(NeuralNetwork):
    def __init__(self, learning_rate=0.5, size=20*20, filename='neural_nets_core/data/weights.csv') -> None:
        super().__init__(input_nodes=1, hidden_nodes=0,
                         output_nodes=0, learning_rate=learning_rate)
        self.init_random_weights = lambda size=1, range = (
            -0.3, 0.3): range[0] + numpy.random.random_sample((size)) * range[1]
        self.size = size
        self.weights = self.init_random_weights(
            size=size) if filename == '' else self.read_weights_to_file()
        self.activate = lambda output: 0 if output < 0 else 1
        self.get_error = lambda target, output: target - output
        # self.query = lambda input_list: self.activate(numpy.sum(numpy.multiply(self.weights, numpy.array(input_list).ravel())))

    def train(self, input_list, target=0) -> None:
        # print(input_list.shape)
        # self.weights += self.learning_rate * self.get_error(target, self.query(input_list.ravel())) * input_list.ravel()
        self.weights = [weight + self.learning_rate * self.get_error(target, self.query(
            input_list)) * input_list[i] for i, weight in enumerate(self.weights)]
        # print(self.get_error(target, self.query(input_list)))
        # print(numpy.mean(self.weights))
        self.save_weights_to_file()

    def query(self, input_list=[]) -> None:
        return self.activate(numpy.sum(numpy.multiply(self.weights, input_list.ravel()[:self.size])))

    def save_weights_to_file(self, filename='neural_nets_core/data/weights.csv'):
        dataframe = DataFrame(data=[self.weights])
        dataframe.save_to_file(filename)

    def read_weights_to_file(self, filename='neural_nets_core/data/weights.csv'):
        dataframe = DataFrame(filename=filename)
        return dataframe.dataframe.to_numpy()[0][1:]


perceptron = OneNeuronPerceptron()


class Hopefield(NeuralNetwork):
    def __init__(self, size=400) -> None:
        self.weights = numpy.zeros([size, size])
        self.size, self.divider = size, 0

        self.activate = lambda output: -1.0 if output <= 0 else 1.0

    def divide(self, value):
        self.divider += 1
        return value / self.divider

    def create(self, input_list):
        return numpy.array([[0.0 if i == j else element for j, element in enumerate(row)] for i, row in enumerate(numpy.outer(input_list, input_list))])

    def query(self, input_list, epochs=100, output=[]):
        for epoch in range(epochs):
            temp = numpy.array([self.activate(numpy.sum(row)) for row in self.weights.dot(
                [[element] for element in input_list])])
            if numpy.array_equal(output, temp):
                break
            output = temp
        return output

    def train(self, input_list):
        self.weights = self.divide(self.weights + self.create(input_list))

    def detrain(self, input_list, threshold=0.5):
        self.weights -= numpy.dot(threshold, self.create(input_list))


hopefield = Hopefield()
