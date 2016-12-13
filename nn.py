from __future__ import print_function
import cv2
import re
import numpy as np
import random
import math
from pprint import pprint
from Tkinter import *
from decimal import *
import os

'''
class data_preprocessor_handler:

	def image_dir_to_matrix_txt(self, dirname):
		new_txt_file = open(dirname+".txt", "a")
		image_file_names = os.listdir(dirname)
		for image_file_name in image_file_names:
		 	if(image_file_name[0:1] != "."):
		 		print(image_file_name)

'''


class character_matrix_data_handler:
	matrix_width = 28
	characters_to_retreive = 10
	data_set = open('newdataset.txt', 'r').read().split(",")
	character_matrices = []
	character_targets = []

	def populate_character_matrices(self):
		px_count = 0
		for i in range(self.characters_to_retreive):
			matrix = np.zeros((self.matrix_width,self.matrix_width), dtype=np.float32)
			for px_col in range(self.matrix_width):
				for px_row in range(self.matrix_width):
					if(px_count%(self.matrix_width**2)==0):
						self.character_targets.append(int(self.data_set[px_count]))
					else:
						matrix[px_col][px_row] = float(self.data_set[px_count]) / 255.0
					px_count += 1
			self.character_matrices.append(matrix)


class user_interface_handler:

	frame_height = 800
	frame_width = 1200

	def __init__(self, tk_main, neural_network):
		self.tk_main = tk_main
		self.ui_frame = Frame(self.tk_main)
		self.tk_main.title("Neural Network Constructor")
		self.ui_frame.grid()
		self.neural_network = neural_network
		self.tk_main.minsize(width=self.frame_width, height=self.frame_height)
 		self.tk_main.maxsize(width=self.frame_width, height=self.frame_height)
 
 		self.render_ui_widgets()
 		self.render_neural_net_visualization()

 	def render_ui_widgets(self):
	 		
	 	min_x_pos = 110
 		min_y_pos = 10
 		option_width = 30
 		x_center_pos = (self.frame_width/2) - option_width/2
 		option_height = 30
 		max_col,max_row = self.ui_frame.grid_size()

 		def render_option(text, command,pos):
 			option = Button(self.ui_frame, text=text, command=command)
 			#option.place(x=pos["row"],  y=pos["col"])
 			return option

 		def render_input_field(default_value, label_text, pos,width):
 			text_input = Entry(self.ui_frame, width=width)
 			text_input.insert(0,str(default_value))
 			text_input.place(x=pos["row"], y=pos["col"])
 			input_label = Label(self.ui_frame, text=label_text+": ")
 			input_label.place(x=pos["row"], y=pos["col"]-1)
 			return text_input


 		start_learning_opt = render_option("START LEARNING", self.neural_network.learn_analyse_iteration,{"row": 40, "col":0})
 		show_visual_nn_opt = render_option("NETWORK VISUALIZATION", self.render_neural_net_visualization, {"row":2, "col":5})

 		learning_rate_input_label = render_input_field("0.5", "Learning Rate", {"row": 5,"col":1},10)
 		learning_rate_input_label = render_input_field("30,30", "Hidden Layers", {"row": 6,"col":1},4)


	def render_neural_net_visualization(self):
		canvas_height = 500
		canvas_width = 1100
		#self.ui_nn_frame = Toplevel(self.tk_main)
		#self.ui_nn_frame.title("Neural Network Visualization")
		tk_nn_visual_canvas = Canvas(self.ui_frame, width=canvas_width, height=canvas_height,background="grey")
		tk_nn_visual_canvas.place(x=100, y=1)
		nn_neurons = self.neural_network.nn_neurons
		biases_for_non_input_layers = self.neural_network.biases_for_non_input_layers

		example_p_limit_count = 25 #zero for all

		highest_layer_count = max([len(neurons) for neurons in nn_neurons])
		if(highest_layer_count > example_p_limit_count):
			highest_layer_count = example_p_limit_count

		highest_layer_height = 0

		neuron_padding = 5		
		neuron_radius = ((canvas_height / highest_layer_count)/2)-neuron_padding
		if(neuron_radius > 30): neuron_radius = 30
		neuron_x = neuron_radius + 10
		neuron_dist_x = (canvas_width / (len(nn_neurons)-1)) - neuron_x
		neuron_color = "blue"

		bias_pos_diff_x = 50
		bias_pos_diff_y = 30
		bias_color = "green"
		bias_pos_y = neuron_radius*2

		def get_layer_height_px(layer_count):
			return (layer_count*(neuron_radius*2 + neuron_padding))

		for neuron_layer in range(0,len(nn_neurons)):
			length_of_layer = len(nn_neurons[neuron_layer])
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count
			curr_layer_height = get_layer_height_px(length_of_layer)
			if(curr_layer_height > highest_layer_height):
				highest_layer_height = curr_layer_height

		

		for neuron_layer in range(0,len(nn_neurons)):
			
			length_of_layer = len(nn_neurons[neuron_layer])
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count

			neuron_ystart = (canvas_height - get_layer_height_px(length_of_layer))/2
			neuron_y = neuron_ystart
			layer_has_bias = ((neuron_layer > 0) and (biases_for_non_input_layers[neuron_layer-1] != 0))
			
			if layer_has_bias == True:
				if(biases_for_non_input_layers[neuron_layer-1] == True):
					bias_y_pos = (canvas_height - highest_layer_height)/2 - bias_pos_diff_y
					bias_x_pos = neuron_x-bias_pos_diff_x
					bias_oval = tk_nn_visual_canvas.create_oval(bias_x_pos-neuron_radius,bias_y_pos-neuron_radius,bias_x_pos+neuron_radius,bias_y_pos+neuron_radius, fill=bias_color,outline=bias_color)
					tk_nn_visual_canvas.tag_raise(bias_oval)

			for single_neuron in range(0,length_of_layer):
				neuron_oval = tk_nn_visual_canvas.create_oval(neuron_x-neuron_radius,neuron_y-neuron_radius,neuron_x+neuron_radius,neuron_y+neuron_radius,fill=neuron_color,outline=neuron_color)
				tk_nn_visual_canvas.tag_raise(neuron_oval)

				if(layer_has_bias == True):
					bias_connector = tk_nn_visual_canvas.create_line(neuron_x, neuron_y, bias_x_pos,bias_y_pos)
					tk_nn_visual_canvas.tag_lower(bias_connector)

				neuron_dist_y = (neuron_radius*2) + neuron_padding
				if(neuron_layer < len(nn_neurons)-1):
					length_of_next_layer = len(nn_neurons[neuron_layer+1])
					if(example_p_limit_count > 0 and example_p_limit_count < length_of_next_layer):
						length_of_next_layer = example_p_limit_count
					neuron_y_for_line = (canvas_height - (length_of_next_layer)*(neuron_radius*2 + neuron_padding))/2
					
					for neuron_weights in range(0,length_of_next_layer):
						neuron_connector = tk_nn_visual_canvas.create_line(neuron_x, neuron_y, neuron_x+neuron_dist_x, neuron_y_for_line)
						tk_nn_visual_canvas.tag_lower(neuron_connector)

						neuron_y_for_line += neuron_dist_y

				neuron_y += neuron_dist_y
			neuron_x += neuron_dist_x
		
class neural_network_handler:

	#construct object to develop specific network structure
	def __init__(self, hidden_layers,
	 				input_count, output_count, matrix_data,matrix_targets,
	  				biases_for_non_input_layers, learning_constant, testing_mode):
		
		self.all_weights = []
		self.nn_neurons = []
		self.weight_changes = []
		self.biases_weights = []
		self.biases_weight_changes = []

		self.matrix_data = matrix_data
		self.hidden_layers = hidden_layers
		self.matrix_targets = matrix_targets
		self.learning_constant = learning_constant
		self.output_count = output_count
		self.input_count = input_count
		self.testing_mode = testing_mode
		self.biases_for_non_input_layers = biases_for_non_input_layers
		
		self.success_records = []
		self.populate_nn_neurons()
		self.populate_all_weights()

	def populate_nn_neurons(self):
		nn_inputs = np.zeros(self.input_count)
		nn_outputs = np.zeros(self.output_count)
		self.nn_neurons.append(nn_inputs)
		for i in self.hidden_layers:
			hidden_layer = np.zeros(i)
			self.nn_neurons.append(hidden_layer)
		self.nn_neurons.append(nn_outputs)

	def populate_all_weights(self):
		for neuron_layer in range(1, len(self.nn_neurons)):
			layer_length = len(self.nn_neurons[neuron_layer])
			weight_layer = []
			weight_changes_layer = []
			for single_neuron in range(0, layer_length):
				prev_layer_count = len(self.nn_neurons[neuron_layer-1])
				neuron_weights = self.initilize_weights(prev_layer_count)
				weights_change_record_neuron = np.zeros(prev_layer_count)
		
				weight_layer.append(neuron_weights)
				weight_changes_layer.append(weights_change_record_neuron)

			self.all_weights.append(weight_layer)
			self.weight_changes.append(weight_changes_layer)

		for layer_count in range(0, len(self.biases_for_non_input_layers)):
			single_bias_weights = []
			single_bias_weights_change = []
			if(self.biases_for_non_input_layers[layer_count]!=0):
				bias_input_count = len(self.nn_neurons[layer_count+1])
				single_bias_weights = self.initilize_weights(bias_input_count)
				single_bias_weights_change = np.zeros(bias_input_count)
			self.biases_weights.append(single_bias_weights)
			self.biases_weight_changes.append(single_bias_weights_change)


	def initilize_weights(self,size):
		return np.random.uniform(low=-1, high=1, size=(size))

	def feed_forward(self, matrix):
		self.populate_input_layer(matrix)
		for after_input_layer in range(1, len(self.nn_neurons)):
			for neuron_count in range(0, len(self.nn_neurons[after_input_layer])):
				relevant_weights = self.all_weights[after_input_layer-1][neuron_count]
				hidden_neuron_sum = np.dot(relevant_weights, self.nn_neurons[after_input_layer-1])
				if(len(self.biases_weights[after_input_layer-1])!=0):
					hidden_neuron_sum += self.biases_for_non_input_layers[after_input_layer-1] * self.biases_weights[after_input_layer-1][neuron_count]
				self.nn_neurons[after_input_layer][neuron_count] = self.activate_threshold(hidden_neuron_sum, "sigmoid")



	def populate_input_layer(self, data):
		value_count = 0
		if(type(data[0]) is not int):
			self.nn_neurons[0] = data.flatten()
		else:
			self.nn_neurons[0] = np.array(data)

	
	testing_output_mode = False
	test_counter = 0
	repeat_count = 50
	correct_count = 0
	test_data_amount = 10000
	
	def back_propagate(self, target_val,repeat_count):
		
		if(len(self.nn_neurons[-1])>1 and type(target_val) is int):
			target_vector = self.populate_target_vector(target_val)
		else:
			target_vector = target_val
		output_error_total = 0

		
		outputs_as_list = self.nn_neurons[-1].tolist()

		if(self.test_counter >= len(self.matrix_data)-self.test_data_amount):

			if(outputs_as_list.index(max(outputs_as_list))==target_val):

				self.correct_count += 1

		for weight_layer_count in range(len(self.all_weights)-1,-1,-1):
			weight_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count+1],axis=1)
			target_vector = np.expand_dims(target_vector,axis=1)
			activated_to_sum_step = weight_neuron_vals * (1-weight_neuron_vals)
			if(weight_layer_count == len(self.all_weights)-1):
				back_prop_cost_to_sum = (weight_neuron_vals - target_vector) * activated_to_sum_step
			else:
				back_prop_cost_to_sum = np.dot(np.asarray(self.all_weights[weight_layer_count+1]).transpose(),back_prop_cost_to_sum) * activated_to_sum_step
			input_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count],axis=1)
			full_back_prop_sum_to_input = np.dot(back_prop_cost_to_sum,input_neuron_vals.transpose())
			current_weight_vals = self.all_weights[weight_layer_count]
			new_weight_vals = current_weight_vals - (self.learning_constant * full_back_prop_sum_to_input)
			self.all_weights[weight_layer_count] = new_weight_vals
	
		self.test_counter += 1


	def learn_analyse_iteration(self):
		
		if(self.testing_mode == True):
			self.repeat_count = 5000
		for i in range(0,self.repeat_count):
			matrix_count = 0
			for matrix in self.matrix_data:
				target_val = self.matrix_targets[matrix_count]
				self.feed_forward(matrix)
				self.back_propagate(target_val,i)
				matrix_count += 1
			
			success_p = (float(self.correct_count)/float(self.test_data_amount))*100
			print("Epoch " + str(i) + ", " + "AV SUCCESS RATE:"+str(success_p)+"%")
			self.test_counter = 0
			self.correct_count = 0

	#activation function for thresholding given values 
	def activate_threshold(self,value, type):
		if(type == "step"):
			if(value>=0.5):
				return 1
			else:
				return 0
		elif(type == "sigmoid"):
			return 1/(1+(math.exp(-value)))


	def populate_target_vector(self,target):
		vector = []
		for i in range(0,self.output_count):
			vector.append(0)
		vector[target] = 1
		return vector


def main():
	
	#neural network options
	testing_mode = False
	if(testing_mode == True):
		input_neuron_count = 2
		hidden_layers = [4] 
		output_neuron_count = 1
		matrix_data = [[1,1],[0,1],[1,0],[0,0]]
		matrix_targets = [[1],[0],[0],[1]]
		biases_for_non_input_layers = [1,1]

	else:
		character_matrix_data = character_matrix_data_handler()
		character_matrix_data.populate_character_matrices()
		input_neuron_count = character_matrix_data_handler.matrix_width * character_matrix_data_handler.matrix_width
		hidden_layers = [70]
		biases_for_non_input_layers = [0,0]
		matrix_data = character_matrix_data.character_matrices
		matrix_targets = character_matrix_data.character_targets
		output_neuron_count = 10
	
	learning_constant = 0.5

	if(len(biases_for_non_input_layers) != len(hidden_layers)+1):
		print("bias count mismatch")

	
	neural_network = neural_network_handler(hidden_layers,										
								input_neuron_count,
								output_neuron_count,
								matrix_data,
								matrix_targets,
								biases_for_non_input_layers,
								learning_constant,
								testing_mode)


	tk_main = Tk()
	user_interface = user_interface_handler(tk_main, neural_network)
	tk_main.mainloop()
	
main()
