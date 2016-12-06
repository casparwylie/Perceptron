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
	characters_to_retreive = 20000
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

	frame_height = 200
	frame_width = 500

	def __init__(self, tk_main, neural_network):
		self.tk_main = tk_main
		self.ui_frame = Frame(self.tk_main)
		self.tk_main.title("Neural Network Constructor")
		self.ui_frame.pack()
		self.neural_network = neural_network
		self.tk_main.minsize(width=self.frame_width, height=self.frame_height)
 		self.tk_main.maxsize(width=self.frame_width, height=self.frame_height)
 		self.quit_opt = Button(self.ui_frame, text="QUIT", fg="red", command=self.ui_frame.quit)
 		self.quit_opt.pack()
 		self.render_ui_widgets()

 	def render_ui_widgets(self):
 		start_learning_opt = Button(self.ui_frame, text="START LEARNING", command=self.neural_network.learn_analyse_iteration)
 		start_learning_opt.pack()

 		show_visual_nn_opt = Button(self.ui_frame, text="SEE NETWORK VISUALIZATION", command=self.render_neural_net_visualization)
 		show_visual_nn_opt.pack()

	def render_neural_net_visualization(self):
		canvas_height = 700
		canvas_width = 1300
		self.ui_nn_frame = Toplevel(self.tk_main)
		self.ui_nn_frame.title("Neural Network Visualization")
		tk_nn_visual_canvas = Canvas(self.ui_nn_frame, width=canvas_width, height=canvas_height,background="grey")
		tk_nn_visual_canvas.pack()
		nn_perceptrons = self.neural_network.nn_perceptrons
		biases_for_non_input_layers = self.neural_network.biases_for_non_input_layers

		example_p_limit_count = 20 #zero for all
		perceptron_radius = 10
		perceptron_x = 50
		perceptron_dist_x = 400
		perceptron_padding = 5
		bias_pos_diff_x = 50
		bias_pos_diff_y = 50
		bias_pos_y = perceptron_radius*2
		highest_layer_count = 0
		highest_layer_height = 0

		def get_layer_height_px(layer_count):
			return (layer_count*(perceptron_radius*2 + perceptron_padding))

		for perceptron_layer in range(0,len(nn_perceptrons)):
			length_of_layer = len(nn_perceptrons[perceptron_layer])
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count
			curr_layer_height = get_layer_height_px(length_of_layer)
			if(curr_layer_height > highest_layer_height):
				highest_layer_height = curr_layer_height
				highest_layer_count = perceptron_layer


		for perceptron_layer in range(0,len(nn_perceptrons)):
			
			length_of_layer = len(nn_perceptrons[perceptron_layer])
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count

			perceptron_ystart = (canvas_height - get_layer_height_px(length_of_layer))/2
			perceptron_y = perceptron_ystart
			layer_has_bias = ((perceptron_layer > 0) and (biases_for_non_input_layers[perceptron_layer-1] != 0))
			
			if layer_has_bias == True:
				if(biases_for_non_input_layers[perceptron_layer-1] == True):
					bias_y_pos = (canvas_height - highest_layer_height)/2 - bias_pos_diff_y
					bias_x_pos = perceptron_x-bias_pos_diff_x
					tk_nn_visual_canvas.create_oval(bias_x_pos-perceptron_radius,bias_y_pos-perceptron_radius,bias_x_pos+perceptron_radius,bias_y_pos+perceptron_radius)
			
			for single_perceptron in range(0,length_of_layer):
				tk_nn_visual_canvas.create_oval(perceptron_x-perceptron_radius,perceptron_y-perceptron_radius,perceptron_x+perceptron_radius,perceptron_y+perceptron_radius)
				if(layer_has_bias == True):
					tk_nn_visual_canvas.create_line(perceptron_x, perceptron_y, bias_x_pos,bias_y_pos)

				perceptron_dist_y = (perceptron_radius*2) + perceptron_padding
				if(perceptron_layer < len(nn_perceptrons)-1):
					length_of_next_layer = len(nn_perceptrons[perceptron_layer+1])
					if(example_p_limit_count > 0 and example_p_limit_count < length_of_next_layer):
						length_of_next_layer = example_p_limit_count
					perceptron_y_for_line = (canvas_height - (length_of_next_layer)*(perceptron_radius*2 + perceptron_padding))/2
					for perceptron_weights in range(0,length_of_next_layer):
						tk_nn_visual_canvas.create_line(perceptron_x, perceptron_y, perceptron_x+perceptron_dist_x, perceptron_y_for_line)
						perceptron_y_for_line += perceptron_dist_y

				perceptron_y += perceptron_dist_y
			perceptron_x += perceptron_dist_x
		
class neural_network_handler:

	#construct object to develop specific network structure
	def __init__(self, hidden_layers,
	 				input_count, output_count, matrix_data,matrix_targets,
	  				biases_for_non_input_layers, learning_constant, testing_mode):
		
		self.all_weights = []
		self.nn_perceptrons = []
		self.weight_change_record = []
		self.biases_weights = []
		self.biases_weight_change_record = []

		self.matrix_data = matrix_data
		self.hidden_layers = hidden_layers
		self.matrix_targets = matrix_targets
		self.learning_constant = learning_constant
		self.output_count = output_count
		self.input_count = input_count
		self.testing_mode = testing_mode
		self.biases_for_non_input_layers = biases_for_non_input_layers
		

		self.populate_nn_perceptrons()
		self.populate_all_weights()

	def populate_nn_perceptrons(self):
		nn_inputs = np.zeros(self.input_count)
		nn_outputs = np.zeros(self.output_count)
		self.nn_perceptrons.append(nn_inputs)
		for i in self.hidden_layers:
			hidden_layer = np.zeros(i)
			self.nn_perceptrons.append(hidden_layer)
		self.nn_perceptrons.append(nn_outputs)

	def populate_all_weights(self):
		for perceptron_layer in range(1, len(self.nn_perceptrons)):
			weight_layer = []
			weight_change_record_layer = []
			layer_length = len(self.nn_perceptrons[perceptron_layer])
			for single_perceptron in range(0, layer_length):
				prev_layer_count = len(self.nn_perceptrons[perceptron_layer-1])
				perceptron_weights = self.initilize_weights(prev_layer_count)
				weights_change_record_perceptron = np.zeros(prev_layer_count)
		
				weight_layer.append(perceptron_weights)
				weight_change_record_layer.append(weights_change_record_perceptron)

			self.all_weights.append(weight_layer)
			self.weight_change_record.append(weight_change_record_layer)

		for layer_count in range(0, len(self.biases_for_non_input_layers)):
			single_bias_weights = []
			single_bias_weights_change = []
			if(self.biases_for_non_input_layers[layer_count]!=0):
				bias_input_count = len(self.nn_perceptrons[layer_count+1])
				single_bias_weights = self.initilize_weights(bias_input_count)
				single_bias_weights_change = np.zeros(bias_input_count)
			self.biases_weights.append(single_bias_weights)
			self.biases_weight_change_record.append(single_bias_weights_change)


	def initilize_weights(self,size):
		return np.random.uniform(low=-1, high=1, size=(size))

	def learn_feed_forward(self, matrix):
		self.populate_input_layer(matrix)
		for after_input_layer in range(1, len(self.nn_perceptrons)):
			for perceptron_count in range(0, len(self.nn_perceptrons[after_input_layer])):
				relevant_weights = self.all_weights[after_input_layer-1][perceptron_count]
				outputs_feed_through_weights = np.dot(relevant_weights, self.nn_perceptrons[after_input_layer-1])
				hidden_perceptron_sum = outputs_feed_through_weights.sum()
				#print("sum",hidden_perceptron_sum)
				if(len(self.biases_weights[after_input_layer-1])!=0):
					hidden_perceptron_sum += self.biases_for_non_input_layers[after_input_layer-1] * self.biases_weights[after_input_layer-1][perceptron_count]
				self.nn_perceptrons[after_input_layer][perceptron_count] = self.activate_threshold(hidden_perceptron_sum, "sigmoid")
				#print("act",self.nn_perceptrons[after_input_layer][perceptron_count])



	def populate_input_layer(self, data):
		value_count = 0
		for col in range(0,len(data)):
			if(type(data[col]) is int):
				self.nn_perceptrons[0][value_count] = data[col]
				value_count += 1
			else:
				for row in range(0,len(data[col])):
					self.nn_perceptrons[0][value_count] = data[col][row]
					value_count += 1

	
	def process_single_full_back_prop(self,input_perceptron_val, act_to_sum, prev_step_back_prop_error_val):
		final_activated_to_weight = act_to_sum * input_perceptron_val
		full_step_back_val = prev_step_back_prop_error_val * final_activated_to_weight
		return full_step_back_val


	test_counter = 0
	test_print_interval = 100
	interval_correct_count = 0
	def learn_back_propagation(self, target_val):
		
		if(len(self.nn_perceptrons[-1])>1):
			target_vector = self.populate_target_vector(target_val)
		else:
			target_vector = [target_val]
		output_error_total = 0

		outputs_as_list = self.nn_perceptrons[-1].tolist()
		if(outputs_as_list.index(max(outputs_as_list))==target_val):
			self.interval_correct_count += 1


		if(self.test_counter % self.test_print_interval == 0):
			print(self.test_counter)
			print(str(self.interval_correct_count)+"%")
			self.interval_correct_count = 0
			print("")

		for weight_layer_count in range(len(self.all_weights)-1,-1,-1):
			for weight_perceptron_count in range(0, len(self.all_weights[weight_layer_count])):
				weight_perceptron_val = self.nn_perceptrons[weight_layer_count+1][weight_perceptron_count]
				final_activated_to_sum_step = weight_perceptron_val * (1-weight_perceptron_val) #if sigmoid
				
				if(weight_layer_count == len(self.all_weights)-1):
					prev_step_back_prop_error_val = weight_perceptron_val - target_vector[weight_perceptron_count]
					output_error_total += (0.5*prev_step_back_prop_error_val)**2
	
				for single_weight_count in range(0, len(self.all_weights[weight_layer_count][weight_perceptron_count])):
					current_weight_val = self.all_weights[weight_layer_count][weight_perceptron_count][single_weight_count]
					
					if(weight_layer_count != len(self.all_weights)-1):
						weight_change_layer_to_sum = weight_layer_count + 1
						prev_step_back_prop_error_val  = 0
						for weight_change_perceptron_count in range(0, len(self.weight_change_record[weight_change_layer_to_sum])):
							previous_weight_change = self.weight_change_record[weight_change_layer_to_sum][weight_change_perceptron_count][weight_perceptron_count]
							prev_step_back_prop_error_val += previous_weight_change

						if(len(self.biases_weights[weight_layer_count])>0):
							prev_step_back_prop_error_val += self.biases_weight_change_record[weight_layer_count][weight_perceptron_count]

					weight_input_perceptron_val = self.nn_perceptrons[weight_layer_count][single_weight_count]
					full_step_back_val = self.process_single_full_back_prop(weight_input_perceptron_val, final_activated_to_sum_step, prev_step_back_prop_error_val)
					self.weight_change_record[weight_layer_count][weight_perceptron_count][single_weight_count] = full_step_back_val
					new_weight_val = current_weight_val - (self.learning_constant * full_step_back_val)
					self.all_weights[weight_layer_count][weight_perceptron_count][single_weight_count] = new_weight_val


				if(len(self.biases_weights[weight_layer_count])>0):
					current_bias_weight_val = self.biases_weights[weight_layer_count][weight_perceptron_count]
					full_step_back_for_bias_weight = self.process_single_full_back_prop(self.biases_for_non_input_layers[weight_layer_count], final_activated_to_sum_step, prev_step_back_prop_error_val)
					self.biases_weight_change_record[weight_layer_count][weight_perceptron_count] = full_step_back_for_bias_weight
					new_bias_weight_val = current_bias_weight_val - (self.learning_constant * full_step_back_for_bias_weight)
					self.biases_weights[weight_layer_count][weight_perceptron_count] = new_bias_weight_val
		self.test_counter += 1

	def learn_analyse_iteration(self):
		repeat_count = 1
		if(self.testing_mode == True):
			repeat_count = 5000
		for i in range(0,repeat_count):
			matrix_count = 0
			for matrix in self.matrix_data:
				target_val = self.matrix_targets[matrix_count]
				self.learn_feed_forward(matrix)
				self.learn_back_propagation(target_val)
				
				matrix_count += 1


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
		input_perceptron_count = 2
		hidden_layers = [4] 
		output_perceptron_count = 1
		matrix_data = [[1,1],[0,1],[1,0],[0,0]]
		matrix_targets = [1,0,0,1]
		biases_for_non_input_layers = [1,1]

	else:
		character_matrix_data = character_matrix_data_handler()
		character_matrix_data.populate_character_matrices()
		input_perceptron_count = character_matrix_data_handler.matrix_width * character_matrix_data_handler.matrix_width
		hidden_layers = [500]
		biases_for_non_input_layers = [1,1]
		matrix_data = character_matrix_data.character_matrices
		matrix_targets = character_matrix_data.character_targets
		output_perceptron_count = 10
	
	learning_constant = 0.5

	if(len(biases_for_non_input_layers) != len(hidden_layers)+1):
		print("bias count mismatch")

	
	neural_network = neural_network_handler(hidden_layers,										
								input_perceptron_count,
								output_perceptron_count,
								matrix_data,
								matrix_targets,
								biases_for_non_input_layers,
								learning_constant,
								testing_mode)


	tk_main = Tk()
	user_interface = user_interface_handler(tk_main, neural_network)
	tk_main.mainloop()
	
main()
