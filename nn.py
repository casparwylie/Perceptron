import cv2
import re
import numpy as np
import random
import math
from pprint import pprint


class matrix_data_handler:
	matrix_width = 28
	characters_to_retreive = 1
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

	frame_height = 1000
	frame_width = 1600
	canvas_height = frame_height * 0.9
	canvas_width = frame_width * 0.9


	def __init__(self):
		self.ui_canvas = np.zeros((self.canvas_height,self.canvas_width), dtype=np.uint8)
		
	def render_neural_net_visualization(self,nn_perceptrons):

		example_input_limit_count = 30 #zero for all
		perceptron_radius = 20
		perceptron_x = 100
		perceptron_dist_x = 400
		perceptron_padding = 5

		for perceptron_layer in range(0,len(nn_perceptrons)):
			
			length_of_layer = len(nn_perceptrons[perceptron_layer])
			
			if(perceptron_layer==0 and example_input_limit_count > 0 and example_input_limit_count < length_of_layer):
				length_of_layer = example_input_limit_count

			perceptron_ystart = (self.frame_height - (length_of_layer*(perceptron_radius*2 + perceptron_padding)))/2
			perceptron_y = perceptron_ystart

			for single_perceptron in range(0,length_of_layer):

				cv2.circle(self.ui_canvas, (perceptron_x,perceptron_y), perceptron_radius, (255), 1)

				perceptron_dist_y = (perceptron_radius*2) + perceptron_padding
				if(perceptron_layer < len(nn_perceptrons)-1):
					perceptron_y_for_line = (self.frame_height - (len(nn_perceptrons[perceptron_layer+1])*(perceptron_radius*2 + perceptron_padding)))/2
					for perceptron_weights in range(0,len(nn_perceptrons[perceptron_layer+1])):
						cv2.line(self.ui_canvas,(perceptron_x, perceptron_y), (perceptron_x+perceptron_dist_x, perceptron_y_for_line),(255),1)
						perceptron_y_for_line += perceptron_dist_y

				perceptron_y += perceptron_dist_y
			perceptron_x += perceptron_dist_x

		cv2.imshow("nn", self.ui_canvas)
		

class neural_network_handler:
	
	#declare key data for nn structure
	all_weights = []
	nn_perceptrons = []
	weight_change_record = []

	#construct object to develop specific network structure
	def __init__(self, hidden_layers, input_count, output_count, character_matrices,character_targets,learning_constant, testing_mode):

		#populate perceptrons
		nn_inputs = np.zeros(input_count)
		nn_outputs = np.zeros(output_count)
		self.nn_perceptrons.append(nn_inputs)
		self.character_matrices = character_matrices
		self.character_targets = character_targets
		self.learning_constant = learning_constant
		self.output_count = output_count
		self.testing_mode = testing_mode

		if(self.testing_mode == True):
			self.of_matrix_to_analyse = 2
			self.of_matrix_to_analyse_start = 10
		else:
			self.of_matrix_to_analyse = len(character_matrices[0])
			self.of_matrix_to_analyse_start = 0

		for i in hidden_layers:
			hidden_layer = np.zeros(i)
			self.nn_perceptrons.append(hidden_layer)

		self.nn_perceptrons.append(nn_outputs)

		#populate weights
		for perceptron_layer in range(1, len(self.nn_perceptrons)):
			weight_layer = []
			weight_change_record_layer = []
			layer_length = len(self.nn_perceptrons[perceptron_layer])
			for single_perceptron in range(0, layer_length):
				perceptron_weights = []
				weight_change_record_perceptron = []
				prev_layer_count = len(self.nn_perceptrons[perceptron_layer-1])
				for sing_percept_weight_count in range(0,prev_layer_count):
					perceptron_weights.append(random.uniform(0,1))
					weight_change_record_perceptron.append(0)

				weight_layer.append(perceptron_weights)
				weight_change_record_layer.append(weight_change_record_perceptron)

			self.all_weights.append(weight_layer)
			self.weight_change_record.append(weight_change_record_layer)


	def learn_feed_forward(self, matrix):
		px_count = 0
		for px_col in range(self.of_matrix_to_analyse_start,self.of_matrix_to_analyse_start+self.of_matrix_to_analyse):
			for px_row in range(self.of_matrix_to_analyse_start,self.of_matrix_to_analyse_start+self.of_matrix_to_analyse):
				self.nn_perceptrons[0][px_count] = matrix[px_col][px_row]
				px_count += 1

		for after_input_layer in range(1, len(self.nn_perceptrons)):
			for perceptron_count in range(0, len(self.nn_perceptrons[after_input_layer])):
				hidden_perceptron_sum = 0
				for prev_perceptron_count in range(0,len(self.nn_perceptrons[after_input_layer-1])): 
					prev_perceptron = self.nn_perceptrons[after_input_layer-1][prev_perceptron_count]
					relevant_weight = self.all_weights[after_input_layer-1][perceptron_count][prev_perceptron_count]
					hidden_perceptron_sum += prev_perceptron * relevant_weight
				
				self.nn_perceptrons[after_input_layer][perceptron_count] = self.activate_threshold(hidden_perceptron_sum, "sigmoid")


	test_counter = 0

	def learn_back_propagation(self, target_val):
		self.test_counter += 1
		if(self.testing_mode == True):
			target_val = random.randint(0, self.output_count-1)
		
		target_vector = self.populate_target_vector(target_val)

		if(self.nn_perceptrons[-1].tolist().index(max(self.nn_perceptrons[-1]))==target_val):
			print("Correct")

		output_error_total = 0
		for weight_layer_count in range(len(self.all_weights)-1,-1,-1):

			

			for weight_perceptron_count in range(0, len(self.all_weights[weight_layer_count])):
				weight_perceptron_val = self.nn_perceptrons[weight_layer_count+1][weight_perceptron_count]
				final_activated_to_sum_step = weight_perceptron_val * (1-weight_perceptron_val) #if sigmoid

				if(weight_layer_count == len(self.all_weights)-1):
					prev_step_back_prop_error_val = weight_perceptron_val - target_vector[weight_perceptron_count-1]
					output_error_total += (0.5*prev_step_back_prop_error_val)**2
	
				for single_weight_count in range(0, len(self.all_weights[weight_layer_count][weight_perceptron_count])):
					current_weight_val = self.all_weights[weight_layer_count][weight_perceptron_count][single_weight_count]
					
					if(weight_layer_count != len(self.all_weights)-1):
						weight_change_layer_to_sum = weight_layer_count + 1
						prev_step_back_prop_error_val  = 0
						for weight_change_perceptron_count in range(0, len(self.weight_change_record[weight_change_layer_to_sum])):
							
							previous_weight_change = self.weight_change_record[weight_change_layer_to_sum][weight_change_perceptron_count][weight_perceptron_count]
							prev_step_back_prop_error_val += previous_weight_change

					weight_input_perceptron_val = self.nn_perceptrons[weight_layer_count][single_weight_count]
					final_activated_to_weight = final_activated_to_sum_step * weight_input_perceptron_val
					full_step_back_val = prev_step_back_prop_error_val * final_activated_to_weight
					self.weight_change_record[weight_layer_count][weight_perceptron_count][single_weight_count] = full_step_back_val
					new_weight_val = current_weight_val - (self.learning_constant * full_step_back_val)
					self.all_weights[weight_layer_count][weight_perceptron_count][single_weight_count] = new_weight_val
		print(output_error_total)

	def learn_analyse_iteration(self):
		matrix_count = 0
		for matrix in self.character_matrices:
			target_val = self.character_targets[matrix_count]
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
			return 1/(1+(math.e**-value))


	def populate_target_vector(self,target):
		vector = []
		for i in range(0,self.output_count):
			vector.append(0)
		vector[target] = 1
		return vector


def main():

	matrix_data = matrix_data_handler()
	matrix_data.populate_character_matrices()
	
	#neural network options
	testing_mode = False

	if(testing_mode == True):
		input_perceptron_count = 4#matrix_data_handler.matrix_width * matrix_data_handler.matrix_width
		hidden_layers = [2,2] #[hidden layer length]
		output_perceptron_count = 3
		'''for i in range(matrix_data.characters_to_retreive):
			cv2.imshow("frame"+str(i)+"-"+str(matrix_data.character_targets[i]),matrix_data.character_matrices[i])'''

	else:
		input_perceptron_count = matrix_data_handler.matrix_width * matrix_data_handler.matrix_width
		hidden_layers = [15,15]
		output_perceptron_count = 10
	
	learning_constant = 1

	neural_network = neural_network_handler(hidden_layers,										
								input_perceptron_count,
								output_perceptron_count,
								matrix_data.character_matrices,
								matrix_data.character_targets,
								learning_constant,
								testing_mode)


	user_interface = user_interface_handler()
	user_interface.render_neural_net_visualization(neural_network.nn_perceptrons)

	neural_network.learn_analyse_iteration()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


main()
