from __future__ import print_function
import cv2,re,numpy as np,random,math,time,thread,decimal,tkMessageBox,tkSimpleDialog

class neural_network:

	#construct object to develop specific network structure
	def initilize_nn(self, hidden_layers,
					input_count, output_count, matrix_data,matrix_targets,
					biases_for_non_input_layers, learning_constant,
					testing_mode,weight_range,epochs,data_to_test,t_type,data_total,user_interface):

		self.user_interface = user_interface
		if(self.user_interface.cancel_training == False):
			self.user_interface.print_console("\n\n\n--------------------------- \n Constructing neural network \n\n")
			self.all_weights = []
			self.nn_neurons = []
			self.weight_changes = []
			self.biases_weights = []
			self.biases_weight_changes = []
			self.epochs = epochs
			divider_to_test = float(data_to_test)/100.0
			self.test_data_amount = int(round(divider_to_test*data_total))
			self.t_type = t_type
			self.matrix_data = matrix_data
			self.hidden_layers = hidden_layers
			self.matrix_targets = matrix_targets
			self.learning_constant = learning_constant
			self.output_count = output_count
			self.input_count = input_count
			self.testing_mode = testing_mode
			self.biases_for_non_input_layers = biases_for_non_input_layers
			self.weight_range = weight_range 
			self.success_records = []
			if(len(self.matrix_targets)<100):
				self.is_small_data = True
			else:
				self.is_small_data = False

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
		if(len(self.weight_range)==1):
			upper_bound = self.weight_range[0]
			lower_bound = upper_bound
		else:
			upper_bound = self.weight_range[1]
			lower_bound = self.weight_range[0]

		return np.random.uniform(low=lower_bound, high=upper_bound, size=(size))

	def feed_forward(self, matrix):
		self.populate_input_layer(matrix)
		for after_input_layer in range(1, len(self.nn_neurons)):
			hidden_neuron_sums = np.dot(np.asarray(self.all_weights[after_input_layer-1]) , self.nn_neurons[after_input_layer-1])
			if(len(self.biases_weights[after_input_layer-1])!=0):
				bias_vals = (self.biases_for_non_input_layers[after_input_layer-1] * self.biases_weights[after_input_layer-1])
				hidden_neuron_sums += bias_vals
			self.nn_neurons[after_input_layer] = self.activate_threshold(hidden_neuron_sums, "sigmoid")
	def populate_input_layer(self, data):
		self.nn_neurons[0] = np.array(data)

	testing_output_mode = False
	test_counter = 0
	correct_count = 0
	error_by_1000 = 0
	error_by_1000_counter = 1
	output_error_total = 0
	
	def back_propagate(self, target_val,repeat_count,t_type):
		if(t_type[0]=="B"):
			target_vector = self.user_interface.data_processor.populate_target_vector(target_val[0],self.output_count)
		else:
			target_vector = target_val

		if(len(self.nn_neurons[-1])>1):
			outputs_as_list = self.nn_neurons[-1].tolist()
			success_condition = (outputs_as_list.index(max(outputs_as_list))==int(target_val[0]))
		else:

			success_condition = (round(self.nn_neurons[-1][0]) == target_vector)

		if(self.test_counter >= len(self.matrix_data)-self.test_data_amount):
			if(success_condition == True):
				self.correct_count += 1
		if(success_condition == False):
				self.error_by_1000 += 1
		if(self.error_by_1000_counter % 1000 == 0):
			self.user_interface.animate_graph_figures(0,self.error_by_1000/10)
			self.error_by_1000 = 0
			self.error_by_1000_counter = 0
		for weight_layer_count in range(len(self.all_weights)-1,-1,-1):
			weight_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count+1],axis=1)
			target_vector = np.expand_dims(target_vector,axis=1)
			activated_to_sum_step = weight_neuron_vals * (1-weight_neuron_vals)
			if(weight_layer_count == len(self.all_weights)-1):
				back_prop_cost_to_sum = (weight_neuron_vals - target_vector) * activated_to_sum_step
			else:
				trans_prev_weights = np.asarray(self.all_weights[weight_layer_count+1]).transpose()
				back_prop_cost_to_sum = np.dot(trans_prev_weights,back_prop_cost_to_sum) * activated_to_sum_step

			if(len(self.biases_weights[weight_layer_count])!=0):
				current_bias_weight_vals = self.biases_weights[weight_layer_count]
				final_bias_change = self.learning_constant * back_prop_cost_to_sum.flatten()
				self.biases_weights[weight_layer_count] = current_bias_weight_vals - final_bias_change
			input_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count],axis=1)
			full_back_prop_sum_to_input = np.dot(back_prop_cost_to_sum,input_neuron_vals.transpose())
			current_weight_vals = self.all_weights[weight_layer_count]
			new_weight_vals = current_weight_vals - (self.learning_constant * full_back_prop_sum_to_input)
			self.all_weights[weight_layer_count] = new_weight_vals
	
		self.test_counter += 1
		self.error_by_1000_counter += 1

	def train(self):
		if(self.user_interface.cancel_training == False):
			success_list = []
			hidden_layer_str = ""
			for layerc in self.hidden_layers:
				hidden_layer_str += str(layerc)+","
			hidden_layer_str = hidden_layer_str[0:-1]
			cancel_training = False
			self.user_interface.print_console(" **TRAINING** \n")
			self.user_interface.print_console("With learning rate: " + str(self.learning_constant))
			self.user_interface.print_console("With hidden layers: " + str(hidden_layer_str))
			self.user_interface.print_console("With test amount by epoch size: " + str(self.test_data_amount)+"/"+str(len(self.matrix_targets)))
			self.user_interface.print_console("With epoch count: " + str(self.epochs))

			if(self.testing_mode == True):
				self.repeat_count = 5000

			epoch_times = []
			for epoch in range(1,self.epochs+1):
				pre_epoch_time = time.time()
				matrix_count = 0
				for matrix in self.matrix_data:
					if(self.user_interface.cancel_training == True):
						break
					target_vals = self.matrix_targets[matrix_count]
					self.feed_forward(matrix)
					self.back_propagate(target_vals,epoch,self.t_type)
					matrix_count += 1
				if(self.user_interface.cancel_training == True):
					break

				success_p = (float(self.correct_count)/float(self.test_data_amount))*100
				self.user_interface.animate_graph_figures(1,success_p)
				e_note_str = " (ep. "+str(epoch)+")"

				if(self.is_small_data == False):
					self.user_interface.update_canvas_info_label("Latest Success",str(round(success_p,2))+"%"+e_note_str)
				success_list.append(success_p)
				self.test_counter = 0
				self.correct_count = 0
				post_epoch_time = time.time() - pre_epoch_time
				if(self.is_small_data == False):
					self.user_interface.update_canvas_info_label("Epoch Duration",str(round(post_epoch_time,2))+"s "+e_note_str)
				epoch_times.append(post_epoch_time)

			if(len(success_list)>0):
				av_success = sum(success_list)/len(success_list)
				highest_success = max(success_list)
				av_epoch_time = sum(epoch_times)/len(epoch_times)
			else:
				av_success = "N/A"
				highest_success = "N/A"
				av_epoch_time = "N/A"
			training_done_msg = "**FINISHED**"
			if(self.user_interface.cancel_training == True):
				training_done_msg = "**CANCELLED**"
			else:
				self.user_interface.cancel_learning()
			self.user_interface.print_console(training_done_msg)
			self.user_interface.print_console("AVERAGE SUCCESS: " + str(av_success) + "%")
			self.user_interface.print_console("HIGHEST SUCCESS: " + str(highest_success) + "%")
			self.user_interface.print_console("TOTAL TIME: " + str(sum(epoch_times,5)) + "s")
			self.user_interface.print_console("AVERAGE EPOCH TIME: " + str(round(av_epoch_time,5)) + "s")

	def activate_threshold(self,value, type):
		if(type == "step"):
			if(value>=0.5):
				return 1
			else:
				return 0
		elif(type == "sigmoid"):
			return 1/(1 + np.exp(-value))




