from __future__ import print_function
import cv2,re,numpy as np,random,math,time,thread,decimal,tkMessageBox,tkSimpleDialog,matplotlib,os,json
from pprint import pprint
import FileDialog
from Tkinter import *
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from PIL import Image, ImageTk

class data_preprocessor_handler():
	
	def __init__(self, ui):
		self.user_interface = ui

	def normalise_text_file(self,text_file,target_val_pos,elements_to_ignore,divider):
		target_val_pos -= 1
		elements_to_ignore.append(target_val_pos)
		name_for_new = text_file[0:text_file.rfind(".")]
		new_txt_file = open("processed_datasets/"+name_for_new+"_new.txt", "a")
		data_by_row = open("original_datasets/"+text_file, 'r').read().split("\n")
		for row in data_by_row:
			row = row.split(",")
			new_row = []
			r_count = 0
			if(len(row)>1):
				for element in row:
					if(r_count not in elements_to_ignore):
						element = element.strip()
						empty_count = 0
						if(element.replace(".", "", 1).isdigit()):
							element = float(element)/divider
							new_row.append(element)
					r_count += 1
				new_row_count = len(new_row)
				new_row.append(row[target_val_pos])
				row_str = ','.join(str(e) for e in new_row)
				row_str += ","
				new_txt_file.write(row_str)
		self.user_interface.print_console("Finished processing. There are now " + str(new_row_count) + " inputs/values per row. Excluding the target.")

	def image_dir_to_matrix_txt(self, dirname):
		new_txt_file = open("processed_datasets/"+dirname+"_new.txt", "a")
		image_file_names = os.listdir(dirname)
		for image_file_name in image_file_names:
			if(image_file_name[0:1] != "."):
				pre_file_type_loc = image_file_name.rfind(".")
				image_name_data = image_file_name[0:pre_file_type_loc]
				target_val = image_name_data.split(",")[1]
				image_matrix = cv2.imread(dirname+"/"+image_file_name)
				image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
				c = 0
				new_txt_file.write(target_val+"")
				for row_px in range(0,len(image_matrix)):
					for col_px in range(0,len(image_matrix[0])):
						new_txt_file.write(str(image_matrix[row_px][col_px]) + ",")
						c+=1
	
class matrix_data_loader_handler:
	def __init__(self, matrix_dims, to_retrieve, file_name,user_interface):
		self.user_interface = user_interface
		self.matrix_width = matrix_dims[0]
		self.matrix_height = matrix_dims[1]
		self.input_total = self.matrix_width*self.matrix_height
		self.to_retrieve = to_retrieve
		self.file_name = file_name
		self.user_interface.print_console("Loading "+str(self.to_retrieve)+" items from " + self.file_name + "... \n")
		self.data_set = open("processed_datasets/"+file_name, 'r').read().replace("\n", ",",1).replace(" ", "",1).split(",")
		self.matrices = []
		self.targets = []
		self.input_divider_val = 1
		self.max_data_amount = int(len(self.data_set) / (self.input_total+1))

		if(self.to_retrieve == "all"):
			self.to_retrieve = self.max_data_amount

	def populate_matrices(self):
		px_count = 0
		done_msg = "Finished loading data \n "
		prev_pos_of_matrix = 0
		target_pos_in_row = -1
		for i in range(1,self.to_retrieve+1):
			if(self.user_interface.cancel_training == True):
				done_msg = "**CANCELLED** \n "
				break
			pos_of_matrix = (i*(self.input_total))+i
			flat_single_item = self.data_set[prev_pos_of_matrix:pos_of_matrix]
			if(len(flat_single_item)>0):
				target_val = int(flat_single_item[target_pos_in_row])
				del flat_single_item[target_pos_in_row]
				item_as_array = np.asarray(flat_single_item, dtype=np.float32) / self.input_divider_val
				array_as_matrix = np.reshape(item_as_array,(self.matrix_width,self.matrix_height),order="A")
				self.matrices.append(array_as_matrix)
				self.targets.append(target_val)
				prev_pos_of_matrix = pos_of_matrix
			if(self.to_retrieve > 10):
				if(i%(int(self.to_retrieve/5))==0):
					self.user_interface.print_console("Loaded "+str(i)+"/"+str(self.to_retrieve))
		self.user_interface.print_console(done_msg)

	def prep_matrix_for_input(self, matrix):
		matrix_float = matrix.astype(np.float32)
		matrix_for_input = matrix_float / float(self.input_divider_val)
		return matrix_for_input

class user_interface_handler:

	frame_height = 800
	frame_width = 1200

	def __init__(self, tk_main):
		self.tk_main = tk_main
		self.ui_frame = Frame(self.tk_main)
		self.ui_frame.pack()
		self.tk_main.title("Perceptron") 
		self.tk_main.minsize(width=self.frame_width, height=self.frame_height)
		self.tk_main.maxsize(width=self.frame_width, height=self.frame_height)
		self.font_face = "Helvetica"
		self.main_font_size = 13
		self.tk_main.protocol('WM_DELETE_WINDOW', self.tk_main.quit)
		self.canvas_height = 500
		self.canvas_width = 950
		self.cancel_training = False
		self.new_line_count = 0
		self.canvas_labels = []
		self.settings_file_name = "resources/settings.json"
		self.can_clear_graph = False
		self.render_ui_frames()
		self.render_ui_widgets()
		
		

	def render_ui_frames(self):

		self.learn_options_frame = Frame(self.ui_frame,width=500)
		self.learn_options_frame.pack(fill=BOTH,side=LEFT)
		#self.console_frame = Frame(self.ui_frame,bg="grey",height=300,width=400)
		#self.console_frame.pack()
		self.c_scrollbar = Scrollbar(self.tk_main)
		self.c_scrollbar.pack(side=RIGHT, fill=Y)

		self.lower_frame = Frame(self.ui_frame)
		self.lower_frame.pack(side=BOTTOM, fill=BOTH)
		self.console_list_box = Text(self.lower_frame,bg="grey",height=19,width=34,borderwidth=0, highlightthickness=0,font=("courier bold", 10))
		self.console_list_box.pack(padx=3,ipady=20,ipadx=10,side=LEFT,fill=Y)
		self.console_list_box.config(yscrollcommand=self.c_scrollbar.set)
		self.console_list_box.configure(state="disabled")
		self.c_scrollbar.config(command=self.console_list_box.yview)
		self.tk_nn_visual_canvas = Canvas(self.ui_frame, width=self.canvas_width, height=self.canvas_height,background="grey")
		self.tk_nn_visual_canvas.pack(side=RIGHT)

		self.g_figures = range(2)
		self.g_axis = range(2)
		self.g_lines = [[],[]]
		self.g_canvas = range(2)

		rcParams.update({'figure.autolayout': True})

		self.line_colors = ["blue","green","red","magneta","cyan","yellow"]
		self.render_graph("% Of Error (from 1000 feedforwards)","1000 forward feeds","%",0,"r")
		self.render_graph("% Of Success (from test data each Epoch)","Epoch","%",1,"b")
		self.prepare_new_line_graph()

	def render_graph(self,title,xlabel,ylabel,line_num,col):

		self.g_figures[line_num] =  plt.figure()
		self.g_axis[line_num] = self.g_figures[line_num].add_subplot(111)
		self.g_axis[line_num].set_ylabel(ylabel)
		self.g_axis[line_num].set_xlabel(xlabel)
		self.g_figures[line_num].suptitle(title)
		self.g_axis[line_num].get_yaxis().set_visible(False)
		self.g_axis[line_num].get_xaxis().set_visible(False)
		self.g_canvas[line_num] = FigureCanvasTkAgg(self.g_figures[line_num], master=self.lower_frame)
		self.g_canvas[line_num].get_tk_widget().config(width=360,height=280)
		self.g_canvas[line_num].get_tk_widget().pack(side=LEFT,fill=X)


	def render_canvas_info_labels(self):
		self.canvas_info_labels = {}
		self.canvas_info_label_vals = {}
		self.canvas_label_names = ["Latest Success", "Latest Error", "Epoch Duration"]
		label_y = 30
		for label_name in self.canvas_label_names:
			self.canvas_info_label_vals[label_name] = StringVar()
			self.canvas_info_label_vals[label_name].set(label_name+":")
			self.canvas_info_labels[label_name] = Label(self.tk_nn_visual_canvas, textvariable=self.canvas_info_label_vals[label_name],font=(self.font_face, self.main_font_size),bg="grey")
			self.canvas_info_labels[label_name].place(x=self.canvas_width-230, y=self.canvas_height-label_y)
			label_y += 20

	def update_canvas_info_label(self,label_name,val):
		self.canvas_info_label_vals[label_name].set(label_name+": "+str(val))


	prev_line_1_data = 0.0
	axis_g_showing = [False,False]
	all_g1_annotations = []

	def animate_graph_figures(self, line,data):
		if(self.axis_g_showing[line]==False):
			self.g_axis[line].get_yaxis().set_visible(True)
			self.g_axis[line].get_xaxis().set_visible(True)
			self.axis_g_showing[line]=True
			
		ydata = self.g_lines[line][-1].get_ydata()
		ydata = np.append(ydata,data)
		#print(ydata)
		self.g_lines[line][-1].set_ydata(ydata)
		self.g_lines[line][-1].set_xdata(range(len(ydata)))
		self.g_axis[line].relim()
		self.g_axis[line].autoscale_view()

		if(line==1): 
			if(data!=self.prev_line_1_data):
				self.all_g1_annotations.append(self.g_axis[line].annotate(str(data)+"%",(len(ydata)-1,data)))
				self.all_g1_annotations[-1].set_fontsize(9)
			self.prev_line_1_data = data

		self.g_figures[line].canvas.draw()

	def clear_graphs(self):
		if(self.can_clear_graph==True):
			for ann in range(len(self.all_g1_annotations)):
				self.all_g1_annotations[ann].remove()
			self.all_g1_annotations = []
			for i in range(2):
				for line in range(len(self.g_lines[i])):
					self.g_lines[i][line].remove()
				self.g_lines[i] = []
				self.g_figures[i].canvas.draw()
			self.new_line_count = 0
			self.can_clear_graph = False
			self.prepare_new_line_graph()

	def prepare_new_line_graph(self):
		for line in range(2):
			new_line, = self.g_axis[line].plot([], [], self.line_colors[self.new_line_count][0:1]+"-")
			self.g_lines[line].append(new_line)
		self.new_line_count += 1


	input_text_length = 8
	default_hidden_layers_str = "10,20"
	default_bias_str = "1,1"
	default_input_dims = "20"
	default_data_set_str = ".txt"
	default_output_count = "5"

	def render_nn_vis_trigger(self,event=None):
		if(event==None):
			hidden_str = self.default_hidden_layers_str
			bias_str = self.default_bias_str
			input_dims = self.default_input_dims
			output_count = int(self.default_output_count)
		else:
			hidden_str = self.input_fields["hidden_layer"].get()
			bias_str = self.input_fields["bias_vals"].get()
			input_dims = self.input_fields["matrix_dims"].get()
			output_count_str = self.input_fields["output_count"].get()
			if(output_count_str.isdigit()):
				output_count = int(output_count_str)
			else:
				output_count = -1

		if(self.check_str_list_valid(hidden_str+bias_str) == True and hidden_str != "" and bias_str != "" and input_dims != ""):
			if(hidden_str[-1]==","):
				hidden_str = hidden_str[0:-1]
			if(bias_str[-1]==","):
				bias_str = bias_str[0:-1]
			if(input_dims[-1]==","):
				input_dims = input_dims[0:-1]
			if(self.check_str_list_valid(input_dims) == True and output_count > 0):
				input_dims = input_dims.split(",")
				inputs_total = int(input_dims[0])
				if(len(input_dims)==2): inputs_total = inputs_total * int(input_dims[1])
				layers = [inputs_total]
				hidden_layers = hidden_str.split(",")
				layers.extend(hidden_layers)
				biases = bias_str.split(",")
				layers.append(output_count)
				layers = map(int,layers)
				biases = map(int,biases)
				if(len(layers) > 0 and len(biases) > 0):
					self.render_neural_net_visualization(layers,biases)

	def render_ui_widgets(self):
		self.render_nn_vis_trigger()
		
		icon = ImageTk.PhotoImage(Image.open("resources/perceptron-header.jpg").resize((230, 100), Image.ANTIALIAS))
		self.icon_view = Label(self.learn_options_frame,image=icon)
		self.icon_view.image = icon
		self.icon_view.pack()
	
		self.choose_settings_frame = Frame(self.learn_options_frame)
		self.choose_settings_frame.pack()
		self.render_settings_opts()

		self.render_canvas_info_labels()

		self.input_fields = {}
		self.clear_graphs_opt = self.render_option("PREPROCCESS DATA", self.preproccess_data_render, self.learn_options_frame)

		self.input_fields["dataset_name"] = self.render_input_field(self.default_data_set_str, "Dataset file","Save a text file into the current directory and enter name here",self.input_text_length,self.learn_options_frame)
		self.input_fields["data_to_retrieve"]= self.render_input_field("all", "Data To Use","Enter 'all' or number",self.input_text_length,self.learn_options_frame)
		self.input_fields["matrix_dims"] = self.render_input_field(self.default_input_dims,"Matrix Input Dimensions","Enter height, width of matrix",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["output_count"] = self.render_input_field(self.default_output_count, "Output Count","Enter output quantity",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["hidden_layer"] = self.render_input_field(self.default_hidden_layers_str, "Hidden Layers", "Enter comma seperated list of layer sizes",self.input_text_length, self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["bias_vals"] = self.render_input_field(self.default_bias_str, "Bias Values", "List must match hidden layer count plus output, but enter 0 for no bias",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["learning_rate"] = self.render_input_field("0.5", "Learning Rate","Enter decimal or integer",self.input_text_length,self.learn_options_frame)
		self.input_fields["weight_range"] = self.render_input_field("-1,1", "Weight Ranges","Enter one value (or two for a range) for initial weight values",self.input_text_length, self.learn_options_frame)
		self.input_fields["epochs"] = self.render_input_field("10", "Epochs","Total number of iterations",self.input_text_length, self.learn_options_frame)
		self.input_fields["test_data_partition"] = self.render_input_field("2000", "Data for Testing","Amount of data to partition from dataset for result testing",self.input_text_length, self.learn_options_frame)
		
		self.start_learning_opt = self.render_option("START LEARNING", self.start_learning_ui_request, self.learn_options_frame)
		self.cancel_learning_opt = self.render_option("STOP", self.cancel_learning, self.learn_options_frame)
		self.cancel_learning_opt.config(state="disabled")
		self.clear_graphs_opt = self.render_option("Clear Graphs", self.clear_graphs, self.learn_options_frame)
		self.save_settings_opt = self.render_option("Save Settings",self.save_settings,self.learn_options_frame)
		self.input_fields["test_input_val"] = self.render_input_field("", "Test Input","Enter the name of an image file, text file, or enter data manually",self.input_text_length, self.learn_options_frame)
		self.test_input_opt = self.render_option("Test", self.test_input, self.learn_options_frame)
		self.test_input_opt.config(state="disabled")

	def render_input_field(self,default_value, label_text,desc_text,width,parent_frame,command=None):
			widget_frame = Frame(parent_frame)
			widget_frame.pack(fill=X,expand=False)
			#desc_frame = Frame(widget_frame, width=50)
			#desc_frame.pack(side=BOTTOM,expand=False)
			text_input = Entry(widget_frame, width=width)
			if(command!=None):
				text_input.bind("<KeyRelease>", command)

			text_input.insert(0,str(default_value))
			text_input.pack(side=RIGHT)
			input_label = Label(widget_frame, text=label_text+": ",font=(self.font_face, self.main_font_size))
			input_label.pack(side=LEFT)
			#label_desc = Label(desc_frame, text="*"+desc_text, font=(self.font_face, 10), fg="#60606b",wraplength=210)
			#label_desc.pack(side=BOTTOM)
			return text_input

	def load_settings(self,value):
		setting_to_load = self.saved_settings_text.get()
		if(setting_to_load != self.saved_settings_dis_text):
			settings_file_json = json.loads(open(self.settings_file_name, "r").read())
			spec_settings = settings_file_json[setting_to_load]
			for input_field in self.input_fields:
				self.input_fields[input_field].delete(0,END)
				self.input_fields[input_field].insert(0,spec_settings[input_field])
			self.render_nn_vis_trigger(True)

	def render_settings_opts(self):
		self.saved_settings_dis_text = "--Saved Settings--"
		settings_str = open(self.settings_file_name, "r").read()
		saved_settings = [self.saved_settings_dis_text]
		if(len(settings_str)>0):
			settings_file_json = json.loads(settings_str)
			for setting in settings_file_json:
				saved_settings.append(setting)
		self.saved_settings_text = StringVar(self.tk_main)
		self.saved_settings_opts = OptionMenu(self.choose_settings_frame, self.saved_settings_text,command=self.load_settings,*saved_settings)
		self.saved_settings_opts.config(width=15)
		self.saved_settings_opts.pack()
		self.saved_settings_text.set(saved_settings[0])

	def save_settings(self):
		settings_name = tkSimpleDialog.askstring("Perceptron", "Setting's Name: ")
		if(settings_name != None):
			if(len(settings_name)>1):
				settings_file_read = open(self.settings_file_name, "r")
				settings_str = settings_file_read.read()
				if(len(settings_str) == 0):
					settings_str = "{}"
				all_settings_as_json = json.loads(settings_str)
				input_values = {}
				for input_field in self.input_fields:
					input_values[input_field] = self.input_fields[input_field].get()
				all_settings_as_json[settings_name] = input_values
				#print(all_settings_as_json)
				all_settings_as_str = json.dumps(all_settings_as_json)
				settings_file_write = open(self.settings_file_name, "w")
				settings_file_write.write(all_settings_as_str)
				settings_file_read.close()	
				settings_file_write.close()
				self.saved_settings_opts.destroy()
				self.render_settings_opts()
	
	def render_option(self,text, command,parent_frame,side=None,anchor=None):
			option = Button(parent_frame, text=text, command=command)
			option.pack(side=side,anchor=anchor)
			return option

	def check_str_list_valid(self,string):
		valid_str_entry = True
		for char in string:
			if(char!="," and char.isdigit()==False):
				valid_str_entry = False
				break

		return valid_str_entry

	prev_guess = -1
	def render_camera(self):
		camera_window = Toplevel(self.tk_main)
		image_frame = Frame(camera_window, width=600, height=500)
		image_frame.pack()
		capture_frame = cv2.VideoCapture(0)
		label_for_cam = Label(image_frame)
		label_for_cam.pack()

		mini_cam_window = Toplevel(self.tk_main,width=300,height=300)
		imagemini_frame = Frame(mini_cam_window, width=600, height=500)
		imagemini_frame.pack()
		label_for_minicam = Label(imagemini_frame)
		label_for_minicam.pack()

		def render_cam_frame():
			_, cv_frame = capture_frame.read()
			cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
			roi_size = 50
			roi_point_1 = (400,200)
			roi_point_2 = (roi_point_1[0]+roi_size,roi_point_1[1]+roi_size)
			roi_matrix = cv_frame[roi_point_1[1]:roi_point_2[1],roi_point_1[0]:roi_point_2[0]]
			_,roi_matrix = cv2.threshold(roi_matrix,100,255,cv2.THRESH_BINARY_INV)
			
			img_miniframe = Image.fromarray(roi_matrix)
			tk_miniframe = ImageTk.PhotoImage(image=img_miniframe)
			label_for_minicam.imgtk = tk_miniframe
			label_for_minicam.configure(image=tk_miniframe)

			roi_matrix = cv2.resize(roi_matrix, (self.matrix_dims[0],self.matrix_dims[1]))
			matrix_float = self.matrix_data_loader.prep_matrix_for_input(roi_matrix)
			outline_vals = [matrix_float[0,:-1], matrix_float[:-1,-1], matrix_float[-1,::-1], matrix_float[-2:0:-1,0]]
			outline_sum = np.concatenate(outline_vals).sum()
			if(int(outline_sum) == 0):
				self.neural_network.feed_forward(matrix_float)
				output_neurons = self.neural_network.nn_neurons[-1].tolist()
				max_val = max(output_neurons)
				if(max_val > 0.9):
					guess_val = output_neurons.index(max_val)
					print(guess_val)

			cv2.rectangle(cv_frame,roi_point_1,roi_point_2, (255), thickness=3, lineType=8, shift=0)

			img_frame = Image.fromarray(cv_frame)
			tk_frame = ImageTk.PhotoImage(image=img_frame)
			label_for_cam.imgtk = tk_frame
			label_for_cam.configure(image=tk_frame)
			label_for_cam.after(10, render_cam_frame) 

		render_cam_frame()

	def preproccess_data_render(self):
		self.preproccess_window = Toplevel(self.ui_frame,width=300,height=400)
		self.prepro_original_file = self.render_input_field(".txt", "Dataset File","Enter the name of the text file",self.input_text_length, self.preproccess_window)
		self.prepro_target_val_pos = self.render_input_field("1", "Target Value Position","Enter position of target value in row",self.input_text_length, self.preproccess_window)
		self.prepro_values_to_ignore = self.render_input_field(".txt", "Values to Ignore","Enter positions of values to be removed/ignored e ",self.input_text_length, self.preproccess_window)
		self.prepro_divider = self.render_input_field("10", "Minimise Data By","Divide all values by this value",self.input_text_length, self.preproccess_window)
		self.prepro_opt = self.render_option("PROCCESS", self.start_preprocess, self.preproccess_window)

	def start_preprocess(self):
		preproccessor_handler = data_preprocessor_handler(self)

		file_name = self.prepro_original_file.get()
		target_pos = self.prepro_target_val_pos.get()
		values_to_ignore = self.prepro_values_to_ignore.get()
		divider = self.prepro_divider.get()
		if(self.check_str_list_valid(values_to_ignore)==False):
			error = "Invalid values to ignore"
		elif values_to_ignore == "":
			values_to_ignore = []
		else:
			values_to_ignore = map(int,values_to_ignore.split(","))
		if(target_pos.isdigit()==False):
			error = "Invalid target position"
		else:
			target_pos = int(target_pos)
		if(divider.isdigit()==False):
			error = "Invalid target position"
		else:
			divider = int(divider)

		preproccessor_handler.normalise_text_file(file_name,target_pos,values_to_ignore,divider)

	def test_input(self):
		input_str = self.input_fields["test_input_val"].get() 
		file_type_pos = input_str.rfind(".")
		valid_files = ["png","jpg","txt"]
		file_type_str = ""

		if(input_str == "camera"):
			self.render_camera()
		else:
			if(file_type_pos != -1):
				file_type_str = input_str[file_type_pos+1:]
			
			if(file_type_str not in valid_files or file_type_str == "txt"):
				if(file_type_str == "txt"):
					input_str = open(input_str, 'r').read()
				input_data = input_str.split(",")
				item_as_array = self.matrix_data_loader.prep_matrix_for_input(np.asarray(input_data))
				matrix_ready = np.reshape(item_as_array,(self.matrix_dims[0],self.matrix_dims[1]),order="A")	
				
			elif(file_type_str in valid_files):
				image_matrix = cv2.imread(file_name)
				image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
				image_matrix = cv2.resize(image_matrix, (self.matrix_dims[0],self.matrix_dims[1]))
				matrix_ready = self.matrix_data_loader.prep_matrix_for_input(image_matrix)
			else:
				output_pos_result = -1
				self.print_console("**ERROR: invalid test input")

			self.neural_network.feed_forward(matrix_ready)
			output_neurons = self.neural_network.nn_neurons[-1].tolist()
			if(len(output_neurons)>1):
				output_pos_result = output_neurons.index(max(output_neurons))
			else:
				output_pos_result = output_neurons[0]


			if(output_pos_result != -1):
				self.print_console("**OUTPUT RESULT: " + str(output_pos_result))
	
	def cancel_learning(self):
		self.cancel_training = True
		self.prepare_new_line_graph()
		self.start_learning_opt.config(state="normal")
		self.cancel_learning_opt.config(state="disabled")
		if(self.input_neuron_count>0):
			self.test_input_opt.config(state="normal")

	def print_console(self,text):
		self.console_list_box.configure(state="normal")
		if(text==" **TRAINING** \n"):
			text += ">>With graph line color: "+self.line_colors[self.new_line_count-1]
		self.console_list_box.insert(END,">>" + text + "\n")
		self.console_list_box.see(END)
		self.console_list_box.configure(state="disabled")

	def check_all_fields_valid(self):
		hidden_str = self.input_fields["hidden_layer"].get()
		bias_str = self.input_fields["bias_vals"].get()
		error = ""
		valid_values = {}
		if(self.check_str_list_valid(hidden_str+bias_str) == False or hidden_str == "" or bias_str == ""):
			error = "You hidden layers or bias values are invalid"
		else:
			valid_values['hidden_layers'] = map(int,hidden_str.split(","))
			valid_values['biases_for_non_input_layers'] = map(int,bias_str.split(","))

			if(len(valid_values['hidden_layers'])+1 != len(valid_values['biases_for_non_input_layers'])):
				error = "Bias count must be equal to "+str(len(valid_values['hidden_layers'])+1)+" (the total layer count expect input)"
		
		learning_constant = self.input_fields["learning_rate"].get()
		if(learning_constant.replace(".", "", 1).isdigit() == False):
			error = "Invalid learning constant"
		else:
			valid_values['learning_constant'] = float(learning_constant)
		valid_values['data_file_name'] = self.input_fields["dataset_name"].get()
		matrix_dims_str = self.input_fields["matrix_dims"].get()
		weight_range_str = self.input_fields["weight_range"].get()
		to_retrieve = self.input_fields["data_to_retrieve"].get()
		output_count = self.input_fields["output_count"].get()
		epochs = self.input_fields["epochs"].get()
		data_to_test_count = self.input_fields["test_data_partition"].get()

		if(self.check_str_list_valid(matrix_dims_str)==False):
			error = "Invalid matrix dimensions"
		else:
			valid_values['matrix_dims'] = map(int,matrix_dims_str.split(","))

		weight_range_str_test = weight_range_str.replace(".", "")
		weight_range_str_test = weight_range_str_test.replace("-", "")
		if(self.check_str_list_valid(weight_range_str_test)==False):
			error = "Invalid weight ranges"
		else:
			valid_values['weight_range'] = map(float,weight_range_str.split(","))
		if(to_retrieve.isdigit() == False and to_retrieve != 'all'):
			error = "Invalid matrices to use entry"

		else:
			if(to_retrieve!='all'):
				valid_values['to_retrieve'] = int(to_retrieve)
			else:
				valid_values['to_retrieve'] = to_retrieve 
		if(output_count.isdigit() == False):
			error = "Invalid output count"
		else:
			valid_values['output_count'] = int(output_count)
		if(epochs.isdigit() == False):
			error = "Invalid epochs entry"
		else:
			valid_values['epochs'] = int(epochs)
		if(data_to_test_count.isdigit() == False):
			error = "Invalid data to test entry"
		else:
			valid_values['data_to_test'] = int(data_to_test_count)
			if(valid_values['to_retrieve'] != 'all'):
				if(valid_values['data_to_test'] > valid_values['to_retrieve']):
					error = "Data to test amount cannot be more than all data to get"

		valid_values['success'] = True

		if(error == ""):
			return valid_values
		else:
			response = {}
			response['success'] = False
			response['error'] = error
			return response

	def start_learning_ui_request(self):
		self.cancel_training = False
		self.can_clear_graph = True
		self.field_result = self.check_all_fields_valid()
		if(self.field_result['success'] ==True):
			self.start_learning_opt.config(state="disabled")
			self.cancel_learning_opt.config(state="normal")
			thread.start_new_thread(self.start_learning_in_thread,())
		else:
			tkMessageBox.showinfo("Error", self.field_result['error'])

	matrix_data = []
	matrix_targets = []
	curr_dataset_name = ""
	input_neuron_count = 0
	prev_to_retrieve_val = ""
	def start_learning_in_thread(self):
		field_result = self.field_result
		testing_mode = False
		if(field_result['to_retrieve']!=self.prev_to_retrieve_val or field_result['data_file_name'] != self.curr_dataset_name):
			if(len(field_result['matrix_dims'])==1):
				field_result['matrix_dims'].append(field_result['matrix_dims'][0])
				field_result['matrix_dims'][0] = 1

			self.curr_dataset_name = field_result['data_file_name']
			self.matrix_dims = field_result['matrix_dims']
			self.matrix_data_loader = matrix_data_loader_handler(field_result['matrix_dims'],field_result['to_retrieve'],field_result['data_file_name'],self)
			self.matrix_data_loader.populate_matrices()
			self.prev_to_retrieve = self.matrix_data_loader.to_retrieve
			self.input_neuron_count = self.matrix_data_loader.matrix_width * self.matrix_data_loader.matrix_height
			self.matrix_data = self.matrix_data_loader.matrices
			self.matrix_targets = self.matrix_data_loader.targets
		
		self.neural_network = neural_network_handler()
		self.neural_network.initilize_nn(field_result['hidden_layers'],
				self.input_neuron_count,field_result['output_count'], self.matrix_data,self.matrix_targets,
				field_result['biases_for_non_input_layers'], field_result['learning_constant'], 
				testing_mode,field_result['weight_range'],field_result['epochs'],field_result['data_to_test'],
				self)
		self.prev_to_retrieve_val = field_result['to_retrieve']
		self.neural_network.train()

	def render_neural_net_visualization(self,layers,biases):
		self.tk_nn_visual_canvas.delete("all")
		for old_labels in self.canvas_labels:
			old_labels.destroy()

		example_p_limit_count = 20 #zero for all
		highest_layer_count = max(layers)
		if(highest_layer_count > example_p_limit_count):
			highest_layer_count = example_p_limit_count

		highest_layer_height = 0

		if(len(layers)-1 != len(biases)):
			diff_b_layers = len(layers)-1 - len(biases)
			if(diff_b_layers < 0):
				biases = biases[0:diff_b_layers]
			else:
				for i in range(diff_b_layers):
					biases.append(0)

		neuron_padding = 5		
		neuron_radius = int((((self.canvas_height / highest_layer_count)/2)-neuron_padding))
		if(neuron_radius > 15): neuron_radius = 15
		neuron_x = neuron_radius + 20
		neuron_dist_x = (self.canvas_width / (len(layers)-1)) - neuron_x*2
		neuron_hidden_c = "blue"
		neuron_outter_c = "red"

		bias_pos_diff_x = 50
		bias_pos_diff_y = 50
		bias_color = "green"
		bias_pos_y = neuron_radius*2

		def get_layer_height_px(layer_count):
			return (layer_count*(neuron_radius*2 + neuron_padding))

		for neuron_layer in range(0,len(layers)):
			length_of_layer = layers[neuron_layer]
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count
			curr_layer_height = get_layer_height_px(length_of_layer)
			if(curr_layer_height > highest_layer_height):
				highest_layer_height = curr_layer_height

		for neuron_layer in range(0,len(layers)):
			length_of_layer = layers[neuron_layer]
			if(example_p_limit_count > 0 and example_p_limit_count < length_of_layer):
				length_of_layer = example_p_limit_count

			neuron_ystart = (self.canvas_height - get_layer_height_px(length_of_layer))/2
			neuron_y = neuron_ystart
			layer_has_bias = ((neuron_layer > 0) and (biases[neuron_layer-1] != 0))
			if layer_has_bias == True:
				bias_y_pos = 20
				bias_x_pos = neuron_x-bias_pos_diff_x
				bias_oval = self.tk_nn_visual_canvas.create_oval(bias_x_pos-neuron_radius,bias_y_pos-neuron_radius,bias_x_pos+neuron_radius,bias_y_pos+neuron_radius, fill=bias_color,outline=bias_color)
				self.tk_nn_visual_canvas.tag_raise(bias_oval)

			neuron_color = neuron_hidden_c
			if(neuron_layer==0 or neuron_layer == len(layers)-1):
				neuron_color = neuron_outter_c
			for single_neuron in range(0,length_of_layer):
				if(single_neuron == 0):
					real_layer_count = layers[neuron_layer]
					extra_str_label = ""
					if(real_layer_count > length_of_layer):
						extra_str_label = "^\n^\n"
					self.canvas_labels.append(Label(self.tk_nn_visual_canvas, text=extra_str_label+str(real_layer_count)))
					self.canvas_labels[-1].place(x=neuron_x-(neuron_radius*2), y=neuron_y-(neuron_radius*3))

				neuron_oval = self.tk_nn_visual_canvas.create_oval(neuron_x-neuron_radius,neuron_y-neuron_radius,neuron_x+neuron_radius,neuron_y+neuron_radius,fill=neuron_color,outline=neuron_color)
				self.tk_nn_visual_canvas.tag_raise(neuron_oval)

				if(layer_has_bias == True):
					bias_connector = self.tk_nn_visual_canvas.create_line(neuron_x, neuron_y, bias_x_pos,bias_y_pos)
					self.tk_nn_visual_canvas.tag_lower(bias_connector)

				neuron_dist_y = (neuron_radius*2) + neuron_padding
				if(neuron_layer < len(layers)-1):
					length_of_next_layer = layers[neuron_layer+1]
					if(example_p_limit_count > 0 and example_p_limit_count < length_of_next_layer):
						length_of_next_layer = example_p_limit_count
					neuron_y_for_line = (self.canvas_height - (length_of_next_layer)*(neuron_radius*2 + neuron_padding))/2
					
					for neuron_weights in range(0,length_of_next_layer):
						neuron_connector = self.tk_nn_visual_canvas.create_line(neuron_x, neuron_y, neuron_x+neuron_dist_x, neuron_y_for_line)
						self.tk_nn_visual_canvas.tag_lower(neuron_connector)

						neuron_y_for_line += neuron_dist_y

				neuron_y += neuron_dist_y
			neuron_x += neuron_dist_x
		
class neural_network_handler:

	#construct object to develop specific network structure
	def initilize_nn(self, hidden_layers,
					input_count, output_count, matrix_data,matrix_targets,
					biases_for_non_input_layers, learning_constant,
					testing_mode,weight_range,epochs,data_to_test,user_interface):

		self.user_interface = user_interface
		if(self.user_interface.cancel_training == False):
			self.user_interface.print_console("\n\n\n--------------------------- \n Constructing neural network \n\n")
			self.all_weights = []
			self.nn_neurons = []
			self.weight_changes = []
			self.biases_weights = []
			self.biases_weight_changes = []
			self.epochs = epochs
			self.test_data_amount = data_to_test
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
				#print("hidden_to_add: ",hidden_neuron_sums, "sh: ",hidden_neuron_sums.shape)
				bias_vals = (self.biases_for_non_input_layers[after_input_layer-1] * self.biases_weights[after_input_layer-1])
				#print("bias_vals_adding: ", bias_vals, "sh: ",bias_vals.shape)
				hidden_neuron_sums += bias_vals
			self.nn_neurons[after_input_layer] = self.activate_threshold(hidden_neuron_sums, "sigmoid")

	def populate_input_layer(self, data):
		value_count = 0
		if(type(data[0]) is not int):
			self.nn_neurons[0] = data.flatten()
		else:
			self.nn_neurons[0] = np.array(data)

	testing_output_mode = False
	test_counter = 0
	correct_count = 0
	error_by_1000 = 0
	error_by_1000_counter = 1
	output_error_total = 0
	
	def back_propagate(self, target_val,repeat_count):

		if(len(self.nn_neurons[-1])>1 and type(target_val) is int):
			target_vector = self.populate_target_vector(target_val)
		else:
			target_vector = target_val
		if(len(self.nn_neurons[-1])>1):
			outputs_as_list = self.nn_neurons[-1].tolist()
			success_condition = (outputs_as_list.index(max(outputs_as_list))==target_val)
		else:
			success_condition = (round(self.nn_neurons[-1][0]) == target_vector)
		if(self.test_counter >= len(self.matrix_data)-self.test_data_amount):
			if(success_condition == True):
				self.correct_count += 1
		if(success_condition == False):
				self.error_by_1000 += 1
		if(self.error_by_1000_counter % 1000 == 0):
			self.user_interface.animate_graph_figures(0,self.error_by_1000/10)
			self.user_interface.update_canvas_info_label("Latest Error",str(self.error_by_1000/10)+"% (ep. "+str(repeat_count)+")")
			self.error_by_1000 = 0
			self.error_by_1000_counter = 0
		for weight_layer_count in range(len(self.all_weights)-1,-1,-1):
			weight_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count+1],axis=1)
			target_vector = np.expand_dims(target_vector,axis=1)
			activated_to_sum_step = weight_neuron_vals * (1-weight_neuron_vals)
			if(weight_layer_count == len(self.all_weights)-1):
				back_prop_cost_to_sum = (weight_neuron_vals - target_vector) * activated_to_sum_step
			else:
				back_prop_cost_to_sum = np.dot(np.asarray(self.all_weights[weight_layer_count+1]).transpose(),back_prop_cost_to_sum) * activated_to_sum_step

			if(len(self.biases_weights[weight_layer_count])!=0):
				current_bias_weight_vals = self.biases_weights[weight_layer_count]
				#print("bpc_bias: ", current_bias_weight_vals.shape)
				#print("bpc_bias: ", back_prop_cost_to_sum.shape)
				self.biases_weights[weight_layer_count] = current_bias_weight_vals - (self.learning_constant * back_prop_cost_to_sum.flatten())
				#print("bp_newb: ", self.biases_weights[weight_layer_count].shape)
			input_neuron_vals = np.expand_dims(self.nn_neurons[weight_layer_count],axis=1)
			full_back_prop_sum_to_input = np.dot(back_prop_cost_to_sum,input_neuron_vals.transpose())
			current_weight_vals = self.all_weights[weight_layer_count]
			new_weight_vals = current_weight_vals - (self.learning_constant * full_back_prop_sum_to_input)
			self.all_weights[weight_layer_count] = new_weight_vals
	
		self.test_counter += 1
		self.error_by_1000_counter += 1

	def save_network_as_json():
		weight_layers = self.all_weights.tolist()

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
					target_val = self.matrix_targets[matrix_count]
					self.feed_forward(matrix)
					self.back_propagate(target_val,epoch)
					matrix_count += 1
				if(self.user_interface.cancel_training == True):
					break

				success_p = (float(self.correct_count)/float(self.test_data_amount))*100
				self.user_interface.animate_graph_figures(1,success_p)
				e_note_str = " (ep. "+str(epoch)+")"

				if(self.is_small_data == False):
					self.user_interface.update_canvas_info_label("Latest Success",str(success_p)+"%"+e_note_str)
				success_list.append(success_p)
				self.test_counter = 0
				self.correct_count = 0
				post_epoch_time = time.time() - pre_epoch_time
				if(self.is_small_data == False):
					self.user_interface.update_canvas_info_label("Epoch Duration",str(round(post_epoch_time,3))+e_note_str)
				epoch_times.append(post_epoch_time)

			if(len(success_list)>0):
				av_success = sum(success_list)/len(success_list)
				highest_success = max(success_list)
				av_epoch_time = sum(epoch_times)/len(epoch_times)
			else:
				av_success = "N/A"
				highest_success = "N/A"
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

	def populate_target_vector(self,target):
		vector = []
		for i in range(0,self.output_count):
			vector.append(0)
		vector[target] = 1
		return vector

def main():
	tk_main = Tk()
	user_interface = user_interface_handler(tk_main)
	tk_main.mainloop()
	
main()
