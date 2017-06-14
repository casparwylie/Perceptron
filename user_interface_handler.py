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
from neural_network_handler import neural_network
from data_handler import data_processor

class user_interface:

	frame_height = 800
	frame_width = 1200

	def __init__(self, tk_main):
		self.tk_main = tk_main
		self.main_bg="#81899f"
		self.ui_frame = Frame(self.tk_main)
		self.ui_frame.configure(background=self.main_bg)
		self.ui_frame.pack()
		self.tk_main.title("Perceptron") 
		self.tk_main.configure(background=self.main_bg)
		self.tk_main.minsize(width=self.frame_width, height=self.frame_height)
		self.tk_main.maxsize(width=self.frame_width, height=self.frame_height)
		self.font_face = "Arial"
		self.main_font_size = 13
		self.tk_main.protocol('WM_DELETE_WINDOW', self.quit_all)
		self.canvas_height = 500
		self.canvas_width = 950
		self.cancel_training = False
		self.new_line_count = 0
		self.canvas_labels = []
		self.settings_file_name = "resources/settings.json"
		self.can_clear_graph = False
		self.opt_bgcolor = "#424e6f"
		self.data_processor = data_processor(self)
		self.render_ui_frames()
		self.render_ui_widgets()

	def quit_all(self):
		self.tk_main.destroy();
		sys.exit();

	def render_ui_frames(self):

		self.learn_options_frame = Frame(self.ui_frame,width=500,background=self.main_bg)
		self.learn_options_frame.pack(fill=BOTH,side=LEFT)
		#self.console_frame = Frame(self.ui_frame,bg="grey",height=300,width=400)
		#self.console_frame.pack()
		self.c_scrollbar = Scrollbar(self.tk_main)
		self.c_scrollbar.pack(side=RIGHT, fill=Y)

		self.lower_frame = Frame(self.ui_frame,background=self.main_bg)
		self.lower_frame.pack(side=BOTTOM, fill=BOTH)
		self.console_list_box = Text(self.lower_frame,height=16,width=34,borderwidth=0, highlightthickness=0,bg="#212737",fg="green",font=("courier",9))
		self.console_list_box.pack(ipady=20,ipadx=10,side=LEFT,fill=Y)
		self.console_list_box.config(yscrollcommand=self.c_scrollbar.set)
		self.console_list_box.configure(state="disabled")
		self.console_list_box.configure(wrap=WORD)
		self.c_scrollbar.config(command=self.console_list_box.yview)
		self.tk_nn_visual_canvas = Canvas(self.ui_frame, width=self.canvas_width, height=self.canvas_height,background="#424e6f",highlightthickness=0)
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

		self.g_figures[line_num] =  plt.figure(facecolor=self.main_bg)
		self.g_axis[line_num] = self.g_figures[line_num].add_subplot(111,axisbg="#b3b8c5")
		self.g_axis[line_num].set_ylabel(ylabel)
		self.g_axis[line_num].set_xlabel(xlabel)
		self.g_figures[line_num].text(0.5,0.97,title,horizontalalignment='center',fontsize=9)
		self.g_axis[line_num].get_yaxis().set_visible(False)
		self.g_axis[line_num].get_xaxis().set_visible(False)
		self.g_canvas[line_num] = FigureCanvasTkAgg(self.g_figures[line_num], master=self.lower_frame)
		self.g_canvas[line_num].get_tk_widget().config(width=340,height=280)
		self.g_canvas[line_num].get_tk_widget().pack(side=LEFT,fill=X)


	def render_canvas_info_labels(self):
		self.canvas_info_labels = {}
		self.canvas_info_label_vals = {}
		self.canvas_label_names = ["Latest Success", "Epoch Duration"]
		label_y = 30
		for label_name in self.canvas_label_names:
			self.canvas_info_label_vals[label_name] = StringVar()
			self.canvas_info_label_vals[label_name].set(label_name+": N/A")
			self.canvas_info_labels[label_name] = Label(self.mid_labels_frame, textvariable=self.canvas_info_label_vals[label_name],font=(self.font_face, self.main_font_size),bg=self.main_bg)
			self.canvas_info_labels[label_name].pack(side=BOTTOM)
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
		self.g_lines[line][-1].set_ydata(ydata)
		self.g_lines[line][-1].set_xdata(range(len(ydata)))
		self.g_axis[line].relim()
		self.g_axis[line].autoscale_view()

		'''	if(line==1): 
			if(round(data)!=self.prev_line_1_data):
				data = round(data,2)
				self.all_g1_annotations.append(self.g_axis[line].annotate(str(data)+"%",(len(ydata)-1,data)))
				self.all_g1_annotations[-1].set_fontsize(7)
			self.prev_line_1_data = round(data)'''

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

	def show_alert(self,header, msg):
		tkMessageBox.showinfo(header, msg)


	input_text_length = 8
	default_hidden_layers_str = "10,20"
	default_bias_str = "1,1,0"
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

	def render_dataset_opts(self):
		avaliable_datasets = ["--select--"]
		for file in self.data_processor.get_avaliable_datasets("new"):
			avaliable_datasets.append(file)
		self.input_fields["dataset_name"] = self.render_input_field(0,"Dataset File","Chose avaliable text file",5,self.learn_options_frame,drop=avaliable_datasets,command=self.update_expected_nn_io_fields)

	def render_ui_widgets(self):
		self.render_nn_vis_trigger()
		
		icon = ImageTk.PhotoImage(Image.open("resources/perceptron-header.png").resize((230, 100), Image.ANTIALIAS))
		self.icon_view = Label(self.learn_options_frame,image=icon,highlightthickness=0,bg=self.main_bg)
		self.icon_view.image = icon
		self.icon_view.pack()
	
		self.choose_settings_frame = Frame(self.learn_options_frame)
		self.choose_settings_frame.pack()
		self.render_settings_opts()
		

		self.all_drops = {}
		self.input_fields = {}
		self.input_labels = {}
		self.input_descs = {}
		self.widget_frames = {}
		self.input_descs_vis = {}

		self.open_prepro_window = self.render_option("DATA PREPROCESSOR", self.preprocess_data_render, self.learn_options_frame,width=18)

		self.render_dataset_opts()
		self.input_fields["data_to_retrieve"]= self.render_input_field("all", "Data To Use","Enter 'all' or number of items to use from dataset",self.input_text_length,self.learn_options_frame)
		self.input_fields["matrix_dims"] = self.render_input_field(self.default_input_dims,"Matrix Input Dimensions","Enter single value or enter height, width of matrix with comma",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["output_count"] = self.render_input_field(self.default_output_count, "Output Count","Enter output quantity",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["hidden_layer"] = self.render_input_field(self.default_hidden_layers_str, "Hidden Layers", "Enter comma seperated list of hidden layer sizes",self.input_text_length, self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["bias_vals"] = self.render_input_field(self.default_bias_str, "Bias Values", "List must match hidden layer count plus output, but enter 0 for no bias",self.input_text_length,self.learn_options_frame,command=self.render_nn_vis_trigger)
		self.input_fields["learning_rate"] = self.render_input_field("0.5", "Learning Rate","Enter decimal or integer",self.input_text_length,self.learn_options_frame)
		self.input_fields["weight_range"] = self.render_input_field("-1,1", "Weight Ranges","Enter one value (or two for a range) for initial weight values",self.input_text_length, self.learn_options_frame)
		self.input_fields["epochs"] = self.render_input_field("100", "Dataset Iterations","Total number of iterations through all data loaded",self.input_text_length, self.learn_options_frame)
		self.input_fields["test_data_partition"] = self.render_input_field("10", "Data for Testing","Amount of data to partition from dataset for result testing",self.input_text_length, self.learn_options_frame)
		
		self.mid_labels_frame = Frame(self.learn_options_frame,bg=self.main_bg)
		self.mid_labels_frame.pack(expand=True,fill=BOTH)
		self.render_canvas_info_labels()
		self.lower_sect = Frame(self.learn_options_frame,background=self.main_bg)
		self.lower_sect.pack(expand=True,fill=BOTH)
		self.opt_cols = Frame(self.lower_sect,bg="red")
		self.opt_cols.pack(side=TOP,expand=True)
		self.left_opt_col = Frame(self.opt_cols,background=self.main_bg)
		self.left_opt_col.pack(side=LEFT)
		self.right_opt_col = Frame(self.opt_cols,background=self.main_bg)
		self.right_opt_col.pack(side=RIGHT)

		self.start_learning_opt = self.render_option("Start Learning", self.start_learning_ui_request, self.left_opt_col)
		self.cancel_learning_opt = self.render_option("Stop Learning", self.cancel_learning, self.left_opt_col)
		self.cancel_learning_opt.config(state="disabled")
		self.clear_graphs_opt = self.render_option("Clear Graphs", self.clear_graphs, self.right_opt_col)
		self.save_settings_opt = self.render_option("Save Settings",self.save_settings,self.right_opt_col)
		self.save_nn_opt = self.render_option("Export Trained NN",self.save_nn,self.right_opt_col)
		self.save_nn_opt.config(state="disabled")
		self.test_input_opt = self.render_option("Test With Input",self.test_input, self.left_opt_col)
		

		self.print_console("Welcome to Perceptron. To get started, preprocess a dataset and then design a neural network for it to use. For more information, see the README file. Click this console to scroll it.")

	def render_input_field(self,default_value, label_text,desc_text,width,parent_frame,command=None,drop=None):
			label_text = label_text+": "
			self.widget_frames[label_text] = Frame(parent_frame,background=self.main_bg)
			self.widget_frames[label_text].pack(fill=X,expand=False)

			desc_frame = Frame(self.widget_frames[label_text], width=50, height=0,background=self.main_bg)
			desc_frame.pack(fill=None,side=BOTTOM,expand=False)

			if(drop!=None):
				input_widget_val = StringVar(self.tk_main)
				input_widget = OptionMenu(self.widget_frames[label_text], input_widget_val,command=command,*drop)
				input_widget.config(bg=self.opt_bgcolor)
				input_widget.config(relief=FLAT)
				input_widget["menu"].config(bg=self.opt_bgcolor)
				input_widget.config(highlightthickness=0)
				input_widget["menu"].config(bg=self.opt_bgcolor)
				input_widget.config(foreground="white")
				self.all_drops[label_text] = input_widget
				input_widget.config(width=15)
				input_widget_val.set(drop[default_value])
			else:
				input_widget = Entry(self.widget_frames[label_text], width=38-len(label_text),bg="#545f7d",font=(self.font_face,11))
				input_widget.insert(0,str(default_value))
				if(command!=None):
					input_widget.bind("<KeyRelease>", command)
			
			input_widget.pack(side=RIGHT,padx=3,ipady=3)

			if(drop!=None): input_widget = input_widget_val

			self.input_labels[label_text] = Label(self.widget_frames[label_text], text=label_text,background=self.main_bg,font=(self.font_face, self.main_font_size))
			self.input_labels[label_text].pack(side=LEFT)
			self.widget_frames[label_text].bind("<Enter>", self.toggle_desc_label)
			self.widget_frames[label_text].bind("<Leave>", self.toggle_desc_label)
			self.input_descs[label_text] = Label(desc_frame, text="*"+desc_text,background=self.main_bg, font=(self.font_face, 1), fg=self.main_bg,wraplength=210)
			self.input_descs[label_text].pack(side=BOTTOM)
			self.input_descs_vis[label_text] = 0
			return input_widget

	def render_option(self,text, command,parent_frame,side=None,anchor=None,width=None,bg=None):
		if(width==None): width = 14
		if(bg==None): bg = self.opt_bgcolor
		option = Button(parent_frame, text=text, command=command,relief=FLAT,width=width, bg=bg,bd=3,foreground="white")
		option.pack(side=side,anchor=anchor,padx=3,pady=3)
		return option

	def update_expected_nn_io_fields(self,event=None):
		if(self.input_fields["dataset_name"].get() != "--select--"):
			dataset = open(self.data_processor.folders_for_data["new"]+"/"+self.input_fields["dataset_name"].get(), 'r').read().split("\n")
			self.dataset_row_count = len(dataset)-1
			t_type = dataset[0]
			sample_dataset_row = dataset[1].split(",")
			self.expected_input_count = len(sample_dataset_row)-1
			self.expected_hidden_count = int(round(math.sqrt(int(round(self.expected_input_count))))+10)
			if(t_type[1]!="B"):
				self.expected_output_count = len(sample_dataset_row[-1].split("/"))
			else:
				self.expected_output_count = int(t_type[7:])
			self.input_fields["output_count"].delete(0,END)
			self.input_fields["output_count"].insert(0,self.expected_output_count)
			self.input_fields["matrix_dims"].delete(0,END)
			self.input_fields["matrix_dims"].insert(0,self.expected_input_count)
			self.input_fields["hidden_layer"].delete(0,END)
			self.input_fields["hidden_layer"].insert(0,self.expected_hidden_count)
			self.input_fields["bias_vals"].delete(0,END)
			self.input_fields["bias_vals"].insert(0,"0,0")
			self.render_nn_vis_trigger(event=True)


	def toggle_desc_label(self, event):
		label_text = event.widget.winfo_children()[2].cget("text")
		if(self.input_descs_vis[label_text] % 2 ==0):
			self.input_descs[label_text].configure(fg="#60606b")
			self.input_descs[label_text].configure(font=(self.font_face, 10))
		else:
			self.input_descs[label_text].configure(fg=self.main_bg)
			self.input_descs[label_text].configure(font=(self.font_face, 1))
		self.input_descs_vis[label_text] += 1

	def save_nn(self):
		nn_name = tkSimpleDialog.askstring("Saving Neural Network", "Neural Net Name: ")
		if(nn_name):
			weight_layers = self.neural_network.all_weights
			weights_as_list = []
			for layer in weight_layers:
				l_layer = []
				for w_group in layer:
					l_group = []
					for w in w_group:	
						l_group.append(w)
					l_layer.append(l_group)
				weights_as_list.append(l_layer)

			weights_as_json = json.dumps(weights_as_list)
			
			new_file = open("saved/nn_"+nn_name+".txt", "a")
			new_file.write(weights_as_json)

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
		self.saved_settings_opts.config(width=16)
		self.saved_settings_opts.config(bg=self.opt_bgcolor)
		self.saved_settings_opts["menu"].config(bg=self.opt_bgcolor)
		self.saved_settings_opts.config(foreground="white")
		self.saved_settings_opts.config(highlightthickness=0)
		self.saved_settings_opts["menu"].config(foreground="white")
		self.saved_settings_opts.config(relief=FLAT)
		self.saved_settings_opts.pack()
		self.saved_settings_text.set(saved_settings[0])

	def save_settings(self):
		settings_name = tkSimpleDialog.askstring("Saving Settings", "Setting's Name: ")
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
				all_settings_as_str = json.dumps(all_settings_as_json)
				settings_file_write = open(self.settings_file_name, "w")
				settings_file_write.write(all_settings_as_str)
				settings_file_read.close()	
				settings_file_write.close()
				self.saved_settings_opts.destroy()
				self.render_settings_opts()
	

	def check_str_list_valid(self,string):
		valid_str_entry = True
		for char in string:
			if(char!="," and char.isdigit()==False):
				valid_str_entry = False
				break

		return valid_str_entry

	def map_to_int_if_valid(self,string):
		if(self.check_str_list_valid(string)==False):
			result = False
		elif string == "":
			result = []
		else:
			string = self.data_processor.real_strip(string,[","])
			result = self.data_processor.strip_row_list(string.split(","))
			result = map(int,result)
		return result

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

		guess_str_val = StringVar()
		label_guess = Label(mini_cam_window, text="",font=(self.font_face, 20),textvariable=guess_str_val)
		label_guess.pack()

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

			roi_matrix = cv2.resize(roi_matrix, (28,28))
			matrix_float = self.data_processor.prep_matrix_for_input(roi_matrix)
			outline_vals = [matrix_float[0,:-1], matrix_float[:-1,-1], matrix_float[-1,::-1], matrix_float[-2:0:-1,0]]
			outline_sum = np.concatenate(outline_vals).sum()
			if(int(outline_sum) == 0):
				self.neural_network.feed_forward(matrix_float.flatten())
				output_neurons = self.neural_network.nn_neurons[-1].tolist()
				max_val = max(output_neurons)
				if(max_val > 0.9):
					guess_val = output_neurons.index(max_val)
					guess_str_val.set(guess_val)

			cv2.rectangle(cv_frame,roi_point_1,roi_point_2, (255), thickness=3, lineType=8, shift=0)

			img_frame = Image.fromarray(cv_frame)
			tk_frame = ImageTk.PhotoImage(image=img_frame)
			label_for_cam.imgtk = tk_frame
			label_for_cam.configure(image=tk_frame)
			label_for_cam.after(10, render_cam_frame) 

		render_cam_frame()


	def preprocess_data_render(self):
		self.preprocess_window = Toplevel(self.ui_frame,width=300,height=400)
		self.preprocess_form = Frame(self.preprocess_window,background=self.main_bg)
		self.preprocess_form.pack(side=LEFT, fill=BOTH)

		self.preprocess_inter_viewer = Frame(self.preprocess_window,bg=self.main_bg)
		self.preprocess_inter_viewer.pack(side=RIGHT, fill=BOTH)
		self.inter_viewer_header= Label(self.preprocess_inter_viewer, text="Samples of processed data...",font=(self.font_face, self.main_font_size),bg=self.main_bg)
		self.inter_viewer_header.pack()
		self.v_scrollbar = Scrollbar(self.tk_main)
		self.v_scrollbar.pack(side=RIGHT, fill=Y)
		self.inter_viewer_box = Text(self.preprocess_inter_viewer,bg="#b3b8c5",height=16,width=100,borderwidth=0, highlightthickness=0,font=("courier bold", 10))
		self.inter_viewer_box.pack(padx=3,ipady=20,ipadx=10,side=RIGHT,fill=Y)
		self.inter_viewer_box.config(yscrollcommand=self.v_scrollbar.set)
		self.inter_viewer_box.configure(state="disabled")
		self.v_scrollbar.config(command=self.inter_viewer_box.yview)

		self.AtN_tran_fields = []
		self.prepro = {}

		avaliable_datasets = ["--select--"]
		for file in self.data_processor.get_avaliable_datasets("old"):
			avaliable_datasets.append(file)
		self.prepro["original_file"]= self.render_input_field(0,"Dataset File","Chose avaliable text file",5,self.preprocess_form,drop=avaliable_datasets, command=self.update_prepro_viewer_for_struct)

		self.prepro["row_separator_char"] = self.render_input_field("\n", "Row Separator Char","Enter the character that separates each row, default is a '\\n' ",self.input_text_length, self.preprocess_form,command=self.update_prepro_viewer_for_struct)
		ig_first_row_opts = ["No", "Yes"]
		self.prepro["ignore_first_row"] = self.render_input_field(0,"Ignore First Row","If first row are labels/column names, remove them.",5,self.preprocess_form,drop=ig_first_row_opts,command=self.update_prepro_viewer_for_struct)
		self.prepro["fields_to_ignore"] = self.render_input_field("", "Fields to Ignore","Enter position of fields to be removed/ignored (where first field is 0) ",self.input_text_length,self.preprocess_form,command=self.update_prepro_viewer_for_struct)
		
		self.prepro["fields_to_min"] = self.render_input_field("", "Fields For Minimisation","Enter position of fields that need minimising (where first field is 0)",self.input_text_length,self.preprocess_form,command=self.add_min_field)
		self.prepro_mins_frame = Frame( self.preprocess_form,bg=self.main_bg)
		self.prepro_mins_frame.pack(fill=BOTH)

		self.prepro["found_alphas_trans"] = {}
		self.alpha_trans_opt = self.render_option("Translate Alphas", self.render_trans_alpha_window, self.preprocess_form,width=20)
		#self.alpha_trans_opt.configure(state="disabled")

		target_types = ["--select--","Binary", "Real"]
		self.prepro["target_val_pos"] = self.render_input_field("", "Target position(s)","Enter position of fields that are target values",self.input_text_length,self.preprocess_form,command=self.update_prepro_viewer_for_struct)
		self.prepro["target_type"] = self.render_input_field(0,"Target Value Type","Choose binary or numeric",5,self.preprocess_form,drop=target_types, command=self.prepro_vb_change)
		
		self.prepro_vb_frame = Frame(self.preprocess_form,bg=self.main_bg)
		self.prepro_vb_frame.pack(fill=BOTH)
		self.prepro["bin_range"] = None

		self.prepro_opt = self.render_option("PROCESS", self.start_preprocess, self.preprocess_form)
		self.reset_opt = self.render_option("RESET", self.reset_prepro, self.preprocess_form)

	def reset_prepro(self):
		self.preprocess_window.destroy()
		self.preprocess_data_render()

	def render_trans_alpha_window(self,event=None):
		self.prepro_transAN_frame = Toplevel( self.ui_frame,width=200,height=200,bg=self.main_bg)
		self.prepro_transAN_frame.protocol('WM_DELETE_WINDOW', self.unset_alpha_fields)
		has_found_alphas = False
		for field in self.data_processor.found_alphas:
			if(len(self.data_processor.found_alphas[field])>0):
				has_found_alphas = True
				alphas_found_as_str =  ','.join(str(e) for e in self.data_processor.found_alphas[field])
				label_txt = "field_"+str(field)+" alphas found: " + alphas_found_as_str
				new_trans_field = self.render_input_field("", label_txt, "Enter values",self.input_text_length,self.prepro_transAN_frame,command=self.update_prepro_viewer_for_struct)
				default_trans_str = ','.join(str(i) for i in range(0,len(self.data_processor.found_alphas[field])))
				new_trans_field.insert(0,default_trans_str)
				self.prepro["found_alphas_trans"][field]=new_trans_field
		if(has_found_alphas == True):
			self.update_prepro_viewer_for_struct()
			self.render_option("Revert To Alphas", self.revert_fields_to_alpha,self.prepro_transAN_frame,width=20)
		else:
			Label(self.prepro_transAN_frame, text="No alphas found",font=(self.font_face, self.main_font_size)).pack()

	def revert_fields_to_alpha(self):
		for field in self.prepro["found_alphas_trans"]:
			self.prepro["found_alphas_trans"][field].delete(0,END)
			self.prepro["found_alphas_trans"][field].insert(0,"")
		self.update_prepro_viewer_for_struct()

	def unset_alpha_fields(self):
		self.prepro_transAN_frame.destroy()
		self.prepro["found_alphas_trans"] = {}

	def update_prepro_viewer_for_struct(self,event=None):
		if(os.path.isfile(self.data_processor.folders_for_data["old"]+"/"+self.prepro["original_file"].get())):
			if(self.prepro["row_separator_char"].get() == "\\n"):
				self.prepro["row_separator_char"].delete(0,END)
				self.prepro["row_separator_char"].insert(0,"\n")
			prepro_vals = self.data_processor.validate_prepro()
			struct_str = self.data_processor.struct_dataset(True, True,prepro_vals)
			self.update_viewer_text(struct_str)

	def update_viewer_text(self, text):
		self.inter_viewer_box.configure(state="normal")
		self.inter_viewer_box.delete(1.0,END)
		self.inter_viewer_box.insert(INSERT, text)
		self.inter_viewer_box.configure(state="disabled")

	def add_AtN_field(self,event=None):
		AtN_field_val = self.prepro["alpha_to_num_fields"].get()
		if(self.check_str_list_valid(AtN_field_val)):
			self.AtN_tran_fields = []
			AtN_fields = self.map_to_int_if_valid(AtN_field_val)
			if(AtN_fields!=False):
				self.clear_frame(self.prepro_transAN_frame)
				for field in AtN_fields:
					new_field_alpha = self.render_input_field("", "String in field_"+str(field),"Enter the string/char/word/phrase(s) that needs translation as a list",self.input_text_length,self.prepro_transAN_frame,command=self.update_prepro_viewer_for_struct)
					new_field_num = self.render_input_field("", "Number for field_"+str(field),"Enter your desired replacement numeric value(s) as a list that links with the string field",self.input_text_length,self.prepro_transAN_frame,command=self.update_prepro_viewer_for_struct)
					self.AtN_tran_fields.append([new_field_alpha,new_field_num])

	def add_min_field(self,event=None):
		min_val = self.prepro["fields_to_min"].get()
		if(min_val == "all"):
			self.min_fields = []
			self.clear_frame(self.prepro_mins_frame);
			min_field_val = self.render_input_field("", "Minimise All Fields By", "Enter the divider value",self.input_text_length,self.prepro_mins_frame,command=self.update_prepro_viewer_for_struct)
			min_field_except = self.render_input_field("", "...Except Fields", "Enter the field positions that shouldn't be divided, or leave blank",self.input_text_length,self.prepro_mins_frame,command=self.update_prepro_viewer_for_struct)
			self.min_fields.append(min_field_val)
			self.min_fields.append(min_field_except)
		elif(self.check_str_list_valid(min_val)):
			min_fields_list = self.map_to_int_if_valid(min_val)
			if(min_fields_list!=False):
				self.min_fields = []
				self.clear_frame(self.prepro_mins_frame)
				for field in min_fields_list:
					min_field = self.render_input_field("", "Minimise field_"+str(field)+" By", "Enter the divider",self.input_text_length,self.prepro_mins_frame,command=self.update_prepro_viewer_for_struct)
					self.min_fields.append(min_field)

	def clear_frame(self,frame):
		for field in frame.winfo_children():
			field.destroy()

	def prepro_vb_change(self,value):
		self.clear_frame(self.prepro_vb_frame)
		if(value == "Binary"):
			self.prepro["bin_range"] = self.render_input_field("","Binary Vector Range","The binary vector range (number of classes)",self.input_text_length,self.prepro_vb_frame,command=self.update_prepro_viewer_for_struct)
		else:
			self.prepro["bin_range"] = None

	def start_preprocess(self):
		poss_errors = self.data_processor.validate_prepro()["error"]
		if(poss_errors == ""):
			prepro_vals = self.data_processor.validate_prepro()
			self.preprocess_window.destroy()
			thread.start_new_thread(self.data_processor.struct_dataset,(False,False,prepro_vals))
		else:
			tkMessageBox.showinfo("Error", poss_errors)

	def refresh_data_drop(self):
		self.all_drops["Dataset File: "].destroy()
		self.render_dataset_opts()

	def test_input(self):
		input_str = tkSimpleDialog.askstring("Enter Input", "Enter the name of an image file, text file, enter row data manually: ")
		if(input_str):
			file_type_pos = input_str.rfind(".")
			valid_files = ["png","jpg","txt"]
			file_type_str = ""
			'''if(input_str == "camera"):
				self.render_camera()
			else:'''
			if(file_type_pos != -1):
				file_type_str = input_str[file_type_pos+1:]
			
			if(file_type_str not in valid_files or file_type_str == "txt"):
				if(file_type_str == "txt"):
					input_str = open(input_str, 'r').read()
				input_data = input_str.split(",")
				matrix_ready = self.data_processor.prep_matrix_for_input(np.asarray(input_data))
				
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
				print(output_neurons)
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
		self.save_nn_opt.config(state="normal")
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

		if(matrix_dims_str.isdigit()==False):
			error = "Invalid input count"
		else:
			valid_values['matrix_dims'] =int(matrix_dims_str)

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

		valid_values["data_file_name"] = self.data_processor.folders_for_data["new"]+"/"+valid_values["data_file_name"]
		if(os.path.isfile(valid_values["data_file_name"])==False):
			error = "File does not exist in processed_datasets"

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
			if(valid_values['data_to_test'] > 50):
				error = "Data to test should be under 50%"

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
			self.save_nn_opt.config(state="disabled")
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

			self.curr_dataset_name = field_result['data_file_name']
			self.matrix_dims = field_result['matrix_dims']
			self.data_processor.load_matrix_data(field_result['to_retrieve'],field_result['data_file_name'],self)
			self.data_processor.populate_matrices()
			self.prev_to_retrieve = self.data_processor.to_retrieve
			self.input_neuron_count = field_result['matrix_dims']
			self.matrix_data = self.data_processor.matrices
			self.matrix_targets = self.data_processor.targets
			self.t_type = self.data_processor.t_type
		
		self.neural_network = neural_network()
		self.neural_network.initilize_nn(field_result['hidden_layers'],
				self.input_neuron_count,field_result['output_count'], self.matrix_data,self.matrix_targets,
				field_result['biases_for_non_input_layers'], field_result['learning_constant'], 
				testing_mode,field_result['weight_range'],field_result['epochs'],field_result['data_to_test'],
				self.t_type,self.dataset_row_count,self)
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
		neuron_hidden_c = "#db7070"
		neuron_outter_c = "#8bd78f"
		line_color = "#a0a6b7"

		bias_pos_diff_x = 50
		bias_pos_diff_y = 50
		bias_color = "#837FD3"
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
					bias_connector = self.tk_nn_visual_canvas.create_line(neuron_x, neuron_y, bias_x_pos,bias_y_pos,fill=line_color)
					self.tk_nn_visual_canvas.tag_lower(bias_connector)

				neuron_dist_y = (neuron_radius*2) + neuron_padding
				if(neuron_layer < len(layers)-1):
					length_of_next_layer = layers[neuron_layer+1]
					if(example_p_limit_count > 0 and example_p_limit_count < length_of_next_layer):
						length_of_next_layer = example_p_limit_count
					neuron_y_for_line = (self.canvas_height - (length_of_next_layer)*(neuron_radius*2 + neuron_padding))/2
					
					for neuron_weights in range(0,length_of_next_layer):
						neuron_connector = self.tk_nn_visual_canvas.create_line(neuron_x, neuron_y, neuron_x+neuron_dist_x, neuron_y_for_line,fill=line_color)
						self.tk_nn_visual_canvas.tag_lower(neuron_connector)

						neuron_y_for_line += neuron_dist_y

				neuron_y += neuron_dist_y
			neuron_x += neuron_dist_x