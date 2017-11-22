from __future__ import print_function
import re,numpy as np,random,math,time,decimal,os,json
import FileDialog


class data_processor():
	
	folders_for_data = {"new":"processed_datasets","old":"original_datasets"}
	def __init__(self, ui):
		self.user_interface = ui

	found_alphas = {}
	prev_file =  ""
	row_len = -1
	target_has_encoded_alphas = False
	def struct_dataset(self, for_viewer, return_str,prepro_vals):
		# Update live viewer and if ready, update dataset
		new_dataset_str = ""
		new_dataset = []
		if(len(prepro_vals["row_separator_char"]) > 0 and os.path.isfile(self.folders_for_data["old"]+"/"+prepro_vals["original_file"])):
			data_by_row = open(self.folders_for_data["old"]+"/"+prepro_vals["original_file"], 'r').read().split(prepro_vals["row_separator_char"])
			dataset_meta = {}
			dataset_meta["target_info"] = [prepro_vals["target_type"],prepro_vals["bin_range"],prepro_vals["target_val_pos"]]
			dataset_meta["minimisations"] = prepro_vals["minimisations"]
			dataset_meta["alphas"] = self.found_alphas
			dataset_meta["fields_to_ignore"] = prepro_vals["fields_to_ignore"]
			if(for_viewer == False):
				name_for_new = prepro_vals["original_file"][0:prepro_vals["original_file"].rfind(".")]
				new_txt_file = open(self.folders_for_data["new"]+"/"+name_for_new+"_new.txt", "a")
				if(prepro_vals["bin_range"]==None): prepro_vals["bin_range"] = ""
				dataset_meta_str = json.dumps(dataset_meta)
				new_txt_file.write(dataset_meta_str)
				new_txt_file.write("\n--\n")
				if(prepro_vals["rows_for_testing"] != None):
					new_testing_file = open(self.folders_for_data["old"]+"/"+name_for_new+"_testing.txt", "a")
				self.user_interface.print_console("WRITING PROCESSED DATASET...")
			if(len(data_by_row)>1):
				if(for_viewer):
					end = 8
				else:
					end = len(data_by_row)
				start = 0

				if(len(self.found_alphas)>0):
					for ig in prepro_vals["fields_to_ignore"]:
						if(ig in self.found_alphas.keys()):
							del self.found_alphas[ig]

				if(prepro_vals["ignore_first_row"]): start += 1
				#prepro_vals["fields_to_ignore"].append(prepro_vals["target_val_pos"])
				for row_i in range(start,end):
					if(row_i not in prepro_vals["rows_for_testing"]):
						row = data_by_row[row_i].split(",")
						if row_i == start: self.row_len = len(row)
						row = self.strip_row_list(row)
						new_row,new_target_list,sing_bin_target = self.change_data_to_processed(dataset_meta,row,prepro_vals["target_val_pos"])

						new_target_list = new_target_list[1:]
						encoded_row_str = ""
						if(for_viewer == False):
							new_row.append(new_target_list)
						elif self.user_interface.is_viewing_trans:
							encoded_row = []
							temp_new_row = new_row
							for to_ig in prepro_vals["fields_to_ignore"]:
								temp_new_row.insert(to_ig,None)
							for el_i in range(0,len(temp_new_row)):
								if(temp_new_row[el_i] != None):
		
									vec = self.alpha_class_to_binary_vector(temp_new_row[el_i],self.found_alphas[el_i])
									if(len(vec)>0): encoded_row.append(vec)

							encoded_row_str = ','.join(str(e) for e in encoded_row)
			
						if(len(encoded_row_str)>0):
							row_str = encoded_row_str
						else:
							row_str = ','.join(str(e) for e in new_row)

						if(new_target_list != "" and new_target_list.replace("/","").replace(".","").isdigit() == False):
							self.target_has_encoded_alphas = True
						else:
							self.target_has_encoded_alphas = False

						if(self.user_interface.is_viewing_trans):
							encoded_targets = []
							for targ_pos in prepro_vals["target_val_pos"]:
								vec = self.alpha_class_to_binary_vector(row[targ_pos],self.found_alphas[targ_pos])
								if(len(vec)>0):
									encoded_targets.append(vec)

							new_target_list_for_dis = "[" +(','.join(str(e) for e in encoded_targets))+ "]"	
						else:
							new_target_list_for_dis = "["+new_target_list.replace("/",",")+"]"
						# Make target structure and turn into string for viewing
						if(new_target_list_for_dis != "[]"):
							new_target_list_for_dis = "with target(s): "+new_target_list_for_dis
						else:
							new_target_list_for_dis = ""
						if(for_viewer and sing_bin_target != None and str(sing_bin_target).isdigit()):
							range_ = prepro_vals['bin_range']
							target_vec_example = self.populate_binary_vector(int(sing_bin_target),int(range_))
							target_vec_ex_str = ','.join(str(e) for e in target_vec_example)
							target_vec_ex_str = "   (as binary vector: [" + target_vec_ex_str + "] )"
							new_target_list_for_dis += target_vec_ex_str

						new_dataset.append(new_row)
						vis_sep = "\n"
						if(for_viewer): 
							if(len(new_target_list_for_dis)>0):
								vis_sep  = "\n *** "+new_target_list_for_dis+" *** \n\n"
							new_dataset_str += row_str + vis_sep + "\n"
						else:
							new_txt_file.write(row_str + vis_sep)
							if(len(data_by_row)>20):
								if(row_i%(int(len(data_by_row)/6))==0):
									percentage = int(row_i/len(data_by_row))*100
									msg = "Written "+str(row_i)+"/"+str(len(data_by_row))+" rows"
									self.user_interface.print_console(msg)

					elif(for_viewer == False):
						new_testing_row = []
						testing_row = data_by_row[row_i].split(",")
						targs_show = []
						for targs in prepro_vals["target_val_pos"]:
							targs_show.append(testing_row[targs])
							del testing_row[targs]
						new_testing_file.write((','.join(str(e) for e in testing_row))+"\n"+"...With correct targets: "+(','.join(str(e) for e in targs_show))+"\n\n")	


				if(for_viewer and (len(self.found_alphas)==0 or prepro_vals["original_file"] != self.prev_file)):
					self.find_alpha_classes(data_by_row,prepro_vals["fields_to_ignore"])
					self.target_has_encoded_alphas = False
				
				self.prev_file = prepro_vals["original_file"]
				if(for_viewer == False):
					self.user_interface.print_console("Finished processing "+name_for_new+".txt,  Check the "+self.folders_for_data["new"]+" folder")
					self.user_interface.render_dataset_opts(True)
				if(return_str):
					return new_dataset_str
				else:
					return new_dataset



	def find_alpha_classes(self,data_by_row,f_ig):
		self.found_alphas = {}
		has_found_alpha = False
		# Start finding  all classification alphas per field 
		for row_i in range(1,len(data_by_row)):
			row = data_by_row[row_i].split(",")
			row = self.strip_row_list(row)
			for el_i in range(0,len(row)):
				if(el_i not in f_ig):
					if(row_i == 1):
						self.found_alphas[el_i] = []
					elif row_i == 2 and has_found_alpha == False:
						break

					element = self.real_strip(row[el_i])
					if(element not in self.found_alphas[el_i] and str(element).replace(".","").replace("-","").isdigit()==False):
						self.found_alphas[el_i].append(element)
						has_found_alpha = True
			if(has_found_alpha == False):
				break


	def alpha_class_to_binary_vector(self,alpha_val,dataset_meta_alphas_list):
		class_range = len(dataset_meta_alphas_list)
		if(class_range>0):
			if(alpha_val != None):
				if(str(alpha_val) not in dataset_meta_alphas_list):
					bin_vector = []
				else:
					target = dataset_meta_alphas_list.index(alpha_val)
					bin_vector = self.populate_binary_vector(target,class_range)
			else:
				bin_vector = []
		else:
			bin_vector = [float(alpha_val)]

		return bin_vector


	def validate_prepro(self):

		prepro_vals = {}
		valid_for_viewer = True
		error = ""
		prepro_vals["original_file"] = self.user_interface.prepro["original_file"].get()
		prepro_vals["row_separator_char"] = self.user_interface.prepro["row_separator_char"].get()
		prepro_vals["ignore_first_row"] = self.user_interface.prepro["ignore_first_row"].get()
		prepro_vals["fields_to_min"] = self.user_interface.prepro["fields_to_min"].get()
		prepro_vals["fields_to_ignore"] = self.user_interface.prepro["fields_to_ignore"].get()
		prepro_vals["target_val_pos"] = self.user_interface.prepro["target_val_pos"].get()
		prepro_vals["target_type"] = self.user_interface.prepro["target_type"].get()
		prepro_vals["rows_for_testing"] = self.user_interface.prepro["rows_for_testing"].get()
		if(self.user_interface.prepro["bin_range"] == None):
			prepro_vals["bin_range"] = None
		else:
			prepro_vals["bin_range"] = self.user_interface.prepro["bin_range"].get()
		prepro_vals["error"] = ""
	
		if(prepro_vals["ignore_first_row"] == "Yes"): 
			prepro_vals["ignore_first_row"] = True
		else:
			prepro_vals["ignore_first_row"] = False

		if(prepro_vals["bin_range"] != None):
			if(prepro_vals["bin_range"].isdigit() == False):
				error = "Invalid binary range, must be integer"
			else:
				prepro_vals["bin_range"] = int(prepro_vals["bin_range"])

		if(os.path.isfile(self.folders_for_data["old"]+"/"+prepro_vals["original_file"]) == False):
			error = "File does not exist or is not in "+self.folders_for_data["old"]+" folder"

		if(prepro_vals["target_type"] == "--select--"):
			error = "You must choose a target type"

		rows_for_testing_list = self.user_interface.map_to_int_if_valid(prepro_vals["rows_for_testing"])
		if(rows_for_testing_list != False):
			prepro_vals["rows_for_testing"] = rows_for_testing_list
		else:
			as_range = prepro_vals["rows_for_testing"].split("-")
			if(len(as_range)==2):
				if(as_range[0].isdigit() and as_range[1].isdigit()):
					prepro_vals["rows_for_testing"] = range(int(as_range[0]),int(as_range[1]))
			else:
				error = "Invalid rows for testing"
				prepro_vals["rows_for_testing"] = None

		field_targ_try =prepro_vals["target_val_pos"].split(",")
		targ_kwords = {"first":0,"last":self.row_len-1}
		prepro_vals["target_val_pos"] = []
		for pos in field_targ_try:
			if(pos.isdigit() or pos in targ_kwords):
				if(pos in targ_kwords):
					pos = targ_kwords[pos]
				else:
					pos = int(pos)
				prepro_vals["target_val_pos"].append(pos)
			else:
				field_targ_try = False
				break
		if(field_targ_try == False):
			error = "Invalid target position(s)"
		else:
			if(prepro_vals["target_val_pos"] != False):
				if(len(prepro_vals["target_val_pos"])>1 and prepro_vals["target_type"]=="Binary"):
					error = "If you are using binary vectors, you can only have one target position"

		field_to_ig_try = self.user_interface.map_to_int_if_valid(prepro_vals["fields_to_ignore"])
		if(field_to_ig_try == False):
			error = "Invalid values to ignore"
		else:
			prepro_vals["fields_to_ignore"] = field_to_ig_try

		def validate_divider(val):
			if(val.replace(".","").isdigit() == False or val.replace(".","") == "0"):
				return 1
			else:
				return float(val)

		prepro_vals["minimisations"] = {}
		prepro_vals["minimisations"]["all"] = None
		prepro_vals["minimisations"]["except"] = []
		field_min_try = self.user_interface.map_to_int_if_valid(prepro_vals["fields_to_min"] )
		if(field_min_try == False):
			if(prepro_vals["fields_to_min"] == "all"):
				prepro_vals["minimisations"]["all"] = validate_divider(self.user_interface.min_fields[0].get())
				prepro_vals["minimisations"]["except"] = self.user_interface.map_to_int_if_valid(self.user_interface.min_fields[1].get())
			else:
				error = "Invalid alpha to num translation"
		else:
			prepro_vals["fields_to_min"]  = field_min_try
			c = 0
			for min_field in prepro_vals["fields_to_min"]:
				min_vals = self.user_interface.min_fields
				divider =  validate_divider(min_vals[c].get())
				prepro_vals["minimisations"][min_field] = divider
				c+=1

		prepro_vals["error"] = error
		return prepro_vals


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

	def change_data_to_processed(self, dataset_meta, row,target_pos=None):
		new_row = []
		sing_bin_target = None
		new_target_list = ""
		if(target_pos != None):
			is_valid_bin_target = (dataset_meta["target_info"][1] != None and len(target_pos)==1)
		if(len(row)>1):
			for el_i in range(0,len(row)):
				new_el = row[el_i]
				if(el_i not in dataset_meta["fields_to_ignore"]):
					if(new_el != None):
						if(new_el == str(new_el)):
							new_el=self.real_strip(new_el)
							if(dataset_meta["minimisations"]["except"] != False and str(new_el).replace(".","").isdigit()):
								if (dataset_meta["minimisations"]["all"] != None) and (el_i not in dataset_meta["minimisations"]["except"]):
									new_el = str(float(new_el)/dataset_meta["minimisations"]["all"])
								if(dataset_meta["minimisations"]["all"] == None):
									if(str(el_i) in dataset_meta["minimisations"]):
										el_i = str(el_i)
									if(el_i in dataset_meta["minimisations"]):
										new_el = str(float(new_el)/dataset_meta["minimisations"][el_i])
		
					if(target_pos != None):									
						if(el_i in target_pos):
							if(is_valid_bin_target):
								sing_bin_target = new_el
							new_target_list+="/"+str(new_el)
						else:
							new_row.append(new_el)
					else:
						new_row.append(new_el)

		return new_row,new_target_list,sing_bin_target


	
	def load_matrix_data(self, to_retrieve, file_name,user_interface):
		self.user_interface = user_interface
		self.to_retrieve = to_retrieve
		self.file_name = file_name
		self.user_interface.print_console("Loading "+str(self.to_retrieve)+" items from " + self.file_name + "... \n")
		self.dataset = open(file_name, 'r').read().split("\n")
		self.dataset_meta =json.loads(self.dataset[0])
		self.dataset_meta["alphas"] = self.sort_dataset_meta_alphas(self.dataset_meta["alphas"])
		self.has_alphas = self.meta_has_alphas(self.dataset_meta)
		self.matrices = []
		self.targets = []
		self.max_data_amount = int(len(self.dataset))-2
		if(self.to_retrieve == "all"):
			self.to_retrieve = self.max_data_amount

	def real_strip(self,string,extra_chars=None):
		discount_chars = ["'", '"']
		if(extra_chars != None): discount_chars = discount_chars + extra_chars
		string = string.strip()
		for char in discount_chars:
			if(len(string)>=2):
				if(string[0] == char and string[-1] == char):
					string = string[1:-1]
					break
		return string

	def strip_row_list(self,row):
		if(self.real_strip(row[-1]) == ""):
			del row[-1]
		elif(self.real_strip(row[0])) == "":
			del row[0]
		return row

	def sort_dataset_meta_alphas(self,dataset_meta_alphas):
		keys = []
		for field_pos in dataset_meta_alphas:
			keys.append(int(field_pos))
		keys.sort()
		new_meta = {}
		for key in keys:
			new_meta[key] = dataset_meta_alphas[str(key)]
		return new_meta

	def meta_has_alphas(self,meta):
		has_alphas = False
		for i in meta["alphas"]:
			if(len(meta["alphas"][i]) > 0):
				has_alphas = True;
				break;

		return has_alphas;


	def populate_matrices(self):
		
		px_count = 0
		done_msg = "Finished loading data \n "
		prev_pos_of_matrix = 0
		for i in range(2,self.to_retrieve-1):
			if(self.user_interface.cancel_training == True):
				done_msg = "**CANCELLED** \n "
				break
			flat_single_item = self.dataset[i].split(",")
			if(len(flat_single_item)>0):
				target_string = flat_single_item[-1]
				target_vals = target_string.split("/")
				del flat_single_item[-1]
				if(self.has_alphas == True):
					item_as_array = np.array(flat_single_item)
				else:
					item_as_array = np.asarray(flat_single_item, dtype=np.float32)
				self.matrices.append(item_as_array)
				self.targets.append(target_vals)
			if(self.to_retrieve > 10):
				if(i%(int(self.to_retrieve/5))==0):
					self.user_interface.print_console("Loaded "+str(i)+"/"+str(self.to_retrieve))
		self.user_interface.print_console(done_msg)

	def prep_matrix_for_input(self, matrix):
		matrix_float = matrix.astype(np.float32)
		matrix_for_input = matrix_float / float(255)
		return matrix_for_input

	def get_avaliable_datasets(self, from_):
		# Search orginial_datasets folder for text files
		avaliable_txts = []
		for f in os.listdir(self.folders_for_data[from_]):
			if(f[-4:] == ".txt"):
				avaliable_txts.append(f)
		return avaliable_txts

	def populate_binary_vector(self,target,output_count):
		# Take index value, and construct binary vector where element of index is 1
		vector = []
		target = int(target)
		if(target < output_count):
			for i in range(0,int(output_count)):
				vector.append(0)
			vector[target] = 1
			return vector
		else:
			return 0