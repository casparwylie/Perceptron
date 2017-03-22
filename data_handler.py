from __future__ import print_function
import re,numpy as np,random,math,time,decimal,os,json


class data_processor():
	
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
	
	def load_matrix_data(self, matrix_dims, to_retrieve, file_name,user_interface):
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

