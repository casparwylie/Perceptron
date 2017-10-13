from __future__ import print_function
import cv2,numpy as np,os
from pprint import pprint 
np.set_printoptions(threshold=np.nan)
class data_preprocessor_handler:
    def image_dir_to_matrix_txt(self, dirname):
        new_txt_file = open(dirname+".txt", "a")
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
               

                print(c)

    def show_matrices_from_file(self,file):
        self.matrix_width = 1
        self.matrix_height = 30
        self.to_retrieve = 30
        self.input_total = self.matrix_width * self.matrix_height
        self.data_set = open(file, 'r').read().split(",")
        matrices = []
        targets = []
        px_count = 0
        prev_pos_of_matrix = 0
        target_pos_in_row = -1
        for i in range(1,self.to_retrieve):
            pos_of_matrix = (i*(self.input_total))+i
            flat_single_item = self.data_set[prev_pos_of_matrix:pos_of_matrix]
           # print(flat_single_item)
            if(len(flat_single_item)>0):
                target_val = flat_single_item[target_pos_in_row]
                del flat_single_item[target_pos_in_row]
                item_as_array = np.asarray(flat_single_item)
                array_as_matrix = np.reshape(item_as_array,(self.matrix_width, self.matrix_height),order="A")
                matrices.append(array_as_matrix)
                targets.append(target_val)
              
                #cv2.imshow(str(i)+"...."+str(target_val),array_as_matrix)
                prev_pos_of_matrix = pos_of_matrix
        #print(matrices)
        print(targets)

    def normalise_text_file(self,text_file):
        
        target_val_pos = 1
        elements_to_ignore = [target_val_pos,0]
        new_txt_file = open(text_file+"_new.txt", "a")
        '''p_schar_intervals = []
        text_file = open(file, 'r').read()
        for schars in poss_split_chars:
            p_schar_repeats.append(0)

        for char in text_file:
            for schar_c in range(0,len(poss_split_chars)):
                if(poss_split_chars[schar_c] != char):
                    p_schar_intervals[schar_c] += 1
                else:
                    if(p_schar_intervals[schar_c] in range(real_row_count-1,real_row_count+1):
                        break'''

        data_by_row = open(text_file, 'r').read().split("\n")
        for row in data_by_row:
            row = row.split(",")
            new_row = []
            r_count = 0
            for element in row:
                if(r_count not in elements_to_ignore):
                    if(element.strip().isdigit()):
                        element = float(element)/255
                    new_row.append(str(element))
                r_count += 1
            new_row.append(row[target_val_pos])
            row_str = ','.join(str(e) for e in new_row)
            row_str += ","
            new_txt_file.write(row_str)

    def find(self,img):
        file = open(file, 'r').read()
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1,1))
        img = clahe.apply(img)
        cv2.imshow('c',img)



def real_strip(string):
        discount_chars = ["'", '"']
        string = string.strip()
        for char in discount_chars:
            if(string[0] == char and string[-1] == char):
                string = string[1:-1]
                break
        return string

def main():
    print(real_strip('"testsdgheh"'))
    #data_handler = data_preprocessor_handler()
   # data_handler.image_dir_to_matrix_txt("test-_imgs")
   # data_handler.normalise_text_file("digits.txt")
    #data_handler.format("train.txt")
   # data_handler.show_eq("2,2.png")
   # cv2.waitKey(0)
    #cv2.destroyWindows()
main()