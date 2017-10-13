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
        matrix_width = 22
        to_retreive = 5
        data_set = open(file, 'r').read().split(",")
        matrices = []
        targets = []
        px_count = 0
        print(int(data_set[1569]))
        for i in range(to_retreive):
            matrix = np.zeros((matrix_width,matrix_width), dtype=np.uint8)
            for px_col in range(matrix_width):
                for px_row in range(matrix_width):
                    if(px_count%((matrix_width*matrix_width)+1)==0):
                        targets.append(int(data_set[px_count]))
                    else:
                        matrix[px_col][px_row] = float(data_set[px_count])
                    px_count += 1
            matrices.append(matrix)
        print(targets)


        for i in range(0,len(matrices)):
            cv2.imshow(str(i)+","+str(targets[i]),matrices[i]) 


    def format(self,file):
        new_txt_file = open("voice_new.txt", "a")
        file = open(file, 'r').read()
        for i in range(0,len(file)):
            w = ""
            if(file[i] == "\r" or file[i] == ","):
                w = ","
            elif(file[i].isdigit() == True or file[i] == "."):
                w = file[i]
            new_txt_file.write(w)

    def see_chars(self,file):
        file = open(file, 'r').read().replace("\r\n", ",",1).replace(" ", "",1).split(",")
        for i in range(0,len(file)):
        	print(file[i])

def main():
    data_handler = data_preprocessor_handler()
   # data_handler.image_dir_to_matrix_txt("test_imgs")
   # data_handler.show_matrices_from_file("test_imgs.txt")
    data_handler.format("voice.txt")
   # cv2.waitKey(0)
    #cv2.destroyWindows()
main()