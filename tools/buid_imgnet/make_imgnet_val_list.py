import sys
from os import listdir
from os.path import isfile, join
if __name__ == "__main__":
	print("Making imagenet list file for images")
	src_folder = sys.argv[1]
	list_file = sys.argv[2]
	dst_file = sys.argv[3]
	
	img_files = [f for f in listdir(src_folder) if isfile(join(src_folder, f))]
	print(len(img_files))
	print(img_files[0])
	f_out = open(dst_file, "w")
	f_val = open(list_file, "r")
	i = 0
	with open(list_file, 'r') as f:
                for line in f:
                        label = line.split()
                        img_file = img_files[i]
			f_out.write(img_file + " " + str(label[0]) + "\n")
			i += 1
	f_out.close()
	
