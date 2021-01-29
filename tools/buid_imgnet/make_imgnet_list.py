import sys
from os import listdir
from os.path import isfile, join
if __name__ == "__main__":
	print("Making imagenet list file for images")
	src_folder = sys.argv[1]
	list_file = sys.argv[2]
	dst_file = sys.argv[3]
	files = {}
	with open(list_file, 'r') as f:
		for line in f:
			filepath, label = line.split()
			files[filepath] = int(label)

	print("Number of classes: ", len(files))
	
	img_files = [f for f in listdir(src_folder) if isfile(join(src_folder, f))]
	print(len(img_files)) 
	print(img_files[0])
	f_out = open(dst_file, "w")
	for img_file in img_files:
		class_name = img_file.split('_')[0]
		#print(class_name)
		if class_name in files:
			class_label = files[class_name]
		#print(class_label)
		f_out.write(img_file + " " + str(class_label) + "\n")
	f_out.close()
	
