import os
import shutil

#number_images = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
number_images = [2000, 3000, 4000]
root_dir = "../data/TimeConsumption/"
original_image = "../data/SinglePolypImage/028.jpg"
for num in number_images:
    target_directory = os.path.join(root_dir, str(num))
    try:
        os.mkdir(target_directory)
    except OSError:
        print("{} target directory already exists.".format(target_directory))
    for i in range(num):
        file_name = original_image.split("/")[-1]
        file_name, extension = file_name.split(".")
        target_file_name = "{}_{}.{}".format(file_name, i, extension)
        target_file_name = os.path.join(target_directory, target_file_name)
        shutil.copyfile(original_image, target_file_name)