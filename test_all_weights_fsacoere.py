import cv2
import glob
import numpy as np
import os


def plot_final_plot():
    # Set the path to the folder containing the images
    folder_path = './plots/'
    # Define the name pattern to filter the images
    image_pattern = "accuracy_per_frame_"
    image_pattern1 = "base_occlusion_"
    
    image_pattern2 = "angle_variation_"
    image_pattern3 = "avg_fscore_perFrame_"
    image_pattern4 = "DQN_occlusion_"

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter the files to only include those with the specified naming pattern
    dic_alldata_by_index={}
    selected_files1 = [file for file in files if file.startswith(image_pattern)]
    selected_files2 = [file for file in files if file.startswith(image_pattern1)]
    selected_files3 = [file for file in files if file.startswith(image_pattern2)]
    selected_files5 = [file for file in files if file.startswith(image_pattern3)]
    selected_files6 = [file for file in files if file.startswith(image_pattern4)]
    selected_files4=[]
    for ii in selected_files3:
        selected_files4.append(ii.split("_")[-1].split(".")[0])
        
    for ii in selected_files1:
        index=ii.split("_")[-1].split(".")[0]
        dic_alldata_by_index[index]=[ii]
    for ii in selected_files2:
        index=ii.split("_")[-1].split(".")[0]
        new_list=dic_alldata_by_index[index]
        new_list.append(ii)
        dic_alldata_by_index[index]=new_list

    for ii in selected_files3:
        index=ii.split("_")[-1].split(".")[0]
        
        new_list=dic_alldata_by_index[index]
       
        new_list.append(ii)
        
        dic_alldata_by_index[index]=new_list

    for ii in selected_files5:
        index=ii.split("_")[-1].split(".")[0]
        new_list=dic_alldata_by_index[index]
        new_list.append(ii)
        dic_alldata_by_index[index]=new_list
    for ii in selected_files6:
        index=ii.split("_")[-1].split(".")[0]
        new_list=dic_alldata_by_index[index]
        new_list.append(ii)
        dic_alldata_by_index[index]=new_list
    #print(dic_alldata_by_index)
    canvas_height = 0
    canvas_width = 0
    current_height = 0
    for i in selected_files4:
        canvas_height = 0
        canvas_width = 0
        current_height = 0
        im_name1 = dic_alldata_by_index[i][0]
        im_name2 = dic_alldata_by_index[i][1]
        im_name3 = dic_alldata_by_index[i][2]
        im_name4 = dic_alldata_by_index[i][3]
        im_name5 = dic_alldata_by_index[i][4]
        iteration=i
        print(im_name1, im_name2, im_name3,im_name4)
        
        # Read the images using OpenCV
        image1 = cv2.imread(folder_path + im_name1)
        image2 = cv2.imread(folder_path + im_name2)
        image3 = cv2.imread(folder_path + im_name3)
        image4 = cv2.imread(folder_path + im_name4)
        image5 = cv2.imread(folder_path + im_name5)
        print(image1.shape, image2.shape, image3.shape)

        # Update the canvas dimensions based on the maximum image width
        canvas_height += max(image1.shape[0], image2.shape[0], image3.shape[0])
        canvas_height += max(image1.shape[0], image2.shape[0], image3.shape[0])
        canvas_height += max(image1.shape[0], image2.shape[0], image3.shape[0])
        canvas_height += max(image1.shape[0], image2.shape[0], image3.shape[0])
        canvas_height += max(image1.shape[0], image2.shape[0], image3.shape[0])
        canvas_width = max(canvas_width, image1.shape[1], image2.shape[1], image3.shape[1])

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place each image on the canvas in separate rows
        canvas[current_height:current_height+image1.shape[0], :image1.shape[1]] = image1
        current_height += image1.shape[0]
        
        canvas[current_height:current_height+image2.shape[0], :image2.shape[1]] = image2
        
        current_height += image2.shape[0]
        canvas[current_height:current_height+image5.shape[0], :image5.shape[1]] = image5
        current_height += image5.shape[0]
        
        
        canvas[current_height:current_height+image3.shape[0], :image3.shape[1]] = image3
        current_height += image3.shape[0]

        canvas[current_height:current_height+image4.shape[0], :image4.shape[1]] = image4
        current_height += image4.shape[0]
        
        # Save the canvas with all the images
        cv2.imwrite("./dump/ANALYSIS_per_frame_"+str(iteration)+".jpg", canvas)
        
        print("done")

plot_final_plot()

