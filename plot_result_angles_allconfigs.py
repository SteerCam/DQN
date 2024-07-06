import matplotlib.pyplot as plt2

import json
import os
import glob

# Specify the file path where the dictionary is saved

def plot_initial_angle_vs_acu_occu_rl_baseline():#dic_action,iteration_number):

   # Specify the folder path where your text files are located
   folder_path = './angle_var/'
   type_im = "./model1initAngle_300000"
   # Use glob to get a list of all .txt files in the folder
   txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
   fig, axs = plt2.subplots(1, 3, figsize=(12, 10))
   # Loop through the list of text files and load their contents
   aa=0
   cluster=[]
   for k in range(10,20,10):
      for l in range(0,10,10):
         for m in range(10,100,10):
            name = "./angle_var/"+str(k)+"_"+str(l)+"_"+str(m)+".txt"
            cluster.append(name)
   #"./angle_var/10_50_30.txt"
   #print(cluster)
   for file_path in cluster:
      #print(file_path)
      #if aa==2:
      #   break
      with open(file_path, 'r') as file:
           loaded_dict_base = json.load(file)
           # Do something with the content, e.g., print it
           dict_items = list(loaded_dict_base.items())
           lenth = len(dict_items)
           if lenth < 499:
            continue
           print(f"File: {file_path}")
           cam1=[i[0] for i in list(loaded_dict_base.values())]
           cam2=[i[1] for i in list(loaded_dict_base.values())]
           cam3=[i[2] for i in list(loaded_dict_base.values())]
           frame_number=list(loaded_dict_base.keys())
           parts = file_path.split("/")
           filename = parts[-1]
           if filename.endswith(".txt"):
             base_name = filename[:-4]
             lable = base_name
           #print(data)
           #print(lable)
           # Add a subplot for each camera
           #for j, camera_data in enumerate([cam1, cam2, cam3]):
           #print(cam3)
           axs[0].plot(frame_number, cam1)
           axs[1].plot(frame_number, cam2)
           axs[2].plot(frame_number, cam3)
            #axs[0, j].set_title(f'{label} - Camera {j+1}')
      aa+=1
   plt2.xlabel("X-axis")
   plt2.ylabel("Y-axis")
   plt2.title( "Angle Variation per frame")
   plt2.legend()
   # Save the plot as an image file
   plt2.savefig(str(type_im)+".png")
   plt2.clf()
   #break

#     # Plot all three sets of data in the same graph
#     #plt.bar(x, y)
#     #data=list(loaded_dict_base.values())
#     #this is the lets for AVG F1score values for cameras
#     lenth=50
#     base_acc_camA = [row[0] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
#     RL_acc_camA = [row[1] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
#     base_acc_camB = [row[0] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
#     RL_acc_camB = [row[1] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
#     base_acc_camC = [row[0] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
#     RL_acc_camC = [row[1] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
#     x_value_indi = list(loaded_dict_Avg_f1socre_CamA.keys())[:lenth]


#     data_suppix="_comAI"
#     iteration_number=300000
#     yescomai_nosteer=[]
#     nocomai_yessteer=[]
#     base_acc = [row[0] for row in loaded_dict_base.values()]
#     yescomai_yessteer_acc = [row[0] for row in loaded_dict_YescomAI_YesSteer.values()]
#     yescomaiall_yessteer_occ = [row[0] for row in loaded_dict_YescomAIALL_YesSteer.values()]
#     print(len(yescomaiall_yessteer_occ))
#     for row in loaded_dict_YescomAIALL_YesSteer.keys():
#         yescomai_nosteer.append(loaded_dict_YescomAI_NoSteer[row])
#         #print(loaded_dict_nocomAI_YesSteer[row])
#         nocomai_yessteer.append(loaded_dict_nocomAI_YesSteer[row])
#         #print(loaded_dict_nocomAI_YesSteer[row])
#         #print()
    
#     AB = [(row[0]/50)*100 for row in loaded_dict_collabPresantage.values()][:lenth]
#     BC = [(row[1]/50)*100 for row in loaded_dict_collabPresantage.values()][:lenth]
#     AC = [(row[2]/50)*100 for row in loaded_dict_collabPresantage.values()][:lenth]
#     x_value= list(loaded_dict_collabPresantage.keys())[:lenth]

#     global_detection_number_base = [row[0] for row in loaded_dict_global_detection_number.values()]
#     global_detection_number_dqn = [row[1] for row in loaded_dict_global_detection_number.values()]
    

#     plt5.figure(figsize=(100, 20)) 
#     plt5.margins(x=0)
#     plt5.plot(list(loaded_dict_global_detection_number.keys()), global_detection_number_base, label="baseline")
#     plt5.plot(list(loaded_dict_global_detection_number.keys()), global_detection_number_dqn, label="dqn ")

#     plt5.xlabel("angles")
    
# #    plt5.ylabel("total number of global detections as a percentage of people", fontsize=18)
# #    plt5.title("Initial Angle")
# #    plt5.yticks(fontsize=16)
# #    plt5.legend(fontsize=16)
# #    plt5.xticks(list(loaded_dict_global_detection_number.keys()),fontsize=16)
# #    plt5.xticks(rotation=90)
# #    # Save the plot as an image file
# #    plt5.savefig("./model1initAngle_vs_rl_baseline_total"+data_suppix+str(iteration_number)+".png")
# #    plt5.clf()

#     #base_cover = [row[4] for row in data]
#     #dqn_cover = [row[5] for row in data]
# #    plt5.figure(figsize=(50, 20)) 
# #    plt5.margins(x=0)
# #    lenth=100
# #    plt5.scatter(list(loaded_dict_base.keys())[:lenth], base_acc[:lenth], label="base_acc ")
# #    plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_yessteer_acc[:lenth], label="yescomai_yessteer_acc ")
# #    plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomaiall_yessteer_occ[:lenth], label="yescomaiall_yessteer_occ ")
# #    plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_nosteer[:lenth], label="yescomai_nosteer ")
# #    plt5.scatter(list(loaded_dict_base.keys())[:lenth], nocomai_yessteer[:lenth], label="nocomai_yessteer ")
# #    plt5.xlabel("Initial Angles",fontsize=32)
# #    plt5.ylabel("Average F1Sore Values",fontsize=32)
# #    #plt5.title("Initial Angle")
# #    plt5.yticks(fontsize=25)
# #    plt5.legend(fontsize=28)
# #    plt5.xticks(list(loaded_dict_base.keys())[:lenth],fontsize=20)
# #    plt5.xticks(rotation=90)
# #    # Save the plot as an image file
# #    plt5.savefig("./model1initAngle_vs_rl_baseline_acc"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
# #    plt5.clf()
# ################## below code is the bar plot  for the collaboration invocation 
#     # Set the width of the bars
#     collaboration_plot_en=0
#     if(collaboration_plot_en ==1) :
#       bar_width = 0.3
#       # Set the positions of the bars on the x-axis
#       index = range(len(x_value))
#       plt5.figure(figsize=(50, 20)) 
#       plt5.margins(x=0)
#       #plt5.plot(list(dic_action.keys()), base_occ, label="base_occ ")
#       #plt5.plot(list(dic_action.keys()), dqn_occ, label="dqn_occ ")
#       plt5.bar([i - bar_width for i in index], AB, bar_width, label='AB')
#       plt5.bar(index, BC, bar_width, label='BC')
#       plt5.bar([i + bar_width for i in index], AC, bar_width, label='AC')

      

#       plt5.xlabel("Initial Angles",fontsize=32)
#       plt5.ylabel("Percentage of Collaboration Invocation,",fontsize=32)
#       plt5.title("Initial Angle")
#       plt5.legend()
#       #plt5.xticks(list(loaded_dict_base.keys()))
#       plt5.xticks(index, x_value)
#       plt5.xticks(rotation=90,fontsize=22)
#       plt5.yticks(fontsize=25)
#       plt5.legend(fontsize=28)
#       plt5.savefig("./model1initAngle_vs_rl_baseline_occ"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
#       plt5.clf()
# ################## below code is the bar plot  for the average F1score for Individual cameras
#     individual_f1score_plot_en=1
#     if(individual_f1score_plot_en ==1) :
#       # Set the width of the bars
#       bar_width = 1
#       # Set the positions of the bars on the x-axis
#       index=[]
#       for j in range(len(x_value_indi)):
#          index.append(j*4)
#       #index = range(len(x_value_indi))
#       plt5.figure(figsize=(50, 20)) 
#       plt5.margins(x=0)
#       #plt5.plot(list(dic_action.keys()), base_occ, label="base_occ ")
#       #plt5.plot(list(dic_action.keys()), dqn_occ, label="dqn_occ ")

#       plt5.bar([i - bar_width for i in index], RL_acc_camA, bar_width, label='RL_acc_camA')
#       plt5.bar(index, RL_acc_camB, bar_width, label='RL_acc_camB')
#       plt5.bar([i + bar_width for i in index], RL_acc_camC, bar_width, label='RL_acc_camC')

#       plt5.bar([i - bar_width for i in index], base_acc_camA, bar_width, label='base_acc_camA')
#       plt5.bar(index, base_acc_camB, bar_width, label='base_acc_camB')
#       plt5.bar([i + bar_width for i in index], base_acc_camC, bar_width, label='base_acc_camC')


#       plt5.xlabel("Initial Angles",fontsize=32)
#       plt5.ylabel("Percentage of Collaboration Invocation,",fontsize=32)
#       plt5.title("Initial Angle")
#       plt5.legend()
#       #plt5.xticks(list(loaded_dict_base.keys()))
#       plt5.xticks(index, x_value_indi)
#       plt5.xticks(rotation=90,fontsize=22)
#       plt5.yticks(fontsize=25)
#       plt5.legend(fontsize=28)
#       plt5.savefig("./model1CamABC_individual_accuracy_comparison"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
#       plt5.clf()

   return 0 
plot_initial_angle_vs_acu_occu_rl_baseline()