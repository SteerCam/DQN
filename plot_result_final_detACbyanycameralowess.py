import matplotlib.pyplot as plt5

import json
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np

# Specify the file path where the dictionary is saved

def plot_initial_angle_vs_acu_occu_rl_baseline():#dic_action,iteration_number):
    file_path = "./global_detection_numberfmap.txt" #global_detection_numberfmap  global_detection_number
    with open(file_path, "r") as file:
       loaded_dict_global_detection_number = json.load(file)

    file_path = "./global_(onlysteering)detection_number.txt"
    with open(file_path, "r") as file:
       loaded_dict_global_steeringdetection_number = json.load(file)
    file_path = "./global_detection_numberfmap_colabonly.txt"
    with open(file_path, "r") as file:
       loaded_dict_global_comAIdetection_number = json.load(file) 
    aa=[]
    for row in loaded_dict_global_detection_number.keys():
      aa.append(loaded_dict_global_detection_number[row][1])
    above=0
    for i in aa:
       if(i>=0.8):
          above+=1
    percentage =above/1000
    print(percentage)
    # Plot all three sets of data in the same graph
    #plt.bar(x, y)
    #data=list(loaded_dict_base.values())
    data_suppix="_comAI"
    iteration_number=300000
    global_detection_number_dqn_steering=[]
    global_detection_number_dqn_comAI=[]
    global_detection_number_dqn=[]
    global_detection_number_base=[]
    plt5.tight_layout()

   # tis code is fro when all dics has 100
   # #print(len(yescomaiall_yessteer_occ))
   # for row in loaded_dict_global_steeringdetection_number.keys():
   #     #print(loaded_dict_global_steeringdetection_number)
   #     #print(loaded_dict_global_comAIdetection_number)
   #     global_detection_number_dqn_steering.append(loaded_dict_global_steeringdetection_number[row][1])
   #     #print(loaded_dict_nocomAI_YesSteer[row])
   #     global_detection_number_dqn_comAI.append(loaded_dict_global_comAIdetection_number[row][1])
   #     #print(loaded_dict_nocomAI_YesSteer[row])
   #     #print()
   # global_detection_number_base = [row[0] for row in loaded_dict_global_detection_number.values()]
   # global_detection_number_dqn = [row[1] for row in loaded_dict_global_detection_number.values()]

    #print(len(yescomaiall_yessteer_occ))
    diff_S=[]
    diff_C=[]
    diff_static=[]
    import statistics
    for row in loaded_dict_global_steeringdetection_number.keys():
        #print(loaded_dict_global_steeringdetection_number)
        #print(loaded_dict_global_comAIdetection_number)
        #global_detection_number_dqn_steering.append(loaded_dict_global_steeringdetection_number[row][1])
        ##print(loaded_dict_nocomAI_YesSteer[row])
        #global_detection_number_dqn_comAI.append(loaded_dict_global_comAIdetection_number[row][1])
        #print(loaded_dict_nocomAI_YesSteer[row])
        #print()
        global_detection_number_base.append(loaded_dict_global_detection_number[row][0])
        global_detection_number_dqn.append(loaded_dict_global_detection_number[row][1])
        global_detection_number_dqn_comAI.append(loaded_dict_global_comAIdetection_number[row][1])
        diff_S.append(loaded_dict_global_detection_number[row][1]-loaded_dict_global_steeringdetection_number[row][1])
        diff_C.append(loaded_dict_global_detection_number[row][1]-loaded_dict_global_comAIdetection_number[row][1])
        diff_static.append(loaded_dict_global_detection_number[row][1]-loaded_dict_global_detection_number[row][0])

    print(max(diff_S),max(diff_C),max(global_detection_number_base))
    mean_s = statistics.mean(diff_S)
    std_s = statistics.stdev(diff_S)
    mean_C = statistics.mean(diff_C)
    std_C = statistics.stdev(diff_C)
    mean_stat = statistics.mean(diff_static)
    std_stat = statistics.stdev(diff_static)
    print(mean_s,std_s,mean_C,std_C,mean_stat,std_stat)
    global_detection_number_dqn_steering = [row[1] for row in loaded_dict_global_steeringdetection_number.values()]
    #global_detection_number_dqn_comAI = [row[1] for row in loaded_dict_global_comAIdetection_number.values()]

    plt5.figure(figsize=(8,6))#40, 20
   #global_detection_number_dqn_steering = [row[1] for row in loaded_dict_global_steeringdetection_number.values()]
   # global_detection_number_dqn_comAI = [row[1] for row in loaded_dict_global_comAIdetection_number.values()]
    shapes = ['o', 's', '^', 'D', 'x']
    print('aa',len(global_detection_number_base))
    print('bb',len(global_detection_number_dqn))
    print('cc',len(global_detection_number_dqn_steering))
    print('dd',len(global_detection_number_dqn_comAI))
    # Calculate Lowess smoothed values
    x = np.linspace(0, 1, len(global_detection_number_dqn_comAI))
    x1 = np.linspace(1,len(global_detection_number_dqn_comAI) ,len(global_detection_number_dqn_comAI))
    f1_smooth1 = lowess(global_detection_number_base, x, frac=0.05)[:, 1]  # adjust frac for smoothing strength
    f1_smooth2 = lowess(global_detection_number_dqn_steering, x, frac=0.05)[:, 1]
    f1_smooth3 = lowess(global_detection_number_dqn, x, frac=0.05)[:, 1]
    f1_smooth4 = lowess(global_detection_number_dqn_comAI, x, frac=0.05)[:, 1]

     
    plt5.margins(x=0)
    xt = np.linspace(0, len(global_detection_number_dqn_comAI)-1, 20)
    new_x=[]
    xx_int=[]
    for jj in xt:
       new_x.append(list(loaded_dict_global_steeringdetection_number.keys())[int(jj)])
       xx_int.append(int(jj))
    #print(x1)
    plt5.xticks(xx_int,new_x)
    plt5.scatter(x1, global_detection_number_base, color='red', marker=shapes[0],label="Static",s=5)
    plt5.plot(x1, f1_smooth1, color='red',linewidth=2)


    plt5.scatter(x1, global_detection_number_dqn_steering, color='purple',marker="x", label="SteerCam-S",s=5)
    plt5.plot(x1, f1_smooth2, color='purple', linewidth=2)

    plt5.scatter(x1, global_detection_number_dqn, color='black',marker=shapes[2], label="SteerCam",s=5)
    plt5.plot(x1, f1_smooth3, color='black', linewidth=2)

    plt5.scatter(x1, global_detection_number_dqn_comAI, color='green', marker=shapes[3],label="SteerCam-C",s=5)
    plt5.plot(x1, f1_smooth4, color='green', linewidth=2)
    plt5.xlabel("Tuple of Initial Angles of Three Cameras",fontsize=16)

    # Set custom x-tick positions and labels
    #custom_xticks = [1, 2.5, 3.5, 4]
    #custom_xtick_labels = ['A', 'B', 'C', 'D']
    #plt.xticks(custom_xticks, custom_xtick_labels)
    plt5.legend(loc='upper left')
    plt5.ylabel("The percentage of global TP", fontsize=24)
    plt5.yticks(fontsize=16)
    #plt5.legend(fontsize=48)
    plt5.legend(fontsize=18)
    plt5.xticks(fontsize=12,rotation=90)
    
    #plt5.xticks()
    # Save the plot as an image file
    plt5.tight_layout()
    #plt5.savefig("./model1initAngle_vs_rl_baseline_globaltotal"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
    plt5.clf()
    return 0 

print("sadsadsadas")
plot_initial_angle_vs_acu_occu_rl_baseline()