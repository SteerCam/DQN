import matplotlib.pyplot as plt5

import json

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

    global_detection_number_dqn_steering = [row[1] for row in loaded_dict_global_steeringdetection_number.values()]
    #global_detection_number_dqn_comAI = [row[1] for row in loaded_dict_global_comAIdetection_number.values()]


   #global_detection_number_dqn_steering = [row[1] for row in loaded_dict_global_steeringdetection_number.values()]
   # global_detection_number_dqn_comAI = [row[1] for row in loaded_dict_global_comAIdetection_number.values()]
    shapes = ['o', 's', '^', 'D', 'x']
    print('aa',len(global_detection_number_base))
    print('bb',len(global_detection_number_dqn))
    print('cc',len(global_detection_number_dqn_steering))
    print('dd',len(global_detection_number_dqn_comAI))
    plt5.figure(figsize=(40, 20)) 
    plt5.margins(x=0)
    plt5.scatter(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_base, color='red', marker=shapes[0],label="Static",s=150)
    plt5.plot(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_base, color='red', linestyle='--')


    plt5.scatter(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn_steering, color='purple',marker="x", label="SteerCam-S",s=150)
    plt5.plot(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn_steering, color='purple', linestyle='--')

    plt5.scatter(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn, color='black',marker=shapes[2], label="SteerCam",s=150)
    plt5.plot(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn, color='black', linestyle='--')

    plt5.scatter(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn_comAI, color='green', marker=shapes[3],label="SteerCam-C",s=150)
    plt5.plot(list(loaded_dict_global_steeringdetection_number.keys()), global_detection_number_dqn_comAI, color='green', linestyle='--')
    plt5.xlabel("Tuple of Initial Angles of Three Cameras",fontsize=72)

    # Set custom x-tick positions and labels
    #custom_xticks = [1, 2.5, 3.5, 4]
    #custom_xtick_labels = ['A', 'B', 'C', 'D']
    #plt.xticks(custom_xticks, custom_xtick_labels)
    
    plt5.ylabel("The percentage of global TP", fontsize=72)
    plt5.yticks(fontsize=48)
    #plt5.legend(fontsize=48)
    plt5.legend(fontsize=72)
    plt5.xticks(fontsize=5,rotation=90)
    #plt5.xticks()
    # Save the plot as an image file
    plt5.tight_layout()
    plt5.savefig("./model1initAngle_vs_rl_baseline_globaltotal"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
    plt5.clf()
    return 0 
plot_initial_angle_vs_acu_occu_rl_baseline()