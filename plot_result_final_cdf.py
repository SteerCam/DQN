import matplotlib.pyplot as plt5

import json
import random
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
# Specify the file path where the dictionary is saved

def plot_initial_angle_vs_acu_occu_rl_baseline():#dic_action,iteration_number):
    file_path = "./base_accuracy.txt"
    with open(file_path, "r") as file:
       loaded_dict_base = json.load(file)

    file_path = "./Steer(Yes)_comAI(yes)_accuracy(ours).txt"
    with open(file_path, "r") as file:
       loaded_dict_YescomAI_YesSteer = json.load(file)

    file_path = "./Steer(Yes)_comAI(No)_accuracy(ours).txt"
    with open(file_path, "r") as file:
       loaded_dict_nocomAI_YesSteer = json.load(file)

    file_path = "./Steer(yes)_comAI(yesALL)_accuracy.txt"
    with open(file_path, "r") as file:
       loaded_dict_YescomAIALL_YesSteer = json.load(file)

    file_path = "./Steer(No)_comAI(yes)_accuracy.txt"
    with open(file_path, "r") as file:
       loaded_dict_YescomAI_NoSteer = json.load(file)
       
    file_path = "./percentageOfcollaboration.txt"
    with open(file_path, "r") as file:
       loaded_dict_collabPresantage = json.load(file)

    file_path = "./global_detection_number.txt"
    with open(file_path, "r") as file:
       loaded_dict_global_detection_number = json.load(file)
    
    file_path = "./global_cameraA_f1score_comparison.txt"
    with open(file_path, "r") as file:
       loaded_dict_Avg_f1socre_CamA = json.load(file)
    file_path = "./global_cameraB_f1score_comparison.txt"
    with open(file_path, "r") as file:
       loaded_dict_Avg_f1socre_CamB = json.load(file)
    file_path = "./global_cameraC_f1score_comparison.txt"
    with open(file_path, "r") as file:
       loaded_dict_Avg_f1socre_CamC = json.load(file)



    # Plot all three sets of data in the same graph
    #plt.bar(x, y)
    #data=list(loaded_dict_base.values())
    #this is the lets for AVG F1score values for cameras
    lenth=100
    num_indices = 1000
    random.seed(34)

    #base_acc_camA = [row[0] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
    #RL_acc_camA = [row[1] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
    #base_acc_camB = [row[0] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
    #RL_acc_camB = [row[1] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
    #base_acc_camC = [row[0] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
    #RL_acc_camC = [row[1] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
    #x_value_indi = list(loaded_dict_Avg_f1socre_CamA.keys())[:lenth]

    base_acc_camA =random.sample( [row[0] for row in loaded_dict_Avg_f1socre_CamA.values()], num_indices)
    RL_acc_camA   =random.sample( [row[1] for row in loaded_dict_Avg_f1socre_CamA.values()], num_indices)
    base_acc_camB =random.sample( [row[0] for row in loaded_dict_Avg_f1socre_CamB.values()], num_indices)
    RL_acc_camB   =random.sample( [row[1] for row in loaded_dict_Avg_f1socre_CamB.values()], num_indices)
    base_acc_camC =random.sample( [row[0] for row in loaded_dict_Avg_f1socre_CamC.values()], num_indices)
    RL_acc_camC   =random.sample( [row[1] for row in loaded_dict_Avg_f1socre_CamC.values()], num_indices)

    

    data_suppix="_comAI"
    iteration_number=300000
    yescomai_nosteer=[]
    nocomai_yessteer=[]

    base_acc = [row[0] for row in loaded_dict_base.values()]
    yescomai_yessteer_acc = [row[0] for row in loaded_dict_YescomAI_YesSteer.values()]
    yescomaiall_yessteer_occ = [row[0] for row in loaded_dict_YescomAIALL_YesSteer.values()]


    print(len(yescomaiall_yessteer_occ))
    for row in loaded_dict_YescomAIALL_YesSteer.keys():
        yescomai_nosteer.append(loaded_dict_YescomAI_NoSteer[row])
        #print(loaded_dict_nocomAI_YesSteer[row])
        nocomai_yessteer.append(loaded_dict_nocomAI_YesSteer[row])
        #print(loaded_dict_nocomAI_YesSteer[row])
        #print()

    a = random.sample(base_acc, num_indices)
    b = random.sample(yescomai_yessteer_acc, num_indices)
    c = random.sample(yescomaiall_yessteer_occ, num_indices)
    d = random.sample(yescomai_nosteer, num_indices)
    e = random.sample(nocomai_yessteer, num_indices)
    
    data_sorted1 = np.sort(a)
    data_sorted2 = np.sort(b)
    data_sorted3 = np.sort(c)
    data_sorted4 = np.sort(d)
    data_sorted5 = np.sort(e)
    int_list1 = [int(num) for num in data_sorted2]
    int_list2 = [int(num) for num in data_sorted3]
    print("ssss",data_sorted2[120],data_sorted3[120])
    statistic, pvalue = stats.ks_2samp(int_list1, int_list2)
    print("One-Sample K-S Test (data1 vs. uniform)")
    print("Statistic:", statistic)
    print("p-value:", pvalue)
    #print(data_sorted2)
    #print(data_sorted3)

    cdf = np.linspace(0, 1, len(data_sorted1))
    mu1, std1 = norm.fit(a)
    mu2, std2 = norm.fit(b)
    mu3, std3 = norm.fit(c)
    mu4, std4 = norm.fit(d)
    mu5, std5 = norm.fit(e)

    d_min= [min(data_sorted1),min(data_sorted2),min(data_sorted3),min(data_sorted4),min(data_sorted5)]
    d_max= [max(data_sorted1),max(data_sorted2),max(data_sorted3),max(data_sorted4),max(data_sorted5)]
    x = np.linspace(min(d_min), max(d_max), 1000)
    cdf_smooth1 = norm.cdf(x, mu1, std1)
    cdf_smooth2 = norm.cdf(x, mu2, std2)
    cdf_smooth3 = norm.cdf(x, mu3, std3)
    cdf_smooth4 = norm.cdf(x, mu4, std4)
    cdf_smooth5 = norm.cdf(x, mu5, std5)
    #print('maxxxxxxxxxxxxxxxxxxxxxxx',data_sorted1)
    

        # Plot the CDF
    plt5.figure(figsize=(8, 6))
    #plt.plot(data_sorted, cdf, marker='.', linestyle='none', label='Empirical CDF')
    plt5.plot(x, cdf_smooth1, label='Static', color='red' , linewidth=2)
    plt5.plot(x, cdf_smooth2, label='SteerCam', color='green' , linewidth=3, linestyle='--')
    plt5.plot(x, cdf_smooth3, label='SteerCam-GreedyC', color='blue' , linewidth=3, linestyle='dashdot')
    plt5.plot(x, cdf_smooth4, label='SteerCam-C', color='black', linewidth=3 , linestyle='dashed')
    plt5.plot(x, cdf_smooth5, label='SteerCam-S', color='purple', linewidth=3 , linestyle=':')  # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    
    #AB = [(row[0]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth]
    #BC = [(row[1]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth]
    #AC = [(row[2]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth]


    AB = random.sample([(row[0]/100)*100 for row in loaded_dict_collabPresantage.values()], num_indices)
    BC = random.sample([(row[1]/100)*100 for row in loaded_dict_collabPresantage.values()], num_indices)
    AC = random.sample([(row[2]/100)*100 for row in loaded_dict_collabPresantage.values()], num_indices)

    

    # Number of random indices you want to select
    
    
    # Select random indices
    random_indices = random.sample(list(loaded_dict_collabPresantage.keys()), num_indices)

    #x_value= list(loaded_dict_collabPresantage.keys())[:lenth]
    x_value=random_indices
    x_value_indi=random_indices

    #print(x_value)
    global_detection_number_base = [row[0] for row in loaded_dict_global_detection_number.values()]
    global_detection_number_dqn = [row[1] for row in loaded_dict_global_detection_number.values()]
    

    #plt5.figure(figsize=(100, 20)) 
    #plt5.margins(x=0)
    #plt5.plot(list(loaded_dict_global_detection_number.keys()), global_detection_number_base, label="baseline")
    #plt5.plot(list(loaded_dict_global_detection_number.keys()), global_detection_number_dqn, label="dqn ")

    #plt5.xlabel("angles")
#    
#    plt5.ylabel("total number of global detections as a percentage of people", fontsize=18)
#    plt5.title("Initial Angle")
#    plt5.yticks(fontsize=16)
#    plt5.legend(fontsize=16)
#    plt5.xticks(list(loaded_dict_global_detection_number.keys()),fontsize=16)
#    plt5.xticks(rotation=90)
#    # Save the plot as an image file
#    plt5.savefig("./model1initAngle_vs_rl_baseline_total"+data_suppix+str(iteration_number)+".png")
#    plt5.clf()

    #base_cover = [row[4] for row in data]
    #dqn_cover = [row[5] for row in data]
    #plt5.figure(figsize=(50, 20)) 
    #plt5.margins(x=0)
    #lenth=100
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], base_acc[:lenth], label="base_acc ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_yessteer_acc[:lenth], label="yescomai_yessteer_acc ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomaiall_yessteer_occ[:lenth], label="yescomaiall_yessteer_occ ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_nosteer[:lenth], label="yescomai_nosteer ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], nocomai_yessteer[:lenth], label="nocomai_yessteer ")

    plt5.xlabel('Average F1score Accuracy',fontsize=20)
    plt5.ylabel('CDF',fontsize=20)
    #plt5.title('Cumulative Distribution Function (CDF)',fontsize=20)
    plt5.legend(loc='upper right')
    plt5.yticks(fontsize=24)
    plt5.xticks(fontsize=24)
    plt5.legend(fontsize=16)
    plt5.grid(True)
    
    # Save the plot as an image file
    plt5.savefig("./model1initAngle_cdf"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
    plt5.clf()
################## below code is the bar plot  for the collaboration invocation 
    # Set the width of the bars
    collaboration_plot_en=0
    if(collaboration_plot_en ==1) :
      bar_width = 0.3
      # Set the positions of the bars on the x-axis
      index = range(len(x_value))
      print(index)
      plt5.figure(figsize=(50, 20)) 
      plt5.margins(x=0)
      #plt5.plot(list(dic_action.keys()), base_occ, label="base_occ ")
      #plt5.plot(list(dic_action.keys()), dqn_occ, label="dqn_occ ")
      plt5.bar([i - bar_width for i in index], AB, bar_width, label='AB')
      plt5.bar(index, BC, bar_width, label='BC')
      plt5.bar([i + bar_width for i in index], AC, bar_width, label='AC')

      

      plt5.xlabel("Initial Angles",fontsize=32)
      plt5.ylabel("Percentage of Collaboration Invocation,",fontsize=32)
      plt5.title("Initial Angle")
      plt5.legend()
      #plt5.xticks(list(loaded_dict_base.keys()))
      plt5.xticks(index, x_value)
      plt5.xticks(rotation=90,fontsize=22)
      plt5.yticks(fontsize=25)
      plt5.legend(fontsize=28)
      plt5.savefig("./model1init_collboration_"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
      plt5.clf()
################## below code is the bar plot  for the average F1score for Individual cameras
    individual_f1score_plot_en=1
    if(individual_f1score_plot_en ==1) :
      # Set the width of the bars
      bar_width = 1
      # Set the positions of the bars on the x-axis
      index=[]
      for j in range(len(x_value_indi)):
         index.append(j*4)
      #index = range(len(x_value_indi))
      plt5.figure(figsize=(50, 20)) 
      plt5.margins(x=0)
      #plt5.plot(list(dic_action.keys()), base_occ, label="base_occ ")
      #plt5.plot(list(dic_action.keys()), dqn_occ, label="dqn_occ ")

      plt5.bar([i - bar_width for i in index], RL_acc_camA, bar_width, label='RL_acc_camA')
      plt5.bar(index, RL_acc_camB, bar_width, label='RL_acc_camB')
      plt5.bar([i + bar_width for i in index], RL_acc_camC, bar_width, label='RL_acc_camC')

      plt5.bar([i - bar_width for i in index], base_acc_camA, bar_width, label='base_acc_camA')
      plt5.bar(index, base_acc_camB, bar_width, label='base_acc_camB')
      plt5.bar([i + bar_width for i in index], base_acc_camC, bar_width, label='base_acc_camC')


      plt5.xlabel("Initial Angles",fontsize=32)
      plt5.ylabel("F1 Score of Individual Camera",fontsize=32)
      plt5.title("Initial Angle")
      plt5.legend()
      #plt5.xticks(list(loaded_dict_base.keys()))
      plt5.xticks(index, x_value_indi)
      plt5.xticks(rotation=90,fontsize=22)
      plt5.yticks(fontsize=25)
      plt5.legend(fontsize=28)
      plt5.savefig("./model1CamABC_individual_accuracy_comparison"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
      plt5.clf()

    return 0 
plot_initial_angle_vs_acu_occu_rl_baseline()