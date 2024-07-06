import matplotlib.pyplot as plt5

import json
import random
import statistics
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
    num_indices = 50
    random.seed(34)

    #base_acc_camA = [row[0] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
    #RL_acc_camA = [row[1] for row in loaded_dict_Avg_f1socre_CamA.values()][:lenth]
    #base_acc_camB = [row[0] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
    #RL_acc_camB = [row[1] for row in loaded_dict_Avg_f1socre_CamB.values()][:lenth]
    #base_acc_camC = [row[0] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
    #RL_acc_camC = [row[1] for row in loaded_dict_Avg_f1socre_CamC.values()][:lenth]
    #x_value_indi = list(loaded_dict_Avg_f1socre_CamA.keys())[:lenth]

    base_acc_camA =random.sample( [int(row[0]) for row in loaded_dict_Avg_f1socre_CamA.values()], num_indices)
    RL_acc_camA   =random.sample( [int(row[1]) for row in loaded_dict_Avg_f1socre_CamA.values()], num_indices)
    base_acc_camB =random.sample( [int(row[0]) for row in loaded_dict_Avg_f1socre_CamB.values()], num_indices)
    RL_acc_camB   =random.sample( [int(row[1]) for row in loaded_dict_Avg_f1socre_CamB.values()], num_indices)
    base_acc_camC =random.sample( [int(row[0]) for row in loaded_dict_Avg_f1socre_CamC.values()], num_indices)
    RL_acc_camC   =random.sample( [int(row[1]) for row in loaded_dict_Avg_f1socre_CamC.values()], num_indices)

    print("calculating the median")
    median_value_base_acc_camA = statistics.median(base_acc_camA)
    median_value_RL_acc_camA = statistics.median(RL_acc_camA)
    median_value_base_acc_camB = statistics.median(base_acc_camB)
    median_value_RL_acc_camB = statistics.median(RL_acc_camB)
    median_value_base_acc_camC = statistics.median(base_acc_camC)
    median_value_RL_acc_camC = statistics.median(RL_acc_camC)

    print(median_value_RL_acc_camA,median_value_base_acc_camA)
    print(median_value_RL_acc_camB,median_value_base_acc_camB)
    print(median_value_RL_acc_camC,median_value_base_acc_camC)

    
    data_suppix="_comAI"
    iteration_number=300000
    yescomai_nosteer=[]
    nocomai_yessteer=[]

    base_acc = [row[0] for row in loaded_dict_base.values()]
    yescomai_yessteer_acc = [row[0] for row in loaded_dict_YescomAI_YesSteer.values()]
    yescomaiall_yessteer_occ = [row[0] for row in loaded_dict_YescomAIALL_YesSteer.values()]


    print(len(yescomaiall_yessteer_occ))
    for row in loaded_dict_YescomAIALL_YesSteer.keys():
        yescomai_nosteer.append(loaded_dict_YescomAI_NoSteer[row][0])
        #print(loaded_dict_nocomAI_YesSteer[row])
        nocomai_yessteer.append(loaded_dict_nocomAI_YesSteer[row][0])
        #print(loaded_dict_nocomAI_YesSteer[row])
        #print()

    a = random.sample(base_acc, num_indices)
    b = random.sample(yescomai_yessteer_acc, num_indices)
    c = random.sample(yescomaiall_yessteer_occ, num_indices)
    d = random.sample(yescomai_nosteer, num_indices)
    e = random.sample(nocomai_yessteer, num_indices)
    ss=[]
    for ll in range(len(b)):
       df=b[ll]-a[ll]
       ss.append(df)
    print('ssssssss',min(ss),max(ss))

    lenth1=1000
 
    AB1 = [(row[0]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]
    BC1 = [(row[1]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]
    AC1 = [(row[2]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]

    #reduction
    AB11 =[(1-row[0]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]
    BC11=[(1-row[1]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]
    AC11=[(1-row[2]/100)*100 for row in loaded_dict_collabPresantage.values()][:lenth1]
    print(AB1[0],AB11[0])
    mean_s = statistics.mean(AB1)
    std_s = statistics.stdev(AB1)
    mean_C = statistics.mean(BC1)
    std_C = statistics.stdev(BC1)
    mean_stat = statistics.mean(AC1)
    std_stat = statistics.stdev(AC1)
    print(mean_s,std_s,mean_C,std_C,mean_stat,std_stat)

    mean_re_AB = statistics.mean(AB11)
    mean_re_BC = statistics.mean(BC11)
    mean_re_AC = statistics.mean(AC11)
    print("ggg",(mean_re_AB+mean_re_BC+mean_re_AC)/3)

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
    plt5.figure(figsize=(50, 20)) 
    plt5.margins(x=0)
    lenth=100
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], base_acc[:lenth], label="base_acc ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_yessteer_acc[:lenth], label="yescomai_yessteer_acc ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomaiall_yessteer_occ[:lenth], label="yescomaiall_yessteer_occ ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], yescomai_nosteer[:lenth], label="yescomai_nosteer ")
    #plt5.scatter(list(loaded_dict_base.keys())[:lenth], nocomai_yessteer[:lenth], label="nocomai_yessteer ")

    plt5.scatter(x_value, a, label="base_acc ", color='red',s=200)
    plt5.plot(x_value, a, color='red', linestyle='--', linewidth=10)
    #plt5.scatter(x_value, b, label="collaboration(yes)_steering(yes)", color='green',s=200)
    #plt5.plot(x_value, b, color='green', linestyle='--')
    #plt5.scatter(x_value, c, label="allcollaboration(yes)_steering(yes)", color='blue',s=200)
    #plt5.plot(x_value, c, color='blue', linestyle='--')
    #plt5.scatter(x_value, d, label="collaboration(yes)_steering(no)", color='black',s=200)
    #plt5.plot(x_value, d, color='black', linestyle='--')
    #plt5.scatter(x_value, e, label="collaboration(no)_steering(yes)", color='purple',s=200)
    #plt5.plot(x_value, e, color='purple', linestyle='--')
    
    #bar_width = 0.3
    ##index = range(len(x_value))
    ##print(len(a),len(b),len(c),len(d),len(e),len(index))
    #print(a)
    #plt5.bar(x_value, a, label="base_acc ", color='red')
    ##plt5.bar(index, a, bar_width, label='BC')
    ##plt5.bar(x_value, a, color='red', linestyle='--')
    #plt5.bar(x_value, b, label="collaboration(yes)_steering(yes)", color='green')
    ###plt5.bar(x_value, b, color='green', linestyle='--')
    #plt5.bar(x_value, c, label="allcollaboration(yes)_steering(yes)", color='blue')
    ###plt5.bar(x_value, c, color='blue', linestyle='--')
    #plt5.bar(x_value, d, label="collaboration(yes)_steering(no)", color='black')
    ##plt5.bar(x_value, d, color='black', linestyle='--')
    #plt5.bar(x_value, e, label="collaboration(no)_steering(yes)", color='purple')
    ##plt5.bar(x_value, e, color='purple', linestyle='--')

    plt5.xlabel("Tuple of Initial Angles of Three Cameras",fontsize=100)
    plt5.ylabel("Average F1Sore Values",fontsize=100)
    #plt5.title("Initial Angle")
    plt5.yticks(fontsize=48)
    plt5.legend(fontsize=48)
    #plt5.xticks(list(loaded_dict_base.keys())[:lenth],fontsize=20)
    plt5.xticks(x_value,fontsize=32)
    plt5.xticks(rotation=90)
    plt5.legend(fontsize=100)
    plt5.tight_layout()
    # Save the plot as an image file
    plt5.savefig("./model1initAngle_vs_rl_baxxxxseline_acc"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
    plt5.clf()
################## below code is the bar plot  for the collaboration invocation 
    # Set the width of the bars
    collaboration_plot_en=1
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

      

      plt5.xlabel("Tuple of Initial Angles of Three Cameras",fontsize=32)
      plt5.ylabel("Percentage of Collaboration Invocation,",fontsize=32)
      plt5.title("Tuple of Initial Angles of Three Cameras")
      
      #plt5.xticks(list(loaded_dict_base.keys()))
      plt5.xticks(index, x_value)
      plt5.xticks(rotation=90,fontsize=22)
      plt5.yticks(fontsize=25)
      plt5.legend(fontsize=28)
      plt5.tight_layout()
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


      plt5.xlabel("Tuple of Initial Angles of Three Cameras",fontsize=32)
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