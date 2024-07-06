import matplotlib.pyplot as plt5

import json

# Specify the file path where the dictionary is saved

def plot_initial_angle_vs_acu_occu_rl_baseline():#dic_action,iteration_number):
   file_path = "./base_accuracy.txt"
   with open(file_path, "r") as file:
      loaded_dict_base = json.load(file)
   file_path = "./Steer(yes)_comAI(yes)_accuracy.txt"
   with open(file_path, "r") as file:
      loaded_dict_YescomAI_YesSteer = json.load(file)
   file_path = "./Steer(Yes)_comAI(No)_accuracy(ours).txt"
   with open(file_path, "r") as file:
      loaded_dict_nocomAI_YesSteer = json.load(file)
   #print(len(list(loaded_dict_base.keys())))
   #print(len(list(loaded_dict_YescomAI_YesSteer.keys())))
   #print(len(list(loaded_dict_nocomAI_YesSteer.keys())))
   data_suppix="_comAI"
   iteration_number=300000
   yescomai_yesssteer=[]
   baseline=[]
   diff_list=[]
   for row in loaded_dict_nocomAI_YesSteer.keys():
    diff= loaded_dict_YescomAI_YesSteer[row][0]-loaded_dict_base[row][0]
    diff_list.append(diff)
    baseline.append(loaded_dict_base[row][0])
    #print(loaded_dict_nocomAI_YesSteer[row])
    yescomai_yesssteer.append(loaded_dict_YescomAI_YesSteer[row][0])
   nocomai_yesssteer = [row[0] for row in loaded_dict_nocomAI_YesSteer.values()]
   min_val=min(diff_list)
   max_value=max(diff_list)
   print("min_val,max_value",min_val,max_value)
   #print(baseline)
   #print(yescomai_yesssteer)
   #print(nocomai_yesssteer)
   individual_f1score_plot_en=1
   x_value_indi = list(loaded_dict_nocomAI_YesSteer.keys())
   if(individual_f1score_plot_en ==1) :
     # Set the width of the bars
     bar_width = 2
     # Set the positions of the bars on the x-axis
     index=[]
     for j in range(len(x_value_indi)):
        index.append(j*4)
     #index = range(len(x_value_indi))
     plt5.figure(figsize=(50, 20)) 
     plt5.margins(x=0)
     #plt5.plot(list(dic_action.keys()), base_occ, label="base_occ ")
     #plt5.plot(list(dic_action.keys()), dqn_occ, label="dqn_occ ")
     #print(baseline)
     
     plt5.bar(index, yescomai_yesssteer, bar_width, label='ours')
     plt5.bar(index, nocomai_yesssteer, bar_width, label='SteeringBenifit',color='y')
     plt5.bar(index, baseline, bar_width+0.2, label='Baseline',color='r')

     plt5.xlabel("Initial Angles",fontsize=32)
     plt5.ylabel("Collaboration and Steering Contribution to F1Score Accuracy Improvement,",fontsize=32)
     plt5.title("Initial Angle")
     plt5.legend()
     #plt5.xticks(list(loaded_dict_base.keys()))
     plt5.xticks(index, x_value_indi)
     plt5.xticks(rotation=90,fontsize=22)
     plt5.yticks(fontsize=25)
     plt5.legend(fontsize=28)
     plt5.savefig("./zzzmodel1colabVSsteeingBenifit"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
     plt5.clf()
    #print(loaded_dict_nocomAI_YesSteer[row])
    #print()
   return 0 
plot_initial_angle_vs_acu_occu_rl_baseline()