import matplotlib.pyplot as plt5

import json

# Specify the file path where the dictionary is saved

def plot_initial_angle_vs_acu_occu_rl_baseline():#dic_action,iteration_number):
   file_path = "./final_all_combinations_accuracy_and_occlusion.txt"
   with open(file_path, "r") as file:
      loaded_dict_base = json.load(file)

   #print(len(list(loaded_dict_base.keys())))
   #print(len(list(loaded_dict_YescomAI_YesSteer.keys())))
   #print(len(list(loaded_dict_nocomAI_YesSteer.keys())))
   data_suppix="_comAI"
   iteration_number=300000

   baseline = [row[2] for row in loaded_dict_base.values()]
   dqn = [row[3] for row in loaded_dict_base.values()]
   individual_f1score_plot_en=1
   x_value_indi = list(loaded_dict_base.keys())
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
     print(baseline)
     plt5.bar(index, dqn, bar_width, label='Dqn Occulusion',color='y')
     plt5.bar(index, baseline, bar_width, label='baseline Occlusion')
     


     plt5.xlabel("Initial Angles",fontsize=32)
     plt5.ylabel("Avegae Occlusion in three view,",fontsize=32)
     plt5.title("Initial Angle")
     plt5.legend()
     #plt5.xticks(list(loaded_dict_base.keys()))
     plt5.xticks(index, x_value_indi)
     plt5.xticks(rotation=90,fontsize=22)
     plt5.yticks(fontsize=25)
     plt5.legend(fontsize=28)
     plt5.savefig("./model1Occlusion"+data_suppix+str(iteration_number)+".png", bbox_inches='tight')
     plt5.clf()
    #print(loaded_dict_nocomAI_YesSteer[row])
    #print()
   return 0 
plot_initial_angle_vs_acu_occu_rl_baseline()