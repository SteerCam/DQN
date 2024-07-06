import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.nonparametric.smoothers_lowess import lowess

file_path = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'
file_path2 = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'
file_path3 = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'
file_path4 = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'
file_path5 = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'
file_path6 = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'

data_dic={}
with open(file_path, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic[list(dict_a.keys())[0]]=list(dict_a.values())[0]

data_dic2={}
with open(file_path2, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic2[list(dict_a.keys())[0]]=list(dict_a.values())[0]

data_dic3={}
with open(file_path3, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic3[list(dict_a.keys())[0]]=list(dict_a.values())[0]

data_dic4={}
with open(file_path4, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic4[list(dict_a.keys())[0]]=list(dict_a.values())[0]

data_dic5={}
with open(file_path5, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic5[list(dict_a.keys())[0]]=list(dict_a.values())[0]


data_dic6={}
with open(file_path6, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic6[list(dict_a.keys())[0]]=list(dict_a.values())[0]

dataA_1=[]
dataB_1=[]
dataC_1=[]

dataA_2=[]
dataB_2=[]
dataC_2=[]

dataA_3=[]
dataB_3=[]
dataC_3=[]

dataA_4=[]
dataB_4=[]
dataC_4=[]


dataA_5=[]
dataB_5=[]
dataC_5=[]

dataA_6=[]
dataB_6=[]
dataC_6=[]


for accuracy in data_dic.values():
    camA_accracy=accuracy[3]
    dataA_1.append(camA_accracy)

    camB_accracy=accuracy[4]
    dataB_1.append(camB_accracy)

    camC_accracy=accuracy[5]
    dataC_1.append(camC_accracy)

for accuracy in data_dic2.values():
    camA_accracy=accuracy[3]
    dataA_2.append(camA_accracy)

    camB_accracy=accuracy[4]
    dataB_2.append(camB_accracy)

    camC_accracy=accuracy[5]
    dataC_2.append(camC_accracy)

for accuracy in data_dic3.values():
    camA_accracy=accuracy[3]
    dataA_3.append(camA_accracy)

    camB_accracy=accuracy[4]
    dataB_3.append(camB_accracy)

    camC_accracy=accuracy[5]
    dataC_3.append(camC_accracy)

for accuracy in data_dic4.values():
    camA_accracy=accuracy[0]
    dataA_4.append(camA_accracy)

    camB_accracy=accuracy[1]
    dataB_4.append(camB_accracy)

    camC_accracy=accuracy[2]
    dataC_4.append(camC_accracy)

for accuracy in data_dic5.values():
    camA_accracy=accuracy[0]
    dataA_5.append(camA_accracy)

    camB_accracy=accuracy[1]
    dataB_5.append(camB_accracy)

    camC_accracy=accuracy[2]
    dataC_5.append(camC_accracy)

for accuracy in data_dic6.values():
    camA_accracy=accuracy[0]
    dataA_6.append(camA_accracy)

    camB_accracy=accuracy[1]
    dataB_6.append(camB_accracy)

    camC_accracy=accuracy[2]
    dataC_6.append(camC_accracy)

data1=dataA_1
data2=dataB_2
data3=dataC_3
data4=dataA_4
data5=dataB_5
data6=dataC_6

#print(data1)
# Example data
#data = [19.995486346197243, 0, 76.85028725920921, 28.765668153813472, 0, 78.15938646215406]
# Sort the data
data_sorted1 = np.sort(data1)
data_sorted2 = np.sort(data2)
data_sorted3 = np.sort(data3)
data_sorted4 = np.sort(data4)
data_sorted5 = np.sort(data5)
data_sorted6 = np.sort(data6)

# Calculate the CDF values
cdf = np.linspace(0, 1, len(data_sorted1))
# We use the normal distribution for demonstration; you can fit other distributions as needed.
mu1, std1 = norm.fit(data1)
mu2, std2 = norm.fit(data2)
mu3, std3 = norm.fit(data3)
mu4, std4 = norm.fit(data4)
mu5, std5 = norm.fit(data5)
mu6, std6 = norm.fit(data6)

d_min= [min(data_sorted1),min(data_sorted2),min(data_sorted3),min(data_sorted4),min(data_sorted5),min(data_sorted6)]
d_max= [max(data_sorted1),max(data_sorted2),max(data_sorted3),max(data_sorted4),max(data_sorted5),max(data_sorted6)]

x = np.linspace(min(d_min), max(d_max), 1000)
cdf_smooth1 = norm.cdf(x, mu1, std1)
cdf_smooth2 = norm.cdf(x, mu2, std2)
cdf_smooth3 = norm.cdf(x, mu3, std3)
cdf_smooth4 = norm.cdf(x, mu4, std4)
cdf_smooth5 = norm.cdf(x, mu5, std5)
cdf_smooth6 = norm.cdf(x, mu6, std6)

# Calculate Lowess smoothed values
f1_smooth1 = lowess(data_sorted1, x, frac=0.05)[:, 1]  # adjust frac for smoothing strength
f1_smooth2 = lowess(data_sorted2, x, frac=0.05)[:, 1]
f1_smooth3 = lowess(data_sorted3, x, frac=0.05)[:, 1]
f1_smooth4 = lowess(data_sorted4, x, frac=0.05)[:, 1]
f1_smooth5 = lowess(data_sorted5, x, frac=0.05)[:, 1]
f1_smooth6 = lowess(data_sorted6, x, frac=0.05)[:, 1]

# Plot the CDF
#lt.figure(figsize=(8, 6))
pp=72
#plt.plot(data_sorted, cdf, marker='.', linestyle='none', label='Empirical CDF')
plt.plot(f1_smooth1, x, label='Camera A(SteerCam)', color='blue', linewidth=2)
plt.plot(f1_smooth2, x, label='Camera B(SteerCam)', color='red', linewidth=2 )
plt.plot(f1_smooth3, x, label='Camera C(SteerCam)', color='green', linewidth=2)


plt.plot(f1_smooth4, x, label='Camera A(Static)', linestyle=':', color='blue', linewidth=3)
plt.plot(f1_smooth5, x, label='Camera B(Static)', linestyle=':', color='red', linewidth=3)
plt.plot(f1_smooth6, x, label='Camera C(Static)', linestyle=':', color='green', linewidth=3)

#plt.plot(x, cdf_smooth4, label='10 Smooth CDF')
#plt.plot(x, cdf_smooth5, label='17 Smooth CDF')
plt.xlabel('F1score Accuracy',fontsize=20)
plt.ylabel('CDF',fontsize=20)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
#plt.title('Cumulative Distribution Function (CDF)',fontsize=20)

plt.legend(fontsize=16)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
# Save the plot as an image file
plt.savefig('./plot_cdf_plot.png', dpi=300) 
