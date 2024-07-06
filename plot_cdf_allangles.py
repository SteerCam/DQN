import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

file_path = './final_all_combinations_accuracy_and_occlusionglobal_test_summery_ABCbest_model_test9.txt'

data_dic={}
with open(file_path, 'r') as file:
    for line in file:
        dict_a = ast.literal_eval("{"+line+"}")
        data_dic[list(dict_a.keys())[0]]=list(dict_a.values())[0]

dataA=[]
dataB=[]
dataC=[]
for accuracy in data_dic.values():
    camA_accracy=accuracy[3]
    dataA.append(camA_accracy)

    camB_accracy=accuracy[4]
    dataB.append(camB_accracy)

    camC_accracy=accuracy[5]
    dataC.append(camC_accracy)

data=dataA
print(data)
# Example data
#data = [19.995486346197243, 0, 76.85028725920921, 28.765668153813472, 0, 78.15938646215406]
# Sort the data
data_sorted = np.sort(data)

# Calculate the CDF values
cdf = np.linspace(0, 1, len(data_sorted))
# We use the normal distribution for demonstration; you can fit other distributions as needed.
mu, std = norm.fit(data)
x = np.linspace(min(data_sorted), max(data_sorted), 1000)
cdf_smooth = norm.cdf(x, mu, std)
# Plot the CDF
plt.figure(figsize=(8, 6))
#plt.plot(data_sorted, cdf, marker='.', linestyle='none', label='Empirical CDF')
plt.plot(x, cdf_smooth, label='Smooth CDF')
plt.xlabel('CamA F1score Accuracy')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig('./plot_cdf_plot.png', dpi=300) 
