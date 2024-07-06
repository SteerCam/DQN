import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

file_path = './global_detection_numberfmap.txt'


data_dic={}
file_path = "./global_detection_numberfmap.txt" #global_detection_numberfmap  global_detection_number
with open(file_path, "r") as file:
   data_dic = json.load(file)



data_dqn=[]
data_base=[]


for accuracy in data_dic.values():
    camA_accracy=accuracy[0]
    data_base.append(camA_accracy)

    camB_accracy=accuracy[1]
    data_dqn.append(camB_accracy)


#print(data_base)
data1=data_base
data2=data_dqn


#print(data1)
# Example data
#data = [19.995486346197243, 0, 76.85028725920921, 28.765668153813472, 0, 78.15938646215406]
# Sort the data
data_sorted1 = np.sort(data1)
data_sorted2 = np.sort(data2)


# Calculate the CDF values
cdf = np.linspace(0, 1, len(data_sorted1))
# We use the normal distribution for demonstration; you can fit other distributions as needed.
mu1, std1 = norm.fit(data1)
mu2, std2 = norm.fit(data2)


d_min= [min(data_sorted1),min(data_sorted2)]
d_max= [max(data_sorted1),max(data_sorted2)]

x = np.linspace(min(d_min), max(d_max), 1000)
cdf_smooth1 = norm.cdf(x, mu1, std1)
cdf_smooth2 = norm.cdf(x, mu2, std2)


# Plot the CDF
plt.figure(figsize=(8, 6))
#plt.plot(data_sorted, cdf, marker='.', linestyle='none', label='Empirical CDF')
plt.plot(x, cdf_smooth2, label='(Ours Method)', color='blue' )

plt.plot(x, cdf_smooth1, label='static', linestyle=':', color='green')

#plt.plot(x, cdf_smooth4, label='10 Smooth CDF')
#plt.plot(x, cdf_smooth5, label='17 Smooth CDF')
plt.xlabel('F1score Accuracy')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function (CDF)')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig('./plot_cdf_plotGlobal.png', dpi=300) 
