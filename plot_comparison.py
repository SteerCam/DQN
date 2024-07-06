import linecache
import matplotlib.pyplot as plt5
import numpy as np

def read_line(filename, line_number):
    line = linecache.getline(filename, line_number)
    return line.strip()

def find_lines(filename, target_lines):
    results_baseline = {}
    results_DQN = {}
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
                target_line = target_lines[0] 
                if target_line in line:
                    if target_line in results_DQN:
                        results_DQN[target_line].append((line_number, line))
                    else:
                        results_DQN[target_line] = [(line_number, line)]
                target_line == target_lines[1] 
                if target_line in line:
                    if target_line in results_baseline:
                        results_baseline[target_line].append((line_number, line))
                    else:
                        results_baseline[target_line] = [(line_number, line)]
    return results_baseline,results_DQN

# Specify the target lines to search for
target_lines = ["average fscore for RL", "average  fsocre for baseline is:"]

# Provide the filename of the log file
filename = "logwithbestPairs.txt"

# Call the function to find the lines and their line numbers
line_results_base,line_results_DQN = find_lines(filename, target_lines)


#print(line_results_base,line_results_DQN)

# Display the results
#print(line_results)
value_x=[]
value_base=[]
value_dqn=[]
plt5.figure(figsize=(20, 10)) 

original_dict={}
for target_line, line_info in line_results_base.items():
    print(f"Line(s) containing '{target_line}':")
    for line_number, line in line_info:
        print(line_number)
        kk=str(read_line("logwithbestPairs.txt",line_number-3))
        pp=read_line("logwithbestPairs.txt",line_number-2)
        dqn = read_line("logwithbestPairs.txt",line_number)
        kk=kk.split(":")
        print("(a)",kk)
        print("(b)",pp)
        print("(c)",dqn)
        pp_base=float(pp.split(":")[-1])
        pp_dqn=float(dqn.split(" ")[-1])
        index=int(kk[-1])
        value_x.append(index)
        value_base.append(pp_base)
        value_dqn.append(pp_dqn)
        original_dict[index] = [pp_base,pp_dqn]

print(original_dict)
sorted_by_keys = {k: original_dict[k] for k in sorted(original_dict)}

value_base=np.array(list(sorted_by_keys.values()))[:,0]
value_dqn=np.array(list(sorted_by_keys.values()))[:,1]
value_x = list(sorted_by_keys.keys())
#plt5.plot(value_x, value_base, label="accyracy Angle")
#plt5.plot(value_x, value_dqn, label="accuracy dqn")
bar_width=0.8
index=range(len(value_x))
plt5.bar(index, value_base, width=bar_width, label='Accuracy baseline', alpha=0.5)

plt5.bar(index, value_dqn, width=bar_width, label='Accuracy DQN', alpha=0.5)

plt5.ylabel("accuracy")
plt5.title("accuracy comparison")
plt5.xticks(index, value_x,fontsize=12)
plt5.xlabel("iteration")
#plt5.xticks(value_x)
plt5.xticks(rotation=90)
plt5.legend(loc='lower right') 

plt5.grid(True)
# Save the plot as an image file
plt5.savefig("./final_rl_base_accuracy.png")
plt5.clf()





#for target_line, line_info in line_results_DQN.items():
#    print(f"Line(s) containing '{target_line}':")
#    for line_number, line in line_info:
#        print(f"Line {line_number}: {line}")


