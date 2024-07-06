import subprocess

command1 =  "apt-get install python3-pip"
command2 =  "python3 -m pip install --upgrade pip"
command3 =  "python3 -m pip install keras==2.3.0"
command4 =  "python3 -m pip install tensorflow_gpu==1.14.0"
command6 =  "python3 -m pip install gymnasium=="

command5 =  "CUDA_VISIBLE_DEVICES=6  python3 SB3_ComAI-RL_imageComAI.py"

result = subprocess.run(command1, shell=True)
result = subprocess.run(command2, shell=True)
result = subprocess.run(command3, shell=True)
result = subprocess.run(command4, shell=True)
result = subprocess.run(command5, shell=True)
print("apt-get upgrade completed successfully.")
# Check the command output
if result.returncode == 0:
    print("Command executed successfully.")
    print("Output:")
    print(result.stdout)

else:
    print("Command execution failed.")
    print("Error:")
    print(result.stderr)
   



