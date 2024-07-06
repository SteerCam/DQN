import statistics

def read_all_lines(filename):
  """
  Reads all lines from a text file and returns them as a list.

  Args:
      filename (str): The path to the text file.

  Returns:
      list: A list of strings, where each element is a line from the file.

  Raises:
      FileNotFoundError: If the specified file is not found.
  """

  try:
    # Open the file in read mode with context manager (automatic closing)
    with open(filename, 'r') as file:
      lines = file.readlines()
  except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {filename}")

  return lines

# Example usage
list_a=[]
filename = "rank1_invokation2.txt"  # Replace with your actual file name
try:
  lines = read_all_lines(filename)
  for line in lines:
    eachline=float(line.strip().split(":")[1])
    list_a.append(eachline)
    #print(eachline)  # Print each line with trailing newline removed
except FileNotFoundError as e:
  print(e)
list_a.sort()
new_list=list_a[-500:]
mean_s = statistics.mean(new_list)
std_s = statistics.stdev(new_list)
print(mean_s,std_s)