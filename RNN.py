#Breakdown text into Characters
#filename = "Immortals_of_Meluha.txt"
filename = "test.txt"

with open(filename,'r') as f:
	content = f.read()

lines = list(filter(None,content.split('\n')))

# print(lines)

