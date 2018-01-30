#Breakdown text into Characters
filename = "Immortals_of_Meluha.txt"
#filename = "test.txt"

with open(filename,'r') as f:
	content = f.read()

lines = list(filter(None,content.split('\n')))


# print(lines)

char_set = set()

for line in lines:
	for char in line:
		char_set.add(char)

print(char_set)