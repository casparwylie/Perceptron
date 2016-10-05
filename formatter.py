dataset = open('characterData.txt', 'r').read()
newdataset = open('newdataset.txt', 'a')
for i in dataset:
	if i != "\r" and i != "\n":
		newdataset.write(i)
