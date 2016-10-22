import cv2
import re
import numpy as np
from Tkinter import *

'''
charFindCount = 41
chars = []
count = 0
dataset = open('newdataset.txt', 'r').read().split(",")
for char in range(charFindCount):
	chars.append(np.zeros((28,28), dtype=np.uint8))
	for pxCol in range(28):
		for pxRow in range(28):
			chars[char][pxCol][pxRow] = dataset[count]
			count += 1
'''

class matrixDataHandler:
	matrixWidth = 28
	charactersToRetreive = 10
	dataSet = open('newdataset.txt', 'r').read().split(",")
	characterMatrices = []

	def populateCharacterMatrices(self):
		pxCount = 0
		for i in range(self.charactersToRetreive):
			matrix = np.zeros((self.matrixWidth,self.matrixWidth), dtype=np.uint8)
			for pxCol in range(self.matrixWidth):
				for pxRow in range(self.matrixWidth):
					matrix[pxCol][pxRow] = self.dataSet[pxCount]
				pxCount += 1


class userInterfaceHandler:

	def __init__(self, tkMain):
		uiFrame = Frame(tkMain)
		uiFrame.pack()
		self.quitOption = Button(uiFrame, text="QUIT", fg="red", command=uiFrame.quit)
		self.quitOption.pack(side=LEFT)


class neuralNetworkHandler:
	
	#declare key data for nn structure
	allWeights = []
	nnPerceptrons = []


	#construct object to develop specific network structure
	def __init__(self, hiddenLayerDimensions, inputCount, outputCount, characterMatrices):

		#populate perceptrons
		nnInputs = np.zeros(inputCount)
		nnOutputs = np.zeros(outputCount)
		self.nnPerceptrons.append(nnInputs)
		self.characterMatrices = characterMatrices

		for i in range(0,hiddenLayerDimensions[0]):
			hiddenLayer = np.zeros(hiddenLayerDimensions[1])
			self.nnPerceptrons.append(hiddenLayer)

		self.nnPerceptrons.append(nnOutputs)
		print(self.nnPerceptrons)

		#populate weights
		'''...'''

	'''

	def learnFeedForward(self):
		takes single matrix
		feeds forward
		returns activated guess

	def learnBackPropagation(self):
		takes activated guess, and target

	def learnAnalyseIteration(self):
		loop through data
		gets guess from ff
		updates weights with bp
	
	'''

	#activation function for thresholding given values 
	def activateThreshold(self,value):
		if(value>=0.5):
			return 1
		else:
			return 0


def main():

	matrixData = matrixDataHandler()
	matrixData.populateCharacterMatrices()

	#neural network options
	inputPerceptronCount = matrixDataHandler.matrixWidth * matrixDataHandler.matrixWidth
	hiddenLayerDimensions = [3,15] #[hiddenLayer quanity, hiddenLayer length]
	outputPerceptronCount = 10

	neuralNetwork = neuralNetworkHandler(hiddenLayerDimensions,										
								inputPerceptronCount,
								outputPerceptronCount,
								matrixData.characterMatrices)


	tkMain = Tk()
	userInterface = userInterfaceHandler(tkMain)
	tkMain.mainloop()
	tkMain.destroy()


main()

'''cv2.waitKey(0)
cv2.destroyAllWindows()'''