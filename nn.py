import cv2
import re
import numpy as np
import random
from Tkinter import *
from pprint import pprint

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

	frameHeight = 700
	frameWidth = 1200
	canvasHeight = frameHeight * 0.8
	canvasWidth = frameWidth * 0.8


	def __init__(self, tkMain):
		self.tkMain = tkMain
		self.uiFrame = Frame(self.tkMain)
		self.uiFrame.pack()
		self.tkMain.minsize(width=self.frameWidth, height=self.frameHeight)
		self.tkMain.maxsize(width=self.frameWidth, height=self.frameHeight)
		self.quitOption = Button(self.uiFrame, text="QUIT", fg="red", command=self.uiFrame.quit)

	def renderNeuralNetVisualization(self,nnPerceptrons):
		pprint(nnPerceptrons)
		tkCanvas = Canvas(self.tkMain, width=self.canvasWidth, height=self.canvasHeight,background="grey")
		tkCanvas.pack()
		perceptronRadius = 10
		perceptronX = 100
		perceptronPadding = 10
		perceptronY = 100
		for perceptronLayer in range(0,len(nnPerceptrons)):
			
			for singlePerceptron in range(0,len(nnPerceptrons[perceptronLayer])):

				tkCanvas.create_oval(perceptronX - perceptronRadius, perceptronY - perceptronRadius,
					perceptronX + perceptronRadius, perceptronY + perceptronRadius)

				for perceptronWeights in range(0,len(nnPerceptrons[perceptronLayer+1])):
					tkCanvas.create_line(perceptronX, perceptronY, 200, 100)
				perceptronY += (perceptronRadius*2) + perceptronPadding

			print(perceptronX, perceptronY)
			perceptronY = 100
			perceptronX += 100
		

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

		#populate weights
		for perceptronLayer in range(0, len(self.nnPerceptrons)-1):
			weightLayer = []
			#print("Layer " + str(perceptronLayer) + ":/n")
			layerLength = len(self.nnPerceptrons[perceptronLayer])
			for singlePerceptron in range(0, layerLength):
				perceptronWeights = []
				#print("		Percept " + str(singlePerceptron)+ ":/n")
				nextLayerCount = len(self.nnPerceptrons[perceptronLayer+1])
				for singPerceptWeightCount in range(0,nextLayerCount):
					#print("			Weight " + str(singPerceptWeightCount) + ":/n")
					perceptronWeights.append(random.uniform(0,1))

				weightLayer.append(perceptronWeights)
			self.allWeights.append(weightLayer)


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
	inputPerceptronCount = 7#matrixDataHandler.matrixWidth * matrixDataHandler.matrixWidth
	hiddenLayerDimensions = [3,5] #[hiddenLayer quantity, hiddenLayer length]
	outputPerceptronCount = 10

	neuralNetwork = neuralNetworkHandler(hiddenLayerDimensions,										
								inputPerceptronCount,
								outputPerceptronCount,
								matrixData.characterMatrices)


	tkMain = Tk()
	userInterface = userInterfaceHandler(tkMain)
	userInterface.renderNeuralNetVisualization(neuralNetwork.nnPerceptrons)
	tkMain.mainloop()


main()

'''cv2.waitKey(0)
cv2.destroyAllWindows()'''