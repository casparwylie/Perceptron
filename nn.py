import cv2
import re
import numpy as np
import random
from pprint import pprint


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
			self.characterMatrices.append(matrix)


class userInterfaceHandler:

	frameHeight = 1000
	frameWidth = 1600
	canvasHeight = frameHeight * 0.9
	canvasWidth = frameWidth * 0.9


	def __init__(self):
		self.uiCanvas = np.zeros((self.canvasHeight,self.canvasWidth), dtype=np.uint8)
		
	def renderNeuralNetVisualization(self,nnPerceptrons):
		pprint(nnPerceptrons)

		perceptronYstart = 20
		perceptronRadius = 10
		perceptronX = 100
		perceptronDistX = 300
		perceptronPadding = 15
		perceptronY = perceptronYstart

		for perceptronLayer in range(0,len(nnPerceptrons)):
			
			for singlePerceptron in range(0,len(nnPerceptrons[perceptronLayer])):

				cv2.circle(self.uiCanvas, (perceptronX,perceptronY), perceptronRadius, (255), 1)

				perceptronDistY = (perceptronRadius*2) + perceptronPadding
				if(perceptronLayer < len(nnPerceptrons)-1):
					perceptronYForLine = perceptronYstart
					for perceptronWeights in range(0,len(nnPerceptrons[perceptronLayer+1])):
						cv2.line(self.uiCanvas,(perceptronX, perceptronY), (perceptronX+perceptronDistX, perceptronYForLine),(255),1)
						perceptronYForLine += perceptronDistY

				perceptronY += perceptronDistY

			perceptronY = perceptronYstart
			perceptronX += perceptronDistX

		cv2.imshow("nn", self.uiCanvas);
		

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
	

	def learnFeedForward(self, matrix):
		ofMatrixToAnalyse = 2 #len(matrix)
		pxCount = 0
		for pxCol in range(ofMatrixToAnalyse):
			for pxRow in range(ofMatrixToAnalyse):
				self.nnPerceptrons[pxCount] = matrix[pxCol][pxRow]
				pxCount += 1

		print(self.nnPerceptrons)

	def learnAnalyseIteration(self):
		for matrix in self.characterMatrices:
			learnFeedForward(matrix)

	#activation function for thresholding given values 
	def activateThreshold(self,value):
		if(value>=0.5):
			return 1
		else:
			return 0


def main():

	matrixData = matrixDataHandler()
	matrixData.populateCharacterMatrices()
	for i in range(9):
		cv2.imshow("frame"+str(i),matrixData.characterMatrices[i])

	#neural network options
	inputPerceptronCount = 20#matrixDataHandler.matrixWidth * matrixDataHandler.matrixWidth
	hiddenLayerDimensions = [3,25] #[hiddenLayer quantity, hiddenLayer length]
	outputPerceptronCount = 20

	neuralNetwork = neuralNetworkHandler(hiddenLayerDimensions,										
								inputPerceptronCount,
								outputPerceptronCount,
								matrixData.characterMatrices)


	userInterface = userInterfaceHandler()
	userInterface.renderNeuralNetVisualization(neuralNetwork.nnPerceptrons)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


main()