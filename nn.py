import cv2
import re
import numpy as np
import random
import math
from pprint import pprint


class matrixDataHandler:
	matrixWidth = 28
	charactersToRetreive = 1000
	dataSet = open('newdataset.txt', 'r').read().split(",")
	characterMatrices = []
	characterTargets = []

	def populateCharacterMatrices(self):
		pxCount = 0
		for i in range(self.charactersToRetreive):
			matrix = np.zeros((self.matrixWidth,self.matrixWidth), dtype=np.uint8)
			for pxCol in range(self.matrixWidth):
				for pxRow in range(self.matrixWidth):
					if(pxCount%(self.matrixWidth**2)==0):
						self.characterTargets.append(int(self.dataSet[pxCount]))
					else:
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

		exampleInputLimitCount = 70 #zero for all
		perceptronYstart = 20
		perceptronRadius = 5
		perceptronX = 100
		perceptronDistX = 120
		perceptronPadding = 15
		perceptronY = perceptronYstart

		for perceptronLayer in range(0,len(nnPerceptrons)):
			
			lengthOfLayer = len(nnPerceptrons[perceptronLayer])
			if(perceptronLayer==0 and exampleInputLimitCount > 0 and exampleInputLimitCount < lengthOfLayer):
				lengthOfLayer = exampleInputLimitCount
			for singlePerceptron in range(0,lengthOfLayer):

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

		cv2.imshow("nn", self.uiCanvas)
		

class neuralNetworkHandler:
	
	#declare key data for nn structure
	allWeights = []
	nnPerceptrons = []
	weightChangeRecord = []

	#construct object to develop specific network structure
	def __init__(self, hiddenLayers, inputCount, outputCount, characterMatrices,characterTargets,learningConstant):

		#populate perceptrons
		print(inputCount, "yyeee")
		nnInputs = np.zeros(inputCount)
		nnOutputs = np.zeros(outputCount)
		self.nnPerceptrons.append(nnInputs)
		self.characterMatrices = characterMatrices
		self.characterTargets = characterTargets
		self.learningConstant = learningConstant
		self.outputCount = outputCount

		for i in hiddenLayers:
			hiddenLayer = np.zeros(i)
			self.nnPerceptrons.append(hiddenLayer)

		self.nnPerceptrons.append(nnOutputs)

		#populate weights
		for perceptronLayer in range(1, len(self.nnPerceptrons)):
			weightLayer = []
			weightChangeRecordLayer = []
			layerLength = len(self.nnPerceptrons[perceptronLayer])
			for singlePerceptron in range(0, layerLength):
				perceptronWeights = []
				weightChangeRecordPerceptron = []
				prevLayerCount = len(self.nnPerceptrons[perceptronLayer-1])
				for singPerceptWeightCount in range(0,prevLayerCount):
					perceptronWeights.append(random.uniform(0,1))
					weightChangeRecordPerceptron.append(0)

				weightLayer.append(perceptronWeights)
				weightChangeRecordLayer.append(weightChangeRecordPerceptron)

			self.allWeights.append(weightLayer)
			self.weightChangeRecord.append(weightChangeRecordLayer)

	def learnFeedForward(self, matrix):
		ofMatrixToAnalyse = len(matrix)
		ofMatrixToAnalyseStart = 0
		pxCount = 0
		for pxCol in range(ofMatrixToAnalyseStart,ofMatrixToAnalyseStart+ofMatrixToAnalyse):
			for pxRow in range(ofMatrixToAnalyseStart,ofMatrixToAnalyseStart+ofMatrixToAnalyse):
				self.nnPerceptrons[0][pxCount] = matrix[pxCol][pxRow]
				pxCount += 1

		for afterInputLayer in range(1, len(self.nnPerceptrons)):
			for perceptronCount in range(0, len(self.nnPerceptrons[afterInputLayer])):
				hiddenPerceptronSum = 0
				for prevPerceptronCount in range(0,len(self.nnPerceptrons[afterInputLayer-1])): 
					prevPerceptron = self.nnPerceptrons[afterInputLayer-1][prevPerceptronCount]
					relevantWeight = self.allWeights[afterInputLayer-1][perceptronCount][prevPerceptronCount]
					hiddenPerceptronSum += prevPerceptron * relevantWeight
				
				self.nnPerceptrons[afterInputLayer][perceptronCount] = self.activateThreshold(hiddenPerceptronSum, "sigmoid")


	ccount = 0

	def learnBackPropagation(self, targetVal):
		targetVector = self.populateTargetVector(targetVal)
		self.ccount += 1
		for weightLayerCount in range(len(self.allWeights)-1,-1,-1):

			for weightPerceptronCount in range(0, len(self.allWeights[weightLayerCount])):
				weightPerceptronVal = self.nnPerceptrons[weightLayerCount+1][weightPerceptronCount]
				finalActivatedToSumStep = weightPerceptronVal * (1-weightPerceptronVal) #if sigmoid

				if(weightLayerCount == len(self.allWeights)-1):
					prevStepBackPropErrorVal = weightPerceptronVal - targetVector[weightPerceptronCount-1]
					if( targetVector[weightPerceptronCount-1] == 1):
						print(prevStepBackPropErrorVal)

				for singleWeightCount in range(0, len(self.allWeights[weightLayerCount][weightPerceptronCount])):
					currentWeightVal = self.allWeights[weightLayerCount][weightPerceptronCount][singleWeightCount]
					
					if(weightLayerCount != len(self.allWeights)-1):
						weightChangeLayerToSum = weightLayerCount + 1
						prevStepBackPropErrorVal  = 0
						for weightChangePerceptronCount in range(0, len(self.weightChangeRecord[weightChangeLayerToSum])):
							
							previousWeightChange = self.weightChangeRecord[weightChangeLayerToSum][weightChangePerceptronCount][weightPerceptronCount]
							prevStepBackPropErrorVal += previousWeightChange

					weightInputPerceptronVal = self.nnPerceptrons[weightLayerCount][singleWeightCount]
					finalActivatedToWeight = finalActivatedToSumStep * weightInputPerceptronVal
					fullStepBackVal = prevStepBackPropErrorVal * finalActivatedToWeight
					self.weightChangeRecord[weightLayerCount][weightPerceptronCount][singleWeightCount] = fullStepBackVal
					newWeightVal = currentWeightVal - (self.learningConstant * fullStepBackVal)
					self.allWeights[weightLayerCount][weightPerceptronCount][singleWeightCount] = newWeightVal

	def learnAnalyseIteration(self):
		matrix = self.characterMatrices[0]
		targetVal = self.characterTargets[0]
		for matrix in self.characterMatrices:
			self.learnFeedForward(matrix)
			self.learnBackPropagation(targetVal)

	#activation function for thresholding given values 
	def activateThreshold(self,value, type):
		if(type == "step"):
			if(value>=0.5):
				return 1
			else:
				return 0
		elif(type == "sigmoid"):
			return 1/(1+(math.e**-value))


	def populateTargetVector(self,target):
		vector = []
		for i in range(0,self.outputCount-1):
			vector.append(0)
		vector[target] = 1
		return vector


def main():

	matrixData = matrixDataHandler()
	matrixData.populateCharacterMatrices()
	'''for i in range(matrixData.charactersToRetreive):
		cv2.imshow("frame"+str(i)+"-"+str(matrixData.characterTargets[i]),matrixData.characterMatrices[i])
	'''
	#neural network options
	inputPerceptronCount = matrixDataHandler.matrixWidth * matrixDataHandler.matrixWidth
	hiddenLayers = [15] #[hidden layer length]
	outputPerceptronCount = 10
	learningConstant = 0.01

	neuralNetwork = neuralNetworkHandler(hiddenLayers,										
								inputPerceptronCount,
								outputPerceptronCount,
								matrixData.characterMatrices,
								matrixData.characterTargets,
								learningConstant)


	#userInterface = userInterfaceHandler()
	#userInterface.renderNeuralNetVisualization(neuralNetwork.nnPerceptrons)

	neuralNetwork.learnAnalyseIteration()

	cv2.waitKey(0)
	cv2.destroyAllWindows()


main()
