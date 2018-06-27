# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
    self.extra = False

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    bestAccuracy = 0
    for iteration in range(self.max_iterations):
        print "Starting iteration ", iteration, "..."
        for i in range(len(trainingData)):
            "*** YOUR CODE HERE ***"
            #Algorith derived from the explanation of perceptron on http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html
            highestScore = None
            bestLabel = None
            datum = trainingData[i]
            for label in self.legalLabels:
                score = 0
                for feature, value in datum.items():
                    score += value * self.weights[label][feature]

                #print score
                if score > highestScore or highestScore is None:
                  highestScore = score
                  bestLabel = label

            trainLbl = trainingLabels[i]
            # update weights
            if bestLabel != trainLbl:
                self.weights[trainLbl] = self.weights[trainLbl] + datum
                self.weights[bestLabel] = self.weights[bestLabel] - datum

        if self.extra:
            guess = self.classify(validationData)
            accuracyCount =  [guess[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            print "i:", iteration, " accuracy:",(100.0*accuracyCount/len(validationLabels))

            if accuracyCount > bestAccuracy:
                bestWeights = self.weights.copy()
                bestAccuracy = accuracyCount

    if self.extra:
        self.weights = bestWeights


  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses


  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    featuresWeights = self.weights[label].sortedKeys()[:100]

    return featuresWeights
