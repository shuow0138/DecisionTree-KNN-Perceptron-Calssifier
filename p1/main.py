import util
import datasets
import binary
import dumbClassifiers

from numpy import *
from pylab import *
from imports import *

import runClassifier
import dt
import knn
import runClassifier
import HighD
# import HighD2
import perceptron

def main():
    """
    h = dumbClassifiers.AlwaysPredictOne({})
    print(h)
    h.train(datasets.TennisData.X, datasets.TennisData.Y)
    print(datasets.TennisData.X)
    print(datasets.TennisData.Y)
    print(datasets.TennisData.Xte)
    print(h.predictAll(datasets.TennisData.X))
    print("====================================================")

    print(mean((datasets.TennisData.Y > 0) == (h.predictAll(datasets.TennisData.X) > 0)))
    print(mean((datasets.TennisData.Yte > 0) == (h.predictAll(datasets.TennisData.Xte) > 0)))
    print("====================================================")
    
    h = dumbClassifiers.AlwaysPredictMostFrequent({})
    runClassifier.trainTestSet(h, datasets.TennisData)
    runClassifier.trainTestSet(h, datasets.TennisData)
    runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.SentimentData)
    #Training accuracy 0.504167, test accuracy 0.5025
    runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.SentimentData)
    #Training accuracy 0.504167, test accuracy 0.5025

    runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData)
    #Training accuracy 0.714286, test accuracy 0.666667
    runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.SentimentData)
    #Training accuracy 0.504167, test accuracy 0.5025

    h = dt.DT({'maxDepth': 1})
    print(h)
    h.train(datasets.TennisData.X, datasets.TennisData.Y)

    h = dt.DT({'maxDepth': 2})
    print(h.train(datasets.TennisData.X, datasets.TennisData.Y))
    h = dt.DT({'maxDepth': 5})
    h.train(datasets.TennisData.X, datasets.TennisData.Y)
    h = dt.DT({'maxDepth': 2})
    h.train(datasets.SentimentData.X, datasets.SentimentData.Y)
    print(h)
    print(datasets.SentimentData.words[626])
    runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
    #Training accuracy 0.630833, test accuracy 0.595
    runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
    #Training accuracy 0.701667, test accuracy 0.6175
    runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)
    #Training accuracy 0.765833, test accuracy 0.625 

    curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 9}), datasets.SentimentData)

    runClassifier.plotCurve('DT on Sentiment Data', curve)

    curve = runClassifier.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,6,8,12,16], datasets.SentimentData)

    runClassifier.plotCurve('DT on Sentiment Data (hyperparameter)', curve)

    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
    #Training accuracy 0.785714, test accuracy 0.833333
    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)
    #Training accuracy 0.857143, test accuracy 0.833333
    
    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
    #Training accuracy 1, test accuracy 1

    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
    #Training accuracy 0.857143, test accuracy 0.833333
    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)
    #Training accuracy 0.642857, test accuracy 0.5

    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.DigitData)
#Training accuracy 0.96, test accuracy 0.64
    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 8.0}), datasets.DigitData)
#Training accuracy 0.88, test accuracy 0.81
    runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 10.0}), datasets.DigitData)
#Training accuracy 0.74, test accuracy 0.74

    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
#Training accuracy 1, test accuracy 0.94
    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
#Training accuracy 0.94, test accuracy 0.93
    runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
#Training accuracy 0.92, test accuracy 0.92

    #curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN': True, 'K': 3}),  datasets.DigitData)
    curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN': True,}), 'K', [1,2,3,4,5,6,7,8,9,10], datasets.DigitData)
    runClassifier.plotCurve('KNN on DigitData (hyperparameter)', curve)
    curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN': False,}), 'eps', [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.], datasets.DigitData)
    runClassifier.plotCurve('ESP on DigitData (hyperparameter)', curve)
    
    curve =runClassifier.learningCurveSet_knn(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
    #curve = runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
    runClassifier.plotCurve('KNN on DigitData (learningCurve)', curve)
    
    #HighD.computeDistances(datasets.DigitData.X)
    #HighD2.computeDistancesDownsampled(datasets.DigitData,1)
    
    #HighD.computeDistancesSubdims(datasets.DigitData.X,d)\
    #HighD.computeDistances(datasets.DigitData.X)
    runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.TennisData)
    #Training accuracy 0.642857, test accuracy 0.666667
    runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.TennisData)
    #Training accuracy 0.857143, test accuracy 1
"""
    # learning curve for epoch = 5
#curve = runClassifier.learningCurveSet(perceptron.Perceptron({'numEpoch': 5}), datasets.SentimentData)
#runClassifier.plotCurve('Perceptron on Sentiment Data', curve)

# different values for epoch
#curve = runClassifier.hyperparamCurveSet(perceptron.Perceptron({}), 'numEpoch', [1,2,3,4,5,6,7,8,9,10], datasets.SentimentData)
#runClassifier.plotCurve('Perceptron on Sentiment Data (hyperparameter)', curve)
    
runClassifier.plotData(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h = perceptron.Perceptron({'numEpoch': 200})
h.train(datasets.TwoDDiagonal.X, datasets.TwoDDiagonal.Y)
h

runClassifier.plotClassifier(array([ 7.3, 18.9]), 0.0)

runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 1}), datasets.SentimentData)
runClassifier.trainTestSet(perceptron.Perceptron({'numEpoch': 2}), datasets.SentimentData)

if __name__ == "__main__":
    main()

