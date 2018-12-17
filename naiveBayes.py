# File: Naive Bayes Classifier
# ----------------------
# This is code for CS109 problem set 6
# Maika Isogawa

def main():
    
    # training portion
    with open("netflix-train.txt") as trainingData: #change file name depending on data
        lines = trainingData.readlines()
    numInputVariables = int(lines[0])
    numDataVectors = int(lines[1])
    y0, y1 = makeVectors(lines)
    probXsGivenY0 = xProbabilities(y0, numInputVariables) # print this for 1.b.iii
    probXsGivenY1 = xProbabilities(y1, numInputVariables) # print this for 1.b.ii
    pY = float(len(y1)) / numDataVectors

    # testing portion - MLE
    with open("netflix-test.txt") as testingData: #change file name depending on data
        testingLines = testingData.readlines()
    numInputVariables = int(testingLines[0])
    numDataVectors = int(testingLines[1])
    testY0, testY1 = makeVectors(testingLines)
    y1Correct = 0
    y0Correct = 0
    for vector in testY1:
        if (vectorProbability(vector, probXsGivenY0, numInputVariables) * (1 - pY)) < (vectorProbability(vector, probXsGivenY1, numInputVariables) * pY):
            y1Correct += 1
    for vector in testY0:
        if (vectorProbability(vector, probXsGivenY0, numInputVariables) * (1 - pY)) > (vectorProbability(vector, probXsGivenY1, numInputVariables) * pY):
            y0Correct += 1

    # print MLE results
    print("Class 0: tested ", str(len(testY0)), ", correctly classified ", str(y0Correct))
    print("Class 1: tested ", str(len(testY1)), ", correctly classified ", str(y1Correct))
    print("Overall: tested ", str(len(testY0) + len(testY1)), ", correctly classified ", str(y0Correct + y1Correct))
    print("Accuracy = ", str(float(y0Correct+y1Correct)/(len(testY0) + len(testY1))))

    # Netflix portion - get 5 most indicative movies
    # getFiveMovies(probXsGivenY1)

    # testing portion - Laplace portion
    laplaceProbXsGivenY0 = laplace(y0, numInputVariables)
    laplaceProbXsGivenY1 = laplace(y1, numInputVariables)
    laplaceY0Correct = 0
    laplaceY1Correct = 0
    for vector in testY1:
        if (vectorProbability(vector, laplaceProbXsGivenY0, numInputVariables) * (1 - pY)) < (vectorProbability(vector, laplaceProbXsGivenY1, numInputVariables) * pY):
            laplaceY1Correct += 1
    
    for vector in testY0:
        if (vectorProbability(vector, laplaceProbXsGivenY0, numInputVariables) * (1 - pY)) > (vectorProbability(vector, laplaceProbXsGivenY1, numInputVariables) * pY):
            laplaceY0Correct += 1

    # print Laplace results
    print("Class 0: tested ", str(len(testY0)), ", correctly classified ", str(laplaceY0Correct))
    print("Class 1: tested ", str(len(testY1)), ", correctly classified ", str(laplaceY1Correct))
    print("Overall: tested ", str(len(testY0) + len(testY1)), ", correctly classified ", str(laplaceY0Correct + laplaceY1Correct))
    print("Accuracy = ", str(float(laplaceY0Correct+laplaceY1Correct)/(len(testY0) + len(testY1))))


# reads data and creates two vectors from the x values that map to y = 0 and y = 1
def makeVectors(lines):
    y0 = []
    y1 = []
    for i in range(len(lines)):
        if i > 1:
            # each line of data of x v y is split by ':' , and x is split by ' '
            xVector = lines[i].split(': ')[0]   # the portion before ': ' is the x values
            yClass = lines[i].split(': ')[1]    # the portion after ': ' is the y class
            if yClass[0] == '1':
                y1.append(xVector.split(' '))  # a vector of all x values that map to y = 1
            elif yClass[0] == '0':
                y0.append(xVector.split(' '))  # a vector of all x values that map to y = 0
    return y0, y1

# creates an array of probabilities of certain parameters for a certain Y class
def xProbabilities(yClass, numInputVariables):
    pxi = [0] * numInputVariables
    for vector in yClass:
        for i in range(len(pxi)):  # iterate through all of the x variables and increment when '1'
            if vector[i] == '1':
                pxi[i] += 1
    for i in range(len(pxi)):       # return probabilities
        pxi[i] = pxi[i] / len(yClass)
    #print(pxi)
    return pxi

# returns the probability that a certain vector as a particular Y value
def vectorProbability(vector, pXGivenY, numInputVariables):
    pXY = 1
    for i in range(numInputVariables):
        pXi = 0
        if vector[i] == '1':
            pXi = pXGivenY[i]
        else:
            pXi = 1 - pXGivenY[i]
        pXY = pXY * pXi
    return pXY

# Laplace estimator - takes yClass as input and returns probabilities as array
def laplace(yClass, numInputVariables):
    px = [0] * numInputVariables
    for vector in yClass:
        for i in range(numInputVariables):
            if vector[i] == '1':
                px[i] += 1
    for i in range(len(px)):
        px[i] = (float(px[i]+1) / (len(yClass) + 2))
    return px

# prints the 5 movies that are most indicative that a user will like Love Actually
def getFiveMovies(probXsGivenY1):
    movies = [0] * 5
    ratios = []
    for i in range(len(probXsGivenY1)): # build list of all probabilities for movies
        ratios.append(float(probXsGivenY1[i])/float((1-probXsGivenY1[i])))
    for i in range(len(probXsGivenY1)): # build list of movies
        for j in range(5):
            cur = movies[j]
            if ratios[i] >= ratios[cur]:  # if better, replace
                spot = i
                while j < 5:
                    cur = movies[j]
                    movies[j] = spot
                    spot = cur
                    j += 1
                break
    print(movies)

if __name__ == '__main__':
    main()
