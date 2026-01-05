import os
import numpy as np 
from PIL import Image 
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
## training code :
##from sklearn.model_selection import train_test_split

## helps order the images for the txt file
def numericKey(fileName):
    name, _ = os.path.splitext(fileName)
    try: 
        return int(name)
    except ValueError:
        ## sorts out the non numeric files
        return float('inf')

## loads training dataset
def loadTrainDataset(rootDir):
    imagePaths = []
    imageLabels = []

    for label in sorted(os.listdir(rootDir)):
        classDir = os.path.join(rootDir, label)

        if os.path.isdir(classDir):            
            ## Loop through all images in this class folder

            for fileName in sorted(os.listdir(classDir), key = numericKey):
                if fileName.lower().endswith(('.jpg')):
                    filePath = os.path.join(classDir, fileName)
                    imagePaths.append(filePath)
                    imageLabels.append(label)

    return imagePaths, imageLabels

## loading test dataset - sorted numerically 
def loadTestDataset(rootDir):
    testPaths = []

    for fileName in sorted(os.listdir(rootDir), key = numericKey):
        if fileName.lower().endswith(('.jpg', '.jpeg', '.png')):

            filePath = os.path.join(rootDir, fileName)
            testPaths.append(filePath)

    return testPaths

## load image as a greyscale float array 
def loadImage(path):
    image = Image.open(path).convert('L')
    return np.asarray(image, dtype=np.float32)

## extract patches of size 8X8, stride 4
def extractPatches(image):
    patchSize = 8
    stride = 4 
    patches = []    
    height, width = image.shape

    ## extracting patch and flattening
    
    for y in range(0, height - patchSize + 1, stride):
        for x in range(0, width - patchSize + 1, stride):            
            patch = image[y:y+patchSize, x:x+patchSize].flatten()

            ## mean centred and normalised
            patch = patch - np.mean(patch)
            norm = np.linalg.norm(patch)

            if norm > 1e-6:
                patch = patch / norm             
            patches.append(patch)

    return np.array(patches)

## building visual vocabulary with KMeans - needs further hyperparameter tuning
## keeps random selection of patches from training samples - limited by maxSamples, returning a trained KMeans model
def buildVocab(trainPaths):
    numClusters = 250
    maxSamples = 15000
    print(numClusters)
    print(maxSamples)
    allPatches = []

    for path in trainPaths:
        image = loadImage(path)
        patches = extractPatches(image)
        allPatches.append(patches)

    allPatches = np.vstack(allPatches)

    ## when allPatches reaches the limit, removes a random selection 
    if len(allPatches) > maxSamples:
        idx = np.random.choice(len(allPatches), maxSamples, replace = False)
        allPatches = allPatches[idx]
        
    ## training KMeans visual vocabulary 
    kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init=5)
    kmeans.fit(allPatches)
    return kmeans

 
## computing a Bag of Visual Words histogram for one image 
## assigns each patch to the nearest word, counts frequencies 
def computeHistogram(image, kmeans):
    patches = extractPatches(image)

    if len(patches) == 0:
        return np.zeros(kmeans.n_clusters, dtype=np.float32)
    
    words = kmeans.predict(patches)

    ## builds the BOVW histogram
    hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(np.float32)
    
    ## L1 normalisation 
    hist = hist / (hist.sum() + 1e-6)
    return hist

## Extracting histograms for a list of images 
def computeBOVW(paths, kmeans):
    features = []

    for path in paths:
        image = loadImage(path)
        features.append(computeHistogram(image,kmeans))

    return np.array(features)

## training a one vs all for each class
def trainClassifiers(trainFeatures, trainLabels):
    classes = sorted(list(set(trainLabels)))
    classifiers = {}

    for cls in classes:
        ## converts labels to binary (current class or other)
        yBinary = []

        for label in trainLabels:
            if label == cls:
                yBinary.append(1)
            else:
                yBinary.append(0)

        clf = LinearSVC(C=1.0)
        clf.fit(trainFeatures, yBinary)
        classifiers[cls] = clf

    return classifiers

## predicting labels using trained SVMs 
## picks highest scoring class
def predict(features, classifiers):
    predictions = []

    for hist in features:

        scores = {cls: clf.decision_function([hist])[0]
                  for cls, clf in classifiers.items()}
        predictions.append(max(scores, key=scores.get))
    
    return predictions

## Computing accuracy measure 
def computeAccuracy(trueLabels, predictedLabels):
    correct = sum(t == p for t, p in zip(trueLabels, predictedLabels))
    return correct/len(trueLabels)


## main code 

## change to match root 
trainRoot = "training"
testRoot = "testing"

trainPaths, trainLabels = loadTrainDataset(trainRoot)
testPaths = sorted(loadTestDataset(testRoot), key = numericKey)

print("Training images: ", len(trainPaths))
print("Testing images: ", len(testPaths))

## training code: 
##trainPathsSplit, valPaths, trainLabelsSplit, valLabels = train_test_split(trainPaths, trainLabels, test_size=0.2, stratify=trainLabels, random_state = 42)

##kmeans = buildVocab(trainPathsSplit)
##trainFeatures = computeBOVW(trainPathsSplit, kmeans)
##valFeatures = computeBOVW(valPaths, kmeans)
##testFeatures = computeBOVW(testPaths, kmeans)

##classifiers = trainClassifiers(trainFeatures, trainLabelsSplit)

##predictions = predict(testFeatures, classifiers)

## comment out if training:
kmeans = buildVocab(trainPaths)
trainFeatures = computeBOVW(trainPaths, kmeans)
testFeatures = computeBOVW(testPaths, kmeans)

classifiers = trainClassifiers(trainFeatures, trainLabels)
predictions = predict(testFeatures, classifiers)

## writes to run2.txt 
outputPath = "run2.txt"
with open(outputPath, "w") as f:
    for path, pred in zip(testPaths, predictions):
        fileName = os.path.basename(path)
        f.write(f"{fileName} {pred}\n")

print("results saved to txt")

## Training code : 
trainPreds = predict(trainFeatures, classifiers)
trainAccuracy = computeAccuracy(trainLabels, trainPreds)
print(f"Training accuracy: {trainAccuracy * 100:.2f}%")
##valPreds = predict(valFeatures, classifiers)
##valAccuracy = computeAccuracy(valLabels, valPreds)
##print(f"Validation accuracy: {valAccuracy * 100:.2f}%")