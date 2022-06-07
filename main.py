from BertDataclass import BertData_Initial, BertData_Hyperparameters, BertData_Fixed, BertData_Variable
from BertModel import BertModel
import json
from numpy import ma

'''
Main File Execution
'''

def bertExecution(fixedArguments,trainableParams):
    #Getting fixed arguments from dict param 'fixedArguments'
    validation = fixedArguments["validation"]
    modelPath = fixedArguments["modelPath"]
    tokenizerPath = fixedArguments["tokenizerPath"]
    inputDataPath = fixedArguments["inputDataPath"]
    extraVocabPath = fixedArguments["extraVocabPath"]
    preTrainedModelPath = fixedArguments["preTrainedModelPath"]
    outputDataPath = fixedArguments["outputDataPath"]
    transformerType = fixedArguments["transformerType"]

    testMode = False #Test mode definition

    #Getting variable params from trainableparams
    batchSize = int(trainableParams[-1])
    learningRates = [float(el)*(10**-5) for el in trainableParams[:-1] if not el is ma.masked]
    epochs = len(learningRates) - 1

    #Setting the fixed params to later use
    bertModelParams : BertData_Initial = BertData_Initial(
        testMode,
        validation,
        modelPath,
        tokenizerPath,
        inputDataPath,
        extraVocabPath,
        preTrainedModelPath,
        outputDataPath,
        transformerType
    )

    #Setting the hyperparameters to later use
    bertHyperparams : BertData_Hyperparameters = BertData_Hyperparameters(learningRates,batchSize,epochs)

    bertModel : BertModel = BertModel(bertModelParams) #Generating the Bert model
    bertModel.setHyperparameters(bertHyperparams) #Setting the hyperparameters
    
    #Writting the base text of the BERT return data
    jsonText = {
        "batchSize": batchSize,
        "epochsNo": epochs,
        "learningRates": learningRates,
        "epochs": {}
    }
    with open("statistics.json","w") as f: json.dump(jsonText,f)


    bertModelResults = bertModel.train() #Training the bert model

    return bertModelResults #Returning 

#Getting the fixed params from json
fixedParams = json.load(open("evoAlgParam.json"))["fitnessFunctionFixedArguments"]

#Setting the learning rates and batch size
treinableParamsList = [3, 2.6, 2.2, 1.8, 1.4, 1.0, 32] #The last element represent the batch size, the others represent the learning rate.

fitnessResultCur = bertExecution(fixedParams,trainableParams)