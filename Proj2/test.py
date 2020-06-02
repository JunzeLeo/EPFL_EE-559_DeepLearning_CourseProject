
import math
import torch    # for data generation
from modules import sequential, Linear, ReLU, batchNormalization, MSELoss, SGD
torch.set_grad_enabled(False)

# In[]
# parameters
sampleNumber = 1000
learningRate = 6e-2
epochNumber  = 50 
batchSize    = 500
evaluateIter = 20

# In[]
# data preparation
train_input, test_input = torch.rand(2, sampleNumber), torch.rand(2, sampleNumber)

# labels
train_target,test_target= torch.zeros_like(train_input), torch.zeros_like(test_input)
train_index = ((train_input-0.5).norm(dim = 0) < 1/math.sqrt(2*math.pi)) * 1
test_index  = ((test_input-0.5).norm(dim = 0) < 1/math.sqrt(2*math.pi)) * 1

train_target[train_index,  torch.arange(0, sampleNumber, 1)] = 1
test_target [test_index ,  torch.arange(0, sampleNumber, 1)] = 1

# normalize inputs
train_input = (train_input - train_input.mean(dim = 1)[:,None]) / train_input.std(dim = 1)[:,None]
test_input  = (test_input  - test_input.mean(dim = 1)[:,None] ) / test_input.std(dim = 1)[:,None]

# In[]
# training

overallTestAcc  = []
overallTrainAcc = []
for eva in range(evaluateIter):
    
    # create a model
    model = sequential(
        Linear(input_size = 2, output_size = 25),
        ReLU(),
        batchNormalization(batchSize, input_size = 25),
        
        Linear(input_size = 25, output_size = 25),
        ReLU(),
        batchNormalization(batchSize, input_size = 25),
        
        Linear(input_size = 25, output_size = 25),
        ReLU(),
        batchNormalization(batchSize, input_size = 25),
        
        Linear(input_size = 25, output_size = 2)
        )
    
    # define criterion and optimizer
    criterion = MSELoss(method = 'mean')
    optimizer = SGD(model.parameters(), lr = learningRate)

    trainLossList = []
    trainNumList  = []
    testLossList  = []
    testNumList   = []

    for epoch in range(epochNumber):
        
        ## use torch.randperm to shuffle the training data so that the model
        ## would not overfit batches. If torch.randperm is not allowed to be
        ## used, comment this block is fine for the training. The final 
        ## results would not change a lot.
        # indices = torch.randperm(sampleNumber)
        # train_input = train_input[:, indices]
        # train_target= train_target[:, indices]
        # train_index = train_index[indices]
        
        for batch in range(0, sampleNumber, batchSize):
            
            # training
            model.train()
            inputs = train_input.narrow(1, batch, batchSize) 
            output = model.forward(inputs)
            loss   = criterion.forward(output, train_target.narrow(1, batch, batchSize))
            
            optimizer.zero_grad()
            model.backward( criterion.backward() )
            optimizer.step()
            
            trainLossList.append(loss)
            trainNumList.append( ((output.argmax(dim=0) == train_index.narrow(0, batch, batchSize))*1.0).mean() )
            
            # testing
            model.test()
            inputs = test_input
            output = model.forward(inputs)
            loss   = criterion.forward(output, test_target)
            
            testLossList.append(loss)
            testNumList.append( ((output.argmax(dim=0) == test_index)*1.0).mean() )
            
            # print('Batch {%.d}, training loss %.4f, testing loss %.4f \t training correct ratio %.2f, testing correct ratio %.2f'
            #       %(batch/batchSize, trainLossList[-1], loss, trainNumList[-1], testNumList[-1]))
            
    # print('\nloss of the last batch, training loss %.4f, testing loss %.4f '%(trainLossList[-1], testLossList[-1]))
    overallTestAcc.append(testNumList)
    overallTrainAcc.append(trainNumList)
    
print('\nWith %d rounds\n'
      'the average training accuracy is %.4f, standard deviation is %.4f \n'
      'the average testing accuracy is %.4f, standard deviation is %.4f '
      %(evaluateIter, 
        torch.tensor(overallTrainAcc).mean(dim=0)[-1], torch.tensor(overallTrainAcc)[:,-1].std(),
        torch.tensor(overallTestAcc).mean(dim=0)[-1], torch.tensor(overallTestAcc)[:,-1].std()))
