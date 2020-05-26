"""

Grid search for some parameters would be conducted in this file. Currently, 
only the learning rate and batch size are supported.

"""

# In[]
# load packages

import torch
import torch.optim as optim

from dlc_practical_prologue import generate_pair_sets
from models import basicFcnet, basicConvnet, parallelConv, siamese
from models import percentageCorrect, count_parameters
torch.set_grad_enabled(True)

# In[]
# prepare dataset

numSamples = 1000
numEpoches = 25
miniBatch  = [25, 50, 100, 200]
stepSize   = [1e-03, 5e-03, 1e-02, 5e-02, 1e-01]
evaIters   = 10
trainList  = [ "benchmark_fc", "benchmark_cnn", "parallelConv", "siamese" ]

train_input, train_target, train_class, test_input, test_target, test_class \
    = generate_pair_sets(numSamples)
    
# if gpu is available
device = ('cpu', 'cuda')[torch.cuda.is_available()]
train_input, train_target, train_class, test_input, test_target, test_class = \
        train_input.to(device), train_target.to(device), train_class.to(device), \
        test_input.to(device), test_target.to(device), test_class.to(device)

# In[]

allLoss_CETr = {}
allLoss_nbTr = {}
allLoss_nbEv = {}

avgInAllRnd = {}

for name in trainList:
    
    allLoss_CETr[name] = {}
    allLoss_nbTr[name] = {}
    allLoss_nbEv[name] = {}
    
    avgInAllRnd[name]  = {}
    
    for lr in stepSize:
    
        for batchSize in miniBatch:
            
            allLoss_CETr[name][str(lr)+'_'+str(batchSize)] = []
            allLoss_nbTr[name][str(lr)+'_'+str(batchSize)] = []
            allLoss_nbEv[name][str(lr)+'_'+str(batchSize)] = []
            
            for eva in range(evaIters):
                
                if name == "benchmark_fc":
                    model = basicFcnet().to(device)
                elif name == "benchmark_cnn":
                    model = basicConvnet().to(device)
                elif name == "parallelConv":
                    model = parallelConv().to(device)
                elif name == "siamese":
                    model = siamese().to(device)
                else:
                    print("model not supported!")
                    break
                
                lossListCETr = []
                lossListnbTr = []
                lossListnbEv = []
                optimizer  = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
                
                print("Start training model {%s} at evaluation iter %d "%(name,eva)) 
                for iterEpoch in range(numEpoches):
                    
                    # print("---- Start epoch: ", iterEpoch)
                    for batch in range(0, numSamples, batchSize):
                        pred = model.forward(train_input.narrow(0, batch, batchSize))
                        loss = model.loss(pred, 
                                          (train_target.narrow(0, batch, batchSize),
                                          train_class.narrow(0, batch, batchSize)[:,0],
                                          train_class.narrow(0, batch, batchSize)[:,1])
                                          )
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        lossListCETr.append(loss.detach())
                        # print("------ batch %6d, CrossEntropy loss : %.4f"%(batch, loss.detach()))
                    
                        pctg_correctTr = percentageCorrect( model(train_input, is_vis=True), train_target )
                        lossListnbTr.append(pctg_correctTr.item())
                        # print("     \t  pctg of wrong train samples  : %.4f"%(pctg_correctTr))
                        
                        pctg_correctEv = percentageCorrect( model(test_input, is_vis=True), test_target )
                        lossListnbEv.append(pctg_correctEv.item())    
                        # print("     \t  pctg of wrong test samples   : %.4f"%(pctg_correctEv))
                        
                allLoss_CETr[name][str(lr)+'_'+str(batchSize)].append(lossListCETr)
                allLoss_nbTr[name][str(lr)+'_'+str(batchSize)].append(lossListnbTr)
                allLoss_nbEv[name][str(lr)+'_'+str(batchSize)].append(lossListnbEv)
                
            avgInAllRnd[name][str(lr)+'_'+str(batchSize)] = torch.tensor(allLoss_nbEv[name][str(lr)+'_'+str(batchSize)]).mean(dim=0)
            
            print("\nfor {%s} with %d parameters, lr = %.e, batchsize = %d:\n " 
                  "--average percentage of correct test samples (final epoch)    : %.4f \n\n"
                  %(name, count_parameters(model), lr, batchSize,
                    avgInAllRnd[name][str(lr)+'_'+str(batchSize)][-1]))
    