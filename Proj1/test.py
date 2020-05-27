"""

OBJECT:
    The objective of this project is to test different architectures to compare 
    two digits visible in a two-channel image. It aims at showing in particular
    the impact of weight sharing, and of the use of an auxiliary loss to help 
    the training of the main objective.

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
miniBatch  = 100
stepSize   = 2e-3
evaIters   = 20
trainList  = [ "siamese" ]
 
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

print("\nWarning, the test.py include several models and their performances are \
estimated over 20 rounds to get their standard deviations. Each round contains \
25 epoches with batchsize being 100. \
\n----THUS IT COULD TAKE A BIT TIME ON THE VIRTUAL MACHINE----\n")

for name in trainList:
    
    allLoss_CETr[name] = []
    allLoss_nbTr[name] = []
    allLoss_nbEv[name] = []
    
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
        optimizer  = optim.SGD(model.parameters(), lr = stepSize, momentum = 0.9)
        
        print("Start training model {%s} at evaluation iter %d "%(name,eva)) 
        for iterEpoch in range(numEpoches):
            
            # print("---- Start epoch: ", iterEpoch)
            for batch in range(0, numSamples, miniBatch):
                pred = model.forward(train_input.narrow(0, batch, miniBatch))
                loss = model.loss(pred, 
                                  (train_target.narrow(0, batch, miniBatch),
                                  train_class.narrow(0, batch, miniBatch)[:,0],
                                  train_class.narrow(0, batch, miniBatch)[:,1])
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
                
        allLoss_CETr[name].append(lossListCETr)
        allLoss_nbTr[name].append(lossListnbTr)
        allLoss_nbEv[name].append(lossListnbEv)
        
    print("\nfor {%s} with %d parameters:\n " 
          "--average percentage of correct test samples    : %.4f ( std=%.4f )\n\n"
          %(name, count_parameters(model), 
            torch.tensor(allLoss_nbEv[name])[:,-1].mean(), torch.tensor(allLoss_nbEv[name])[:,-1].std()))
    
    avgInAllRnd[name] = torch.tensor(allLoss_nbEv[name]).mean(dim=0)
    
    # plt.figure()
    # plt.plot(torch.tensor(allLoss_nbEv[name]).t())
    # plt.title(name)
    
    # plt.figure()
    # plt.plot(avgInAllRnd["benchmark_fc"]);
    # plt.plot(avgInAllRnd["benchmark_cnn"]);
    # plt.plot(avgInAllRnd['parallelConv'])
    # plt.plot(avgInAllRnd['siamese'])
    # plt.legend(trainList)
    # plt.grid()
    # plt.title("Average evaluation accuracy")
    
