
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
miniBatch  = 50
stepSize   = 0.005
evaIters   = 20
trainList  = [ "parallelConv_NoAux", "parallelConv", "siamese_NoAux", "siamese" ]
 
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
        
        if "parallelConv" in name:
            model = parallelConv().to(device)
        elif "siamese" in name:
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
                                  train_class.narrow(0, batch, miniBatch)[:,1]),
                                  auxLoss = "NoAux" not in name,
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

    # color = ['gray', 'blue', 'red', 'green']
    # plt.figure()
    # for ind, key in enumerate(avgInAllRnd.keys()):
    #     mu = torch.tensor(allLoss_nbEv[key]).mean(dim=0)
    #     var= torch.tensor(allLoss_nbEv[key]).std(dim=0)
    
    #     plt.plot(mu, '-', color=color[ind])
    #     plt.fill_between(torch.arange(250), mu - var, mu + var,
    #                       color=color[ind], alpha=0.2)
        
    #     print('mu and var: ',torch.tensor(allLoss_nbEv[key])[:,-1].mean(), torch.tensor(allLoss_nbEv[key])[:,-1].std())
    # plt.xlabel('batches')
    # plt.ylabel('accuracy')    
    # plt.grid()
    # plt.legend(list(avgInAllRnd.keys()))
    # plt.savefig('comparison.jpg', dpi=600)
    # plt.show()
    
    # color = ['red', 'blue']
    # plt.figure()
        
    # mu = torch.tensor(overallTrainAcc).mean(dim=0)
    # var= torch.tensor(overallTrainAcc).std(dim=0)
    # plt.plot(mu, '-', color=color[0])
    # plt.fill_between(torch.arange(100), mu - var, mu + var,
    #                   color=color[0], alpha=0.2)
    
    # mu = torch.tensor(overallTestAcc).mean(dim=0)
    # var= torch.tensor(overallTestAcc).std(dim=0)
    # plt.plot(mu, '-', color=color[1])
    # plt.fill_between(torch.arange(100), mu - var, mu + var,
    #                   color=color[1], alpha=0.2)
    
    # print('mu and var: ',torch.tensor(overallTrainAcc)[:,-1].mean(), torch.tensor(overallTrainAcc)[:,-1].std())
    # print('mu and var: ',torch.tensor(overallTestAcc)[:,-1].mean(), torch.tensor(overallTestAcc)[:,-1].std())
    # plt.xlabel('batches')
    # plt.ylabel('accuracy')    
    # plt.grid()
    # plt.legend(['train_accuracy', 'test_accuracy'])
    # plt.savefig('comparison.jpg', dpi=600)
    # plt.show()
    
    # plt.hist(x.flatten(), bins = 50)
    # plt.xlabel('value')
    # plt.ylabel('number of values')
    # plt.grid()
    # plt.savefig('comparison.jpg', dpi=600)
    # plt.show()