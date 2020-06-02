
from torch import empty

_EPS = 1e-16


class weight:
    '''
    This is a class representing trainable paramenters. It contains three 
    members: name, value and grad.
    
    The value contains the parameters; the grad contains the gradients of 
    parameters.
    
    '''
    
    def __init__(self, input_size : int, output_size : int):
        '''
        initial the weight/paramenter matrix.

        Parameters
        ----------
        input_size : int
            Number of columns of the parameter matrix. In Linear layer, it 
            represents the size of the previous layer.
        output_size : int
            Number of rows of the parameter matrix. In Linear layer, it
            represents the size of the next layer

        Returns
        -------
        None.

        '''
        self.name = 'weight'
        
        self.value = empty(output_size, input_size).normal_(mean = 0, std = 1e-3)
        self.grad  = empty(output_size, input_size).zero_()
        
    def zero_grad(self):
        '''
        Set the gradient of parameters to zero.

        Returns
        -------
        None.

        '''
        self.grad.zero_()


class tanh:
    '''
    This is a class implementing the tanh() activation function.
    
    '''
    
    def __init__(self):
        self.name = 'tanh'
        
    def forward(self, x):
        """
        Activateion function, computing element wise tanh.
    
        Parameters
        ----------
        x : Tensor
            input float tensor.
    
        Returns
        -------
        Tensor
            tanh of the input tensor.
    
        """
        
        out = x.tanh()
        
        return out
    
    def backward(self, x):
        """
        It computes the first derivative of the activateion function tanh.
    
        Parameters
        ----------
        x : Tensor
            input float tensor.
    
        Returns
        -------
        Tensor
            derivative of tanh of the input tensor.
    
        """
        
        exp2x = (2*x).exp()
        out = 4*exp2x / (exp2x + 1).pow(2)
        
        return out       


class ReLU:
    '''
    This is a class implementing the ReLU() activation function. It contains
    two members: name and input. 
    
    The input is used in backpropagation.
    
    '''
    def __init__(self):
        '''
        It initializes the name and input.

        Returns
        -------
        None.

        '''
        self.name  = 'ReLU'
        self.input = 0
        
    def forward(self, x):
        '''
        ReLU forward pass. 
        
        All elements smaller than 0 are set to zeros.

        Parameters
        ----------
        x : 
            Input data.

        Returns
        -------
        x : 
            Activation results.

        '''
        self.input = x
        x[x < 0] = 0
        return x
        
    def backward(self, x):
        '''
        ReLU backward pass.
        
        Only those gradients that correpsond to a positive input data can be 
        backpropagated.

        Parameters
        ----------
        x : 
            Input gradient.

        Returns
        -------
        grad : 
            Backpropagated gradient.

        '''
        grad = (self.input > 0) * x
        return grad


class batchNormalization:
    '''
    This is a class implementing the batch normalization.
    
    Since BN layer has different forward pass for training and testing, we 
    define two mode to represent the current state, i.e training or testing.
    
    '''
    def __init__(self, batchSize: int, input_size: int):
        '''
        It initializes a batchnormalization layer by setting name, initializing
        gama, beta and all intermediate variables.
        
        Default mode is set to 'training'

        Parameters
        ----------
        batchSize : int
            The size of input batch.
        input_size : int
            The size of feature vector.

        Returns
        -------
        None.

        '''
        self.name = 'BN'
        
        # store mu and std of the training set for testing, cnt is used for 
        # the moving average.
        self.mode = 'training'
        self.cnt  = 1
        self.tmu  = empty(0, 0).new_zeros(1, input_size)
        self.tstd = empty(0, 0).new_zeros(1, input_size)
        
        self.batchSize = batchSize
        self.inputSize = input_size
        
        # The gama and beta need to be initialized to 1 and 0, respectively, 
        # which is different from the default initialization in class weight.
        self.gama = weight(input_size, 1)
        self.beta = weight(input_size, 1)
        self.gama.value = empty(0, 0).new_ones(self.gama.value.shape)  
        self.beta.value = empty(0, 0).new_zeros(self.beta.value.shape)
        
        # intermediate variables computed in forward pass and would be used 
        # in the backpropagation pass.
        self.xcen = empty(batchSize, input_size)
        self.xsqr = empty(batchSize, input_size)
        self.xsum = empty(1, input_size)
        self.xsqt = empty(1, input_size)
        self.xinv = empty(1, input_size)
        self.xhat = empty(batchSize, input_size)
        
    
    def changeMode(self, mode: str):
        '''
        Change the current mode of the BN layer.

        Parameters
        ----------
        mode : str
            The mode to switch to.

        Returns
        -------
        None.

        '''
        self.mode = mode
    
    
    def forward(self, x):
        '''
        Forward pass of the batchnormalization layer.
        
        During training, it centeralize the input batch by removing the mean 
        value and then remove the standard derivation by dividing the std. 
        Finally, the batch is rescales and biased with gama and beta. 

        Parameters
        ----------
        x : 
            Input batch, to be normalized.

        Returns
        -------
        normalized batch

        '''
        x = x.t()
        # assert(x.shape[0] == self.batchSize and x.shape[1] == self.inputSize)
        
        if self.mode == 'training':

            mu = x.sum(dim = 0)/self.batchSize
        
            self.xcen = x - mu
            self.xsqr = self.xcen**2
            self.xsum = self.xsqr.sum(dim = 0)/self.batchSize
            self.xsqt = self.xsum.sqrt()
            self.xinv = 1.0/(self.xsqt + _EPS)
            self.xhat = self.xcen * self.xinv
            
            output = self.gama.value * self.xhat + self.beta.value
        
            # saving mu and std of each training batch, using moving average.
            self.tmu = (self.cnt - 1)*self.tmu/self.cnt + mu/self.cnt
            self.tstd= (self.cnt - 1)*self.tstd/self.cnt + self.xsum/self.cnt
            self.cnt += 1
        
        elif self.mode == 'testing':
            
            Ex = self.tmu
            Ed = self.tstd * x.shape[0] / (x.shape[0] - 1)
             
            # in testing mode, using the expectation of mu and var to do BN
            # on the testing set.
            output = self.gama.value*(x - Ex)/(Ed.sqrt() + _EPS) + self.beta.value
        
        return output.t()
    
    
    def backward(self, grad):
        '''
        Backward pass of the batchnormalization layer. 
        
        It follows the computatino graph of batch normalization and computes 
        the gradient for the gama and beta. Then it propagates the gradient
        to lower layers.

        Parameters
        ----------
        grad : 
            Gradient from the upper layer.

        Returns
        -------
        Gradient to be passed to lower layers.

        '''
        grad = grad.t()
        
        self.beta.grad = grad.sum(dim = 0)/self.batchSize
        self.gama.grad = (grad * self.xhat).sum(dim = 0)/self.batchSize
        
        grad_xhat = grad * self.gama.value
        grad_xcen1= grad_xhat * self.xinv
        
        grad_xinv = (grad_xhat * self.xcen).sum(dim = 0)
        grad_xsqt = grad_xinv * (-1.0 / (self.xsqt ** 2 + _EPS))
        grad_xsum = grad_xsqt * 0.5 / (self.xsum.sqrt() + _EPS)
        grad_xsqr = grad_xsum * empty(0, 0).new_ones(self.batchSize, self.inputSize) / self.batchSize 
        grad_xcen2= grad_xsqr * 2.0 * self.xcen
        
        grad_xcen = grad_xcen1 + grad_xcen2
        grad_x1   = grad_xcen
        grad_mu   = -1.0 * grad_xcen
        grad_x2   = grad_mu * empty(0, 0).new_ones(self.batchSize, self.inputSize) / self.batchSize
        
        grad_x    = grad_x1 + grad_x2
        
        return grad_x.t()


class Linear:
    '''
    This is a class implementing the Linear layer.
    
    '''
    def __init__(self, input_size : int, output_size : int):
        '''
        It initializes a Linear layer by setting name, initializing weight, 
        bias and input.

        Parameters
        ----------
        input_size : int
            Size of the input feature.
        output_size : int
            Size of the output feature.

        Returns
        -------
        None.

        '''
        self.name = 'Linear'
        
        self.weight = weight(input_size, output_size)
        self.bia    = weight(1, output_size)
        
        self.input  = None    
        self.gradX  = None
        
    def forward(self, x):
        '''
        Forward pass of the Linear layer. 
        
        It is basically a matrix multiplication. The input is cached for the 
        back propagation.

        Parameters
        ----------
        x : 
            input (batch of) features.

        Returns
        -------
        y : 
            output features.

        '''
        self.input = x
        y = self.weight.value @ self.input + self.bia.value
        
        return y
    
    
    def backward(self, grad):
        '''
        Backward pass of the Linear layer. 

        Parameters
        ----------
        grad : 
            Gradient from the upper layer.

        Returns
        -------
        Gradient to be passed to lower layers.

        '''
        self.weight.grad = grad @ self.input.t()
        self.bia.grad    = grad.sum(dim = 1)[:,None]
        self.gradX       = self.weight.value.t() @ grad
        
        return self.gradX
    
    
class MSELoss:
    '''
    This is a class implementing the MSE loss.
    
    '''
    def __init__(self, method : str = 'mean'):
        '''
        It initializes the MSE loss by setting name, initializing method, 
        input and target.

        Parameters
        ----------
        method : str, optional
            It indicates how the loss is computed. 'mean' means the loss is 
            average over all elements.'sum' means the loss is averaged over 
            batch. The default is 'mean'.

        Returns
        -------
        None.

        '''
        self.name = 'MSELoss'
        
        self.input = 0
        self.target= 0
        
        self.size = 0
        self.method = method
        
    def forward(self, prediction, target):
        '''
        Given the prediction and traget, it computes MSE loss.

        Parameters
        ----------
        prediction : 
            Prediction vector/matrix.
        target : 
            Target vector.

        Raises
        ------
        NotImplementedError
            Currently, only 'mean' and 'sum' methods are implemented.

        Returns
        -------
        loss : TYPE
            DESCRIPTION.

        '''
        self.input = prediction
        self.target= target
        
        if self.method == 'mean':
            self.size = prediction.nelement()
            loss = (prediction - target).pow(2).sum()/prediction.nelement()
        elif self.method == 'sum':
            self.size = prediction.size(1)
            loss = (prediction - target).pow(2).sum()/prediction.size(1)
        else:
            raise NotImplementedError
        
        return loss
                
    def backward(self):
        '''
        It computes the gradient of the loss to the prediction.

        Returns
        -------
        grad : 
            The gradient of the loss to the prediction.

        '''
        grad = 2 * (self.input - self.target)/self.size
        
        return grad


class SGD:
    '''
    This is a class implementing the SGD training strategy. 
    
    '''
    def __init__(self, parameterList : list, lr : float = 1e-2):
        '''
        It initializes a SGD training strategy by storing the given parameters
        and learning rate.

        Parameters
        ----------
        parameterList : list
            The parameter list of the network to be trained.
        lr : float, optional
            leaning rate. The default is 1e-2.

        Returns
        -------
        None.

        '''
        self.parameters = parameterList
        self.stepSize   = lr
    
    def zero_grad(self):
        '''
        It sets the gradient of all parameters to zero by calling the 
        zero_grad() function of each parameter.

        Returns
        -------
        None.

        '''
        for item in self.parameters:
            item.zero_grad()
            
    def step(self):
        '''
        It updates parameters by one step.

        Returns
        -------
        None.

        '''
        for item in self.parameters:
            item.value -= self.stepSize * item.grad


class sequential:
    '''
    This is a class implementing the sequential method to create a network.

    '''
    def __init__(self, *structure):
        '''
        It initializes a network by storing the given structure of the net.

        Parameters
        ----------
        *structure : 
            A list containing the network structure, i.e. layers.

        Returns
        -------
        None.

        '''
        self.model = structure
        print('The structure of the network is:\n' + '\n'.join(str(self.model).split(',')))
        
        
    def forward(self, x):
        '''
        It calls the forward function of each layer in order to accomplish 
        a full forward pass.

        Parameters
        ----------
        x : 
            Input data to the network.

        Returns
        -------
        x : 
            The prediction of the network.

        '''
        for item in self.model:
            x = item.forward(x)
        
        return x
    
    
    def backward(self, grad):
        '''
        It calls the backward function of each layer in order (reverse order) 
        to accomplish a full backward pass.

        Parameters
        ----------
        grad : 
            The gradient to the last layer of the network.

        Returns
        -------
        None.

        '''
        for item in reversed(self.model):
            grad = item.backward(grad)


    def parameters(self):
        '''
        It returns all trainable parameters of the network. Currently, only 
        Linear layer and BN layer are supported.

        Returns
        -------
        output : 
            A list of the trainable parameters of the network.

        '''
        output = []
        for item in self.model:
            if item.name == 'Linear':
                output.append(item.weight)
                output.append(item.bia)
            if item.name == 'BN':
                output.append(item.beta)
                output.append(item.gama)
                
        return output
    
    
    def test(self):
        '''
        Changing model mode to testing.
        
        Currently it changes the mode of BN layer.

        Returns
        -------
        None.

        '''
        for item in self.model:
            if item.name == 'BN':
                item.changeMode('testing')
        
                
    def train(self):
        '''
        Changing model mode to training.
        
        Currently it changes the mode of BN layer.

        Returns
        -------
        None.

        '''
        for item in self.model:
            if item.name == 'BN':
                item.changeMode('training')