import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

# initialize the devices
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU detected! Loading the model to CUDA...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AutoEncoder(torch.nn.Module):
    def __init__(self, latent=[256,64,4]):
        super(AutoEncoder,self).__init__()
        layer_number = 1
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, latent[0])
        )
        for i in range(1,len(latent)):
            self.encoder.add_module('%d'%layer_number,torch.nn.ReLU())
            self.encoder.add_module('%d'%(layer_number+1),torch.nn.Linear(latent[i-1],latent[i]))
            layer_number = layer_number + 2
        layer_number = 2
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent[-1], latent[-2]),
            torch.nn.ReLU()
        )
        for i in range(len(latent)-2,0,-1):
            self.decoder.add_module('%d'%layer_number,torch.nn.Linear(latent[i],latent[i-1]))
            self.decoder.add_module('%d'%(layer_number+1),torch.nn.ReLU())
            layer_number = layer_number + 2
        self.decoder.add_module('%d'%layer_number,torch.nn.Linear(latent[0],28*28))
        #self.decoder.add_module('%d'%(layer_number+1),torch.nn.Tanh())
    
    def forward(self, inputs):
        flatten = inputs.view(inputs.shape[0],-1)
        latent = self.encoder(flatten)
        return self.decoder(latent)
    
    def predict(self, inputs):
        flatten = inputs.view(inputs.shape[0],-1)
        latent = self.encoder(flatten)
        return latent, self.decoder(latent)


def main():
    # dataset preparation
    #mnist = tf.keras.datasets.mnist
    mnist = np.load('./mnist.npz')
    x_train, x_test = mnist['arr_0'], mnist['arr_2']
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("x_train.shape:", x_train.shape)
    print("x_test.shape:", x_test.shape)
    np.set_printoptions(precision=2)

    # data type conversion
    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    print("x_train.shape", x_train.shape)

    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)

    # construct the tensor dataset and data loader
    train = TensorDataset(x_train)
    loader = DataLoader(dataset=train,batch_size=32,shuffle=True)

    # initialize the model and training modules
    model = AutoEncoder().to(device)
    Loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    print(model)

    epochs, losses = 5, list()
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for step,data in enumerate(loader):
            inputs = data[0]
            inputs = Variable(inputs.to(device))
            out = model(inputs)
            loss = Loss(out,inputs)
            batch_loss.append(loss.data.item())
            if step%10==9: print('epoch %d | batch %d: loss = %.4f'%(epoch+1,step+1,batch_loss[-1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(np.array(batch_loss)))
        print('epoch: %d: loss = %.4f, mean_loss = %.4f'%(epoch+1,batch_loss[-1],losses[-1]))
        print('---------------------------------------------------')

    torch.save(model.state_dict(),'./checkpoint/ae.pth')
    plt.plot(losses)
    plt.grid()
    plt.show()

if __name__=='__main__':
    #main()
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load('./checkpoint/ae.pth'))
    
    #mnist = tf.keras.datasets.mnist
    mnist = np.load('./mnist.npz')
    x_train, x_test = mnist['arr_0'], mnist['arr_2']
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.concatenate((x_train,x_test),axis=0)
    x_train = x_train / 255.0
    print("x_train.shape:", x_train.shape)
    np.set_printoptions(precision=2)

    # visualize a sample in the dataset
    sample = np.random.randint(0,x_train.shape[0],size=5)
    x_test = x_train[sample,...]
    print("x_test.shape:", x_test.shape)

    x_test = x_test.reshape(x_test.shape[0],28*28)
    y_test = model(torch.FloatTensor(x_test).to(device))
    y_test = y_test.cpu().detach().numpy()
    print(y_test.shape)
    x_test = x_test.reshape(x_test.shape[0],28,28)
    y_test = y_test.reshape(y_test.shape[0],28,28)

    for i in range(5):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i], cmap='gray')
        plt.subplot(2,5,i+6)
        plt.imshow(y_test[i], cmap='gray')
    plt.show()
