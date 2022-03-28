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


class InverseFlatten2d(torch.nn.Module):
    def __init__(self, shape):
        super(InverseFlatten2d,self).__init__()
        self.shape = shape
    def forward(self, x):
        channel = int(x.shape[1]/self.shape[0]/self.shape[1])
        return x.view((x.shape[0], channel, self.shape[0], self.shape[1]))

class CVAE(torch.nn.Module):
    def __init__(self,latent=32):
        super(CVAE,self).__init__()
        self.latent_dim = latent
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=(3,3),stride=(2,2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,kernel_size=(3,3),stride=(2,2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64*6*6,latent*2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent,32*7*7),
            torch.nn.ReLU(),
            InverseFlatten2d((7,7)),
            torch.nn.ConvTranspose2d(32,64,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64,32,kernel_size=(3,3),stride=(2,2),output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32,1,kernel_size=(3,3),padding=(1,1))
        )

    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(mean)
        return epsilon*torch.exp(logvar/2.0)+mean

    def KLdivergence(self, mean, logvar):
        return 0.5*torch.mean(torch.sum(mean**2.0+torch.exp(logvar)-logvar-1.0,dim=-1))

    def encode(self, inputs):
        latent = self.encoder(inputs)
        mean = latent[:,:self.latent_dim]
        logvar = latent[:,self.latent_dim:]
        return mean, logvar

    def predict(self, inputs):
        mean, logvar = self.encode(inputs)
        latent = self.reparameterize(mean, logvar)
        return self.decoder(latent)
    
    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        latent = self.reparameterize(mean, logvar)
        return mean, logvar, self.decoder(latent)


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
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    print("x_train.shape", x_train.shape)

    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)

    # construct the tensor dataset and data loader
    train = TensorDataset(x_train)
    loader = DataLoader(dataset=train,batch_size=32,shuffle=True)

    # initialize the model and training modules
    model = CVAE().to(device)
    Loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
    print(model)

    epochs, losses = 10, list()
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for step,data in enumerate(loader):
            inputs = data[0]
            inputs = Variable(inputs.to(device))
            mean, logvar, out = model(inputs)
            loss = torch.mean(torch.sum((out-inputs)**2,dim=(1,2,3))) + model.KLdivergence(mean, logvar)
            batch_loss.append(loss.data.item())
            if step%10==9: print('epoch %d | batch %d: loss = %.4f'%(epoch+1,step+1,batch_loss[-1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(np.mean(np.array(batch_loss)))
        print('epoch: %d: loss = %.4f, mean_loss = %.4f'%(epoch+1,batch_loss[-1],losses[-1]))
        print('---------------------------------------------------')

    torch.save(model.state_dict(),'./checkpoint/vae.pth')
    plt.plot(losses)
    plt.grid()
    plt.show()

if __name__=='__main__':
    #main()
    model = CVAE().to(device)
    model.load_state_dict(torch.load('./checkpoint/vae.pth'))
    
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
    x_test = x_test.reshape(x_test.shape[0],1,28,28)
    print("x_test.shape:", x_test.shape)
    y_test = model.predict(torch.FloatTensor(x_test).to(device))
    y_test = y_test.cpu().detach().numpy()
    print("y_test.shape:", y_test.shape)

    # generative task
    z_test = torch.rand((5,model.latent_dim)).to(device)
    y_gene = model.decoder(z_test).cpu().detach().numpy()
    print("y_gene.shape:", y_gene.shape)

    for i in range(5):
        plt.subplot(3,5,i+1)
        plt.imshow(np.squeeze(x_test[i]), cmap='gray')
        plt.subplot(3,5,i+6)
        plt.imshow(np.squeeze(y_test[i]), cmap='gray')
        plt.subplot(3,5,i+11)
        plt.imshow(np.squeeze(y_gene[i]), cmap='gray')
    plt.show()