
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset #alows dataset managment, littel batches to train....

#%%
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import itertools as it
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import seaborn as sn

#%%
#Load data and tragtes
data = pd.read_csv('/home/paux/FMP/NN/nn_data/whole_nn_data.csv', index_col = 0) 
data = data.drop(columns = ['locus', 'alleles', 'rsid'])
#Load targets
y = pd.read_csv('/home/paux/FMP/NN/nn_data/target_data.csv', index_col = 0 )
print(data.shape, y.shape)
Counter(y.encoded_label)
 
#%%
#Input size and number of classes definition
input_size =  data.shape[1] #number of columns
num_classes =  len(y.encoded_label.unique()) #number of labels on clinvar_clnsig

#%%
#Hyperparameters
num_neurons =  60
num_epoch =  30
learning_rate = 0.001
batch_size = 100

#%%
#Print the hyper parameters used
print(('Neurons: %s, epochs: %s, learning_rate: %s and batch size: %s')%(num_neurons, num_epoch, learning_rate, batch_size ))

#Shuffel the columns
data = data.sample(axis = 1, frac =1, random_state = 22)

#Split data in to train and test datasets (8:2)
x_train, x_test, y_train, y_test =  train_test_split(data, y, test_size= 0.2, random_state= 22)

#Define a dtaset class
class MyDataset(Dataset):
    
    def __init__(self, x, y):
        
        self.x_train = torch.tensor(x.to_numpy(), dtype = torch.float32)
        self.y_train = torch.tensor(y.to_numpy(), dtype = torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

#Create the iterable objects to train and test the NN 
train_set = MyDataset(x_train, y_train)
test_set = MyDataset(x_test, y_test)
train_load = DataLoader(train_set, batch_size= batch_size, shuffle= True) 
test_load = DataLoader(test_set,  batch_size= batch_size, shuffle= True)

#Create the fully conected NN
class NN(nn.Module):
    def __init__(self, input_size, num_classes): 
        super(NN, self).__init__()  #Calls the init of the parent class (nn.MOdule)
        self.fc1 = nn.Linear(input_size, num_neurons) #Input layer, input_size has the same size as input_vector
                                            #50 is the num of nodes next layer
        self.fc2 = nn.Linear(num_neurons, num_classes) #Hidden layer, num_classes: the number of items on the classificator
        self.dropout = nn.Dropout(p=0.15) #Drop 15% nodes randomply, decreas overfiting
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Set device (CPU or GPU), checks if GPU available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize network 
model = NN(input_size = input_size, num_classes = num_classes).to(device)

#Initialize variables for the loss plot (loss history)
y_loss = []
y_err = []
x_epoch = []

#Function nedeed fro loss plot (this must be before the training proces)
#Plot loss curve
fig = plt.figure()
ax0 = fig.add_subplot(title="Loss")

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss, 'bo-', label='train', lw=2 )
    
    if current_epoch == 0:
        ax0.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig.savefig(os.path.join('/home/paux/FMP/NN/loss_graphs/loss_nn%s_ep%s_lr%s_bs%s.jpg' % (num_neurons, num_epoch, learning_rate, batch_size)))

# Loss and optimizer functions 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Initialize lists to store results for ROC curve 
labels= []
scor = []

#Train the network (Train Loop)
for epoch in range(num_epoch):
    running_loss = 0.0
    running_corrects = 0.0
    print(epoch)
    for batch_id, (data, targets) in enumerate(train_load):
        
        #Get data to cuda if possible
        data = data.to(device = device)
        targets = targets.to(device = device)
        
        #Get correct type (long tensor) and shape
        targets = targets.type(torch.LongTensor)
        targets = torch.reshape(targets, (-1,))
        
        #Forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        #Backward
        optimizer.zero_grad()
        loss.backward()
        
        #Gredient descent or adam step
        optimizer.step()

        #Graph loss
        running_loss += loss.item() * batch_id
        _, predictions = scores.max(1)
        running_corrects += float(torch.sum(predictions == targets.data))
    
    #Graph 
    epoch_loss = running_loss / len(x_train)
    epoch_acc = running_corrects / len(x_train)
    
    y_loss.append(epoch_loss)
    y_err.append(1.0 - epoch_acc)
    
    draw_curve(epoch)

#Check acurecy on the training and test to see performance
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): #Let know pytorch not need to calculate gradient douring the avaluation
        for x, y in loader:
            
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)
            y = torch.reshape(y, (-1,))
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y ).sum().item()
            num_samples += predictions.size(0)

            #Store results
            labels.append(y.tolist())
            scor.append(scores.tolist())

        accuracy = round(float(num_correct)/float(num_samples) *100, 4)
        print('Epoch: %s, batch_size: %s, lr: %s number of classes: %s. Got %s / %s, with accuracy of %s' % (num_epoch, batch_size, learning_rate, num_classes, num_correct, num_samples, accuracy))
    
    model.train()
    return accuracy


#See accuracy on the training and test data
accuracy_train = check_accuracy(train_load, model)

#Re-initialite ROC variables 
labels= []
scor = []
accuracy_test= check_accuracy(test_load, model)

#Roc curve and AUC for multy class classiffier
labels = list(it.chain.from_iterable(labels))
labels = label_binarize(labels, classes = list(range(num_classes)))
scor = list(it.chain.from_iterable(scor))
scor = np.array(scor)

# Compute ROC curve and AUC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels[:, i], scor[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), scor.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#Target dictionary from pandas script
target_dict = {0: 'B',
                1: 'C',
                2: 'LB',
                3: 'LP',
                4: 'P',
                5: 'VUS'}

# Finally average it and compute AUC
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize= (10, 10 ))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "yellow", "purple"])
for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw= 2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(target_dict[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw= 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristics, accuracy test: %s"%(accuracy_test))
plt.legend(loc="lower right", prop={'size': 10})
plt.savefig('/home/paux/FMP/NN/ROC_curve/ROC_nn%s_ep%s_lr%s_bs%s_acc%s.jpg' % (num_neurons, num_epoch, learning_rate, batch_size, accuracy_test))

#Create and plot confusion matrix
y_pred = []
y_true = []

#Iterate over test data
for inputs, labels in test_load:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

#To translate the number to its label
classes = target_dict.values()

#Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i in classes],
                    columns = [i for i in classes])
#Plot CM and save it
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title('Confusion matrix, accuracy test: %s'%(accuracy_test))
plt.savefig('/home/paux/FMP/NN/confusion_matrix/conf_mx_nn%s_ep%s_lr%s_bs%s_acc%s.jpg' % (num_neurons, num_epoch, learning_rate, batch_size, accuracy_test))

#Rename Loss graph to include accuracy
os.rename('/home/paux/FMP/NN/loss_graphs/loss_nn%s_ep%s_lr%s_bs%s.jpg' % (num_neurons, num_epoch, learning_rate, batch_size)
, '/home/paux/FMP/NN/loss_graphs/loss_nn%s_ep%s_lr%s_bs%s_acc%s.jpg' % (num_neurons, num_epoch, learning_rate, batch_size,accuracy_test))




# %%
