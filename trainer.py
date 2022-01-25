from data_load import get_data_set
from ensemble_model import MyEnsemble
from Net import ExtractionCapNet
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train,y_train, X_test,y_test = get_data_set()

X_train,y_train, X_test,y_test  = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test,y_test)
batch_size = 8
test_dl = DataLoader(test_ds, batch_size, shuffle=True)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = ExtractionCapNet(word_embed_dim=300,
                      capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)
# extract_net_2 = ExtractionCapNet(word_embed_dim=300, output_size=4, hidden_size=128,
#                       capsule_num=16, filter_ensemble_size=4, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)
# extract_net_3 = ExtractionCapNet(word_embed_dim=300, output_size=4, hidden_size=128,
#                       capsule_num=16, filter_ensemble_size=5, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)
# model = MyEnsemble(modelA=extract_net_1,modelB=extract_net_2, modelC=extract_net_3)

model.to(device)

loss_fn =torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.NLLLoss()
# loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.7,0.999),weight_decay=1e-5)   

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl, test_dl):
    model.train()
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        count = 0
        train_acc, correct_train, train_loss, target_count = 0, 0, 0, 0
        # Train with batches of data
        for xb,yb in train_dl:
            xb = Variable(xb.long()).to(device)
            yb = Variable(yb.float()).to(device)
            y_b = yb.argmax(-1)
            # a = list(model.parameters())[1].clone()
            opt.zero_grad()
            # 1. Generate predictions
            pred = model(xb)
            # print(pred.shape)
            # 2. Calculate loss
            loss = loss_fn(torch.log(pred), y_b)
            # print(loss)
            # 3. Compute gradients
            loss.backward()
            # print(list(model.parameters())[0])
            
            # 4. Update parameters using gradients
            opt.step()
           
            # accuracy
            train_loss +=loss.item()
            # _, predicted = torch.max(pred.data, 1)
            predicted = pred.argmax(-1)

            target_count += y_b.size(0)
            correct_train += (y_b == predicted).sum().item()
            train_acc = (100 * correct_train) / target_count
            train_loss = train_loss / target_count
        val_acc, val_loss = validate(model=model,criterion= loss_fn,val_loader=test_dl)
        print("Epoch {0}: train_acc {1} \t train_loss {2} \t val_acc {3} \t val_loss {4}".format(epoch, train_acc, train_loss, val_acc, val_loss))
        # Print the progress
        # if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

def validate(model, val_loader, criterion):
    model.eval()
    val_acc, correct_val, val_loss, target_count = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda()
            input_var = Variable(input.long()).to(device)
            target_var = Variable(target.float()).to(device)
            target_var = target_var.argmax(-1)
            output = model(input_var)
            loss = criterion(torch.log(output), target_var)
            val_loss += loss.item()

            # accuracy
            predicted = output.argmax(-1)
            target_count += target_var.size(0)
            correct_val += (target_var == predicted).double().sum().item()
        return (correct_val * 100) / target_count, val_loss / target_count
# val_acc, val_loss = validate(model=model,criterion= loss_fn,val_loader=test_dl)
# print(val_acc, val_loss)
fit(1, model=model,loss_fn=loss_fn,opt=opt,train_dl=train_dl, test_dl=test_dl)

print(model)
