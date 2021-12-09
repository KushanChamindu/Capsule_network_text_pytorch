from data_load import get_data_set
from Net import ExtractionNet
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


X_train,y_train, X_test,y_test = get_data_set()

X_train,y_train, X_test,y_test  = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test,y_test)
batch_size = 8
test_dl = DataLoader(test_ds, batch_size, shuffle=True)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
model = ExtractionNet(word_embed_dim=300, output_size=4, hidden_size=128,
                      capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)

loss_fn =torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.NLLLoss()
# loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam
# count = 0
# for xb,yb in train_dl:
#     count+=1
#     # print(model(xb.to(torch.long)).size())
#     # print(xb.size(), yb.size())
#     yb = yb.argmax(-1)
#     output = model(xb.to(torch.long))
#     print("model output size - ",output.size())
#     print("label_size - ", yb.size())
#     print(yb)
#     print(yb.argmax(-1))
#     loss = loss_fn((output), yb)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print(loss)
#     if count ==2: break
    

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt_fn, train_dl, test_dl):
    lost_list = []
    opt = opt_fn(model.parameters(), lr=0.002, betas=(0.7,0.999),weight_decay=1e-5)
    model.train()
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        count = 0
        train_acc, correct_train, train_loss, target_count = 0, 0, 0, 0
        # Train with batches of data
        for xb,yb in train_dl:
            xb = Variable(xb.long())
            yb = Variable(yb.float())
            y_b = yb.argmax(-1)
            # a = list(model.parameters())[1].clone()
            
            # 1. Generate predictions
            pred = model(xb)
            # print(torch.log(pred))
            # print(y_b)
            # print(pred.size())
            # # 2. Calculate loss
            loss = loss_fn(torch.log(pred), y_b)
            # print(loss)

            # print(loss)
            lost_list.append(loss)
            train_loss +=loss.item()
            
            ## 3. Compute gradients
            loss.backward()
            # print(list(model.parameters())[0])
            
            # # 4. Update parameters using gradients
            opt.step()
            # b = list(model.parameters())[1].clone()
            # print(torch.equal(a.data, b.data))
            # print("from train.py - ",list(model.parameters())[-2].size())
            # print(list(model.parameters())[3])
            ## 5. Reset the gradients to zero
            opt.zero_grad()
        #     if(count ==2): break
        #     count= count+1
        # if(count ==2): break
            # accuracy
            # _, predicted = torch.max(pred.data, 1)
            predicted = pred.argmax(-1)
            # print(predicted)
            # print(y_b)
            target_count += y_b.size(0)
            correct_train += (y_b == predicted).sum().item()
            train_acc = (100 * correct_train) / target_count
            train_loss = train_loss / target_count
        val_acc, val_loss = validate(model=model,criterion= loss_fn,val_loader=test_dl)
        print("Epoch {0}: train_acc {1} \t train_loss {2} \t val_acc {3} \t val_loss {4}".format(epoch, train_acc, train_loss, val_acc, val_loss))
        # Print the progress
        # if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print("Lowerest lost -  {:.4f}".format(torch.min(loss)))
def validate(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        val_acc, correct_val, val_loss, target_count = 0, 0, 0, 0
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda()
            input_var = Variable(input.long())
            target_var = Variable(target.float())
            target_var = target_var.argmax(-1)
            output = model(input_var)
            loss = criterion(torch.log(output), target_var)
            val_loss += loss.item()

            # accuracy
            predicted = output.argmax(-1)
            target_count += target_var.size(0)
            correct_val += (target_var == predicted).sum().item()
            val_acc = 100 * correct_val / target_count
        return (val_acc * 100) / target_count, val_loss / target_count
fit(50, model=model,loss_fn=loss_fn,opt_fn=opt,train_dl=train_dl, test_dl=test_dl)

print(model)
