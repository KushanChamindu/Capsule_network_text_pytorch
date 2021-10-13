from data_load import get_data_set
from Net import ExtractionNet
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


X_train,y_train, X_test,y_test = get_data_set()

X_train,y_train, X_test,y_test  = torch.from_numpy(X_train),torch.from_numpy(y_train),torch.from_numpy(X_test),torch.from_numpy(y_test)

train_ds = TensorDataset(X_train, y_train)
batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
model = ExtractionNet(word_embed_dim=300, output_size=4, hidden_size=128,
                      capsule_num=16, filter_ensemble_size=3, dropout_ratio=0.8, intermediate_size=(128, 8), sentence_length=30)

loss_fn =torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.NLLLoss()
# loss_fn = F.mse_loss
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
def fit(num_epochs, model, loss_fn, opt_fn, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        count = 0
        opt = opt_fn(model.parameters(), lr=0.002, betas=(0.7,0.999))
        model.train()
        # Train with batches of data
        for xb,yb in train_dl:
            yb = yb.argmax(-1)
            # a = list(model.parameters())[1].clone()
            
            # 1. Generate predictions
            pred = model(xb.to(torch.long))
            # print(pred.size())
            # # 2. Calculate loss
            loss = loss_fn(pred, yb)

            print(loss)
            
            # 3. Compute gradients
            loss.backward()
            # print(list(model.parameters())[0])
            
            # # 4. Update parameters using gradients
            opt.step()
            # b = list(model.parameters())[1].clone()
            # print(torch.equal(a.data, b.data))
            # print("from train.py - ",list(model.parameters())[-2].size())
            # print(list(model.parameters())[3])
            # 5. Reset the gradients to zero
            opt.zero_grad()
        #     if(count ==2): break
        #     count= count+1
        # if(count ==2): break
        
        
        # Print the progress
        # if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(5, model=model,loss_fn=loss_fn,opt_fn=opt,train_dl=train_dl)

print(model)
