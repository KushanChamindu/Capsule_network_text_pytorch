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

# loss_fn =torch.nn.CrossEntropyLoss
loss_fn = F.mse_loss
for xb,yb in train_dl:
    # print(model(xb.to(torch.long)).size())
    # print(xb.size(), yb.size())
    output = model(xb.to(torch.long))
    print("model output size - ",output.size())
    print("label_size - ", yb.size())
    loss = loss_fn((output), yb)
    print(loss)
    break



