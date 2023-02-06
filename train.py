import torch
from torch.utils.data import Dataset, DataLoader
import time
from tensorboardX import SummaryWriter

from model import Color 
from dataset import MY_DATA_SET
writer = SummaryWriter()

#HYPER PARAMETER
BATCH = 10
EPOCHS = 400

train_dataset = MY_DATA_SET()
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
criterion = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023)

model = Color(in_dim = 35000)
model.to(device)

def train(dataloader, epoch, count):
    model.train()

    for _, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        x, r, g, b, R, G, B = data
        count += len(x)
        
        R_out, G_out, B_out = model(x)
        R_pred = torch.argmax(R_out, dim=1)
        G_pred = torch.argmax(G_out, dim=1)
        B_pred = torch.argmax(B_out, dim=1)
        R_pred = torch.tensor(R_pred, dtype=torch.float32)
        G_pred = torch.tensor(G_pred, dtype=torch.float32)
        B_pred = torch.tensor(B_pred, dtype=torch.float32)
        
        loss_R = criterion(R_out, r)
        loss_G = criterion(G_out, g)
        loss_B = criterion(B_out, b)        
        loss = (loss_R + loss_G + loss_B) 
        writer.add_scalar('loss Cross Entropy', loss, count)
        loss_MSE = criterion2(torch.stack([R_pred,G_pred,B_pred]), torch.stack([R,G,B]))
        loss += loss_MSE
        loss.backward()
        writer.add_scalar('loss_MSE', loss_MSE, count)
        writer.add_scalar('Total Loss', loss, count)
        writer.add_scalar('epoch', epoch, count)
        optimizer.step()
    return count

optimizer = torch.optim.Adam(model.parameters())

count = 0

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    count = train(train_loader, epoch, count)
    print(optimizer.param_groups[0]['lr'])
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - epoch_start_time))
    print('-' * 59)
    
    # SAVE MODEL
    #if epoch % 20 == 0:
    #    torch.save(model.state_dict(), 'RGB__ep{}.pth'.format(epoch))
