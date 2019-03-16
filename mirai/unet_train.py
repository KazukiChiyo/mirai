import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from backend import UNet, Train, Test
from data import CamVid, PILToLongTensor
from data.utils import median_freq_balancing
import torchvision.transforms as transforms

root_dir = './CamVid/'
batch_size = 8
lr = 8e-5
n_epoches = 50
lr_decay = 0.75
lr_decay_epoches = 50
l2_decay = 5e-4

def checkpoint(name, epoch, avg_loss):
    model_out_path = "./run2/{}_epoch_{}_loss_{:.4f}.pth".format(name, epoch, avg_loss)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('No GPU found, running on CPU')
        device = torch.device("cpu")
        pin_memory = False
    else:
        device = torch.cuda.current_device()
        print('Using ' + torch.cuda.get_device_name(device))
        pin_memory = True

    image_transform = transforms.Compose([transforms.ToTensor()])
    label_transform = transforms.Compose([PILToLongTensor()])

    print('--- Loading datasets ---')
    train_dataset = CamVid(root_dir=root_dir, mode='train', transform=image_transform, label_transform=label_transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    valid_dataset = CamVid(root_dir=root_dir, mode='val', transform=image_transform, label_transform=label_transform)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    class_encoding = train_dataset.color_encoding
    del class_encoding['road_marking']
    num_classes = len(class_encoding)
    class_weights = median_freq_balancing(train_loader, num_classes)
    class_weights = torch.from_numpy(class_weights).float().to(device)

    print('--- Building model ---')
    model = UNet(in_channels=3, out_channels=num_classes, feature_scale=1).to(device)
    model_name = 'UNet'
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoches, gamma=lr_decay)

    train_obj = Train(model=model, data_loader=train_loader, optim=optimizer, criterion=criterion, device=device)
    valid_obj = Test(model=model, data_loader=valid_loader, criterion=criterion, device=device)
    for epoch in range(1, n_epoches + 1):
        scheduler.step()
        train_loss = train_obj.run_epoch()
        print(">>>> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, train_loss))

        if epoch%5 == 0:
            valid_loss = valid_obj.run_epoch()
            print(">>>> Validation: Avg. Loss: {:.4f}".format(valid_loss))
            checkpoint(model_name, epoch, valid_loss)
