import torch
import torch.utils.data as data
from data import CamVid, PILToLongTensor, LongTensorToRGBPIL
import torchvision.transforms as transforms
from data.utils import batch_transform, imshow_batch

root_dir = './CamVid/'
batch_size = 4
ckpt_path = './run2/UNet_epoch_45_loss_1.0840.pth'

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
    test_dataset = CamVid(root_dir=root_dir, mode='test', transform=image_transform, label_transform=label_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    class_encoding = test_dataset.color_encoding
    del class_encoding['road_marking']
    num_classes = len(class_encoding)
    imgs, _ = iter(test_loader).next()
    print('--- Loading model ---')
    model = torch.load(ckpt_path).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(imgs.to(device))
    pred = torch.argmax(pred.data, 1)
    label_to_rgb = transforms.Compose([
        LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    img_pred = batch_transform(pred.cpu(), label_to_rgb)
    imshow_batch(imgs.data.cpu(), img_pred)
