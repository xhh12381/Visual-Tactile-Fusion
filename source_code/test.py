import torch
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from VTFNet import VTFNet
from dataset import LeatherDataset
from main import mean_iou, mean_pixel_accuracy
from deeplabv3plus import DeepLabV3Plus

if __name__ == '__main__':
    device = 'cuda:0'
    # net = VTFNet()
    net = DeepLabV3Plus(in_channel=3)
    # net = DeepLabV3Plus(in_channel=4)
    net.load_state_dict(torch.load('./checkpoints/deeplab_epoch=100.pth'))
    net.to(device)
    net.eval()
    train_set = LeatherDataset(
        visual_dir="./data/datasets/test",
        tactile_dir="./data/datasets/tactile_images",
        masks_dir="./data/datasets/ground_truth"
    )
    train_iter = DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    pbar = tqdm(train_iter, desc="Eval...")
    mious = []
    mpas = []
    for i, data in enumerate(pbar):
        mask = data["mask"].to(device)
        visual = data["visual"].to(device)
        tactile = data["tactile"].to(device)

        # output = net(visual, tactile)
        output = net(visual)
        # output = net(torch.cat((visual, tactile), dim=1))
        output = torch.softmax(output, dim=1)
        _, pred = torch.max(output, dim=1)
        miou = mean_iou(mask.detach().cpu().long().numpy(), pred.detach().cpu().long().numpy())
        mpa = mean_pixel_accuracy(mask.detach().cpu().long().numpy(), pred.detach().cpu().long().numpy())
        mious.append(miou)
        mpas.append(mpa)
        torchvision.utils.save_image(visual, f"results/{i}.png")
        torchvision.utils.save_image(tactile, f"results/{i}_tactile.png")
        torchvision.utils.save_image(mask.float().unsqueeze(dim=1), f"results/{i}_mask.png")
        torchvision.utils.save_image(pred.float(), f"results/{i}_pred.png")
    print(f"mean_iou: {sum(mious) / len(mious)}, mpa: {sum(mpas) / len(mpas)}")
