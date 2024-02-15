import torchvision.transforms as transforms

# transformation of images, could be changed according to our needs

def get_transforms():
    transforms_to_image = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(0, 1)
    ])
    return transforms_to_image