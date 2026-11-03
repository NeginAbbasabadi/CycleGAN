import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from Models import GeneratorResNet

def style_transfer_joensuu_photo(image_path, generator):
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(img).unsqueeze(0)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    generator = generator.to(device)

    with torch.no_grad():
        generated_tensor = generator(img_tensor)

    generated_image = (generated_tensor.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
    generated_pil = transforms.ToPILImage()(generated_image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img.resize((256, 256)))
    ax[0].set_title("Original Joensuu Photo")
    ax[0].axis('off')

    ax[1].imshow(generated_pil)
    ax[1].set_title("Generated Art")
    ax[1].axis('off')
    plt.show()

if __name__ == "__main__":
    # Initialize the blank generator and load your saved weights
    G_A2B = GeneratorResNet()
    weights_path = './CycleGAN_Checkpoints/G_A2B_epoch_4.pth'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    G_A2B.load_state_dict(torch.load(weights_path, map_location=device))
    G_A2B.eval()

    # Run the style transfer
    image_path = '/Users/macbook/CycleGAN/Joensuu-avara_2.JPEG' 
    style_transfer_joensuu_photo(image_path, G_A2B)
