"""Image preprocessing and augmentation transforms."""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, center_crop: int = 224) -> dict:
    """Get train and test transforms for MVTec AD images.

    Args:
        image_size: Size to resize images to before cropping.
        center_crop: Size of the center crop.

    Returns:
        Dict with 'image' and 'mask' transform pipelines.
    """
    image_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
    ])

    return {"image": image_transform, "mask": mask_transform}
