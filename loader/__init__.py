from loader.cropped_dataset import CroppedDataset

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'cropped':croppedDataset
    }[name]
