import torch
# torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()
BATCH_SIZE = 7 # increase / decrease according to GPU memeory
RESIZE_TO_WIDTH = 1920 # resize the image width for training and transforms
RESIZE_TO_HEIGHT = 1080 # resize the image height for training and transforms
NUM_EPOCHS = 90 # number of epochs to train for
NUM_WORKERS = 0

DEVICE = torch.device('cuda') #if torch.cuda.is_available() #else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'data/Endo/train'
# validation images and XML files directory
VALID_DIR = 'data/Endo/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'Adhesions.Dense', 'Adhesions.Filmy', 'Superficial.Black', 'Superficial.White',
    'Superficial.Red', 'Superficial.Subtle', 'Ovarian.Endometrioma[B]', 'Ovarian.Chocolate Fluid', 'Deep Endometriosis'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'