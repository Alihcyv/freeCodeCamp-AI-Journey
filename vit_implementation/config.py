class Config:
    BATCH_SIZE = 128
    IMAGE_SIZE = 32
    CHANNELS = 3
    NUM_CLASSES = 10
    
    PATCH_SIZE = 4
    EMBED_DIM = 256
    DEPTH = 6
    NUM_HEADS = 8
    MLP_DIM = 512
    DROP_RATE = 0.1
    
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    SEED = 42
    DEVICE = 'cuda'
