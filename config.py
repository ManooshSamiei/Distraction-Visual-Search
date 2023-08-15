"""General training parameters that define the maximum number of
training epochs, the batch size, and learning rate for the ADAM
optimization method. To reproduce the results from the paper,
these values should not be changed. The device can be either
"cpu" or "gpu", which then optimizes the model accordingly after
training or uses the correct version for inference when testing.
"""

PARAMS = {
    "n_epochs": 5,
    "batch_size": 1,
    "n_training_steps": 10000,
    "learning_rate": 1e-5,
    "learning_power": 0.5,
    "momentum": 0.9,
    "device": "gpu"
}

"""The predefined input image sizes for the search and target images.
To reproduce the results from the paper, these values should
not be changed. They must be divisible by 8 due to the model's
downsampling operations.
""" 

DIMS = {
    "image_size_cocosearch": (320, 512),
    "image_target_size_cocosearch": (64, 64)
}