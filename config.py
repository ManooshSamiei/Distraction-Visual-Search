"""General training parameters that define the maximum number of
training epochs, the batch size, and learning rate for the ADAM
optimization method. To reproduce the results from the paper,
these values should not be changed. The device can be either
"cpu" or "gpu", which then optimizes the model accordingly after
training or uses the correct version for inference when testing.
"""

PARAMS = {
    "n_epochs": 25,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "device": "gpu"
}

"""The predefined input image sizes for each of the 3 datasets.
To reproduce the results from the paper, these values should
not be changed. They must be divisible by 8 due to the model's
downsampling operations. Furthermore, all pretrained models
for download were trained on these image dimensions.
"""

DIMS = {
    "image_size_cocosearch": (512, 320),
    "image_target_size_cocosearch": (64, 64)
}
