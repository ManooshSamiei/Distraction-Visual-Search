import os
import wget
import zipfile
import gdown


def unzip(zip_path, extract_path):
    """extracts the files in a zip file
    in the specified directory"""

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            zip_ref.extract(file, extract_path)

    os.remove(zip_path)

    return


def download_cocosearch(data_path):
    """Downloads the COCOSearch18 dataset. The dataset
       contains the image stimuli and label/annotation files.
    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading COCOSearch18 dataset...", end="", flush=True)

    os.makedirs(data_path, exist_ok=True)

    urls = ['http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip']# ,
    #'https://saliency.tuebingen.ai/data/coco_search18_TP.zip']

    for url in urls:
        filename = wget.download(url, data_path)
        unzip(filename, data_path)

    '''url = "https://drive.google.com/uc?export=download&id=1vEzgF54LPK2adlI7DdlXWGkYV76L-jjK"

    gdown.download(url, data_path + '/targets.zip', quiet=False)
    unzip(data_path + '/targets.zip', data_path)

    # Downloading target object bounding box annotation
    url = "https://drive.google.com/uc?id=1OkpX_Md-lFwCo5TB_cq0Qxoe4oEB8eKG"
    output_path = data_path + '/bbox_annos.npy'
    gdown.download(url, output_path, quiet=False)

    url = "https://drive.google.com/u/0/uc?export=download&confirm=ATmP&id=1ff0va472Xs1bvidCwRlW3Ctf7Hbyyn7p"
    weights_path = data_path + '/weights'
    os.makedirs(weights_path, exist_ok=True)
    gdown.download(url, weights_path + '/vgg16_hybrid.zip', quiet=False)
    unzip(weights_path + '/vgg16_hybrid.zip', weights_path)'''

    print("done!", flush=True)
    return
