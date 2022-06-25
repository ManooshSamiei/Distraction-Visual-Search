from git import Repo
import os
import wget
import zipfile
import gdown
import argparse


def git_maskRCNN(save_dir):
    """clones Mask-RCNN repository.
    args:
        save_dir (str): The path to save the respository.
    """

    repo_folder_path = os.path.join(save_dir, 'Mask_RCNN')

    if not os.path.exists(repo_folder_path):
        Repo.clone_from('https://github.com/matterport/Mask_RCNN.git', repo_folder_path)
    else:
        return


def unzip(zip_path, extract_path):
    """extracts the files in a zip file
       in the specified directory
    args:
        zip_path (str): the path to the zip file
        extract_path (str): the path to save the extracted files
    """

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

    os.makedirs(data_path, exist_ok=True)

    print(">> Cloning MaskRCNN Repository...", end="", flush=True)
    git_maskRCNN(data_path)

    print(">> Downloading COCOSearch18 dataset...", end="", flush=True)
    urls = ['http://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip',
            'https://saliency.tuebingen.ai/data/coco_search18_TP.zip']
    for url in urls:
        filename = wget.download(url, data_path)
        unzip(filename, data_path)

    print(">> Downloading Segmentation Annotations...", end="", flush=True)
    url = "https://drive.google.com/uc?export=download&id=1ri6IToZzj9FUcXCK3PHXRhslXhwV4rHV"
    gdown.download(url, os.path.join(data_path, 'annotations.zip'), quiet=False)
    unzip(os.path.join(data_path, 'annotations.zip'), data_path)

    print(">> Downloading target object bounding box annotation...", end="", flush=True)
    url = "https://drive.google.com/uc?id=1OkpX_Md-lFwCo5TB_cq0Qxoe4oEB8eKG"
    output_path = os.path.join(data_path, 'bbox_annos.npy')
    gdown.download(url, output_path, quiet=False)
    print("done!", flush=True)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dldir', type=str, required=True, help='The directory to download the dataset.',
                        default='../')
    args = parser.parse_args()

    download_cocosearch(args.dldir)