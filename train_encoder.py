import argparse
import json
import os
import shutil

import torch
import torch.nn as nn

import numpy as np

from torchvision import transforms
from torchvision.datasets import coco
from tqdm.auto import tqdm

from model import EncoderCNN


TRAIN_DIR_IMGS = "./train2017/"
TRAIN_DIR_CAPTIONS = "./annotations/captions_train2017.json"
DATA_DIR = "./data/"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable-cuda", action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    preprocess = transforms.Compose((
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ))


    coco_train = coco.CocoCaptions(
        TRAIN_DIR_IMGS,
        TRAIN_DIR_CAPTIONS,
        transform=preprocess
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=coco_train,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4
    )

    model = EncoderCNN()
    model = nn.DataParallel(model.train(False).to(args.device))

    vectors, captions = [], []
    with torch.no_grad():
        for img_batch, capt_batch in tqdm(data_loader):
            capt_batch = list(zip(*capt_batch))
            vec_batch = model(img_batch)

            captions.extend(capt_batch)
            vectors.extend([vec for vec in vec_batch])

    captions_tokenized = list([[caption.lower() for caption in caption_list] 
                                for caption_list in captions])

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    else:
        shutil.rmtree(DATA_DIR)

    np.save(DATA_DIR + "image_codes.npy", np.asarray(vectors))
    with open(DATA_DIR + "captions_tokenized.json", "w") as file_capt:
        json.dump(captions_tokenized, file_capt)

if __name__ == "__main__":
    main()