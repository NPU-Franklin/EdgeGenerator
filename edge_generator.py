import glob
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def edge_generator(mask, thickness: int = 5):
    shrink_operator = np.ones_like(mask)
    edge = np.zeros_like(mask)
    shrink_mask = mask
    for i in range(thickness):
        contours, hierarchy = cv2.findContours(shrink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(shrink_operator, contours, -1, 0, 1)
        cv2.drawContours(edge, contours, -1, 255, 1)
        shrink_mask *= shrink_operator
    return edge


if __name__ == "__main__":
    train_masks = glob.glob("./input/WarwickQU/train/Mask/*.png")
    test_masks = glob.glob("./input/WarwickQU/test/Mask/*.png")
    train_edge_dir = "./input/WarwickQU/train/Edge"
    test_edge_dir = "./input/WarwickQU/test/Edge"

    for train_mask in tqdm(train_masks, unit="img(s)"):
        mask = cv2.imread(train_mask, cv2.IMREAD_GRAYSCALE)
        edge = edge_generator(mask)
        cv2.imwrite(os.path.join(train_edge_dir, train_mask.split("\\")[-1]), edge)

    for test_mask in tqdm(test_masks, unit="img(s)"):
        mask = cv2.imread(test_mask, cv2.IMREAD_GRAYSCALE)
        edge = edge_generator(mask)
        cv2.imwrite(os.path.join(test_edge_dir, test_mask.split("\\")[-1]), edge)

    logging.info("Complete!")
