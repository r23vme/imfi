""" 
A41
    Меню «Обработка»  
        Получение изображений с высоким динамическим диапазоном (HDR)

https://github.com/opencv/opencv/blob/3.4/samples/python/tutorial_code/photo/hdr_imaging/hdr_imaging.py

Merten - is the way! Debevec & Robertson provide worse result at greater computational cost.
"""

import os
import cv2 as cv
import numpy as np
from PIL import Image
from typing import List


class DTO:
    img_in_list: List[any] = None
    img_out: Image = None
    # hdr
    method: int = None  # [0,1,2] for [merten, debevec, robertson]
    exposures = None  # list of exposures
    gamma: float = None


def hdr(dto: DTO) -> DTO:
    ocv_imgs = [pil_to_ocv(img) for img in dto.img_in_list]  # convert

    # do the thing
    if dto.method == 0:
        merge_mertens = cv.createMergeMertens()
        res = merge_mertens.process(ocv_imgs)
    elif dto.method == 1:
        merge_debevec = cv.createMergeDebevec()
        hdr_debevec = merge_debevec.process(ocv_imgs, dto.exposures)
        tonemap = cv.createTonemap(gamma=dto.gamma)
        res = tonemap.process(hdr_debevec)
    elif dto.method == 2:
        merge_robertson = cv.createMergeRobertson()
        hdr_robertson = merge_robertson.process(ocv_imgs, times=dto.exposures)
        tonemap = cv.createTonemap(gamma=dto.gamma)
        res = tonemap.process(hdr_robertson)

    dto.img_out = ocv_to_pil(res)  # convert back
    return dto


# Helper: Convert img from OpenCV to PIL.Image
def ocv_to_pil(img: np.array) -> Image:
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # return Image.fromarray(img)
    img = cv.normalize(img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img))
    return img


# Helper: Convert img from PIL.Image to OpenCV
def pil_to_ocv(img: Image) -> np.array:
    ocv_img = np.array(img)
    return ocv_img[:, :, ::-1].copy()


# Helper: Import images and exposures
def loadExposureSeq(path):
    images = []
    times = []
    with open(os.path.join(path, "list.txt")) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        # images.append(cv.imread(os.path.join(path, tokens[0]))) # ocv format
        images.append(Image.open(os.path.join(path, tokens[0])))  # pil format
        times.append(1 / float(tokens[1]))

    return images, np.asarray(times, dtype=np.float32)


# Just for testing
if __name__ == "__main__":

    def empty(*args):
        pass

    def show(dto):
        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)

        # dto.img_in = img

        cv.createTrackbar("method", win_name, 0, 2, empty)
        cv.createTrackbar("gamma", win_name, 50, 100, empty)

        while True:
            dto.method = cv.getTrackbarPos("method", win_name)
            dto.gamma = cv.getTrackbarPos("gamma", win_name) / 10

            res_dto = hdr(dto)

            ocv_img = pil_to_ocv(res_dto.img_out)

            cv.imshow(win_name, ocv_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return res_dto

    path = "imgs/hdr"

    dto = DTO
    dto.img_in_list, dto.exposures = loadExposureSeq(path)

    res_dto = show(dto)

    # convert
    ocv_img = pil_to_ocv(res_dto.img_out)

    cv.imwrite("img_out.jpg", ocv_img)
