""" 
A31
Меню «Изображение» Настроить:  
    Авто уровень
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Авто уровень
    method: int = None  # [0, 2] , 0 - disabled, 1 - histogramm, 2 - clahe(the best one, should be default)


def colors_auto(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    # do the thing
    if dto.method == 1:
        (B, G, R) = cv.split(ocv_img)
        R = cv.equalizeHist(R)
        G = cv.equalizeHist(G)
        B = cv.equalizeHist(B)
        res = cv.merge([B, G, R])
    elif dto.method == 2:
        lab = cv.cvtColor(ocv_img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv.merge((l, a, b))
        res = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    else:
        res = ocv_img

    dto.img_out = ocv_to_pil(res)  # convert back
    return dto


# Helper: Convert img from OpenCV to PIL.Image
def ocv_to_pil(img: np.array) -> Image:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)


# Helper: Convert img from PIL.Image to OpenCV
def pil_to_ocv(img: Image) -> np.array:
    ocv_img = np.array(img)
    return ocv_img[:, :, ::-1].copy()


# Just for testing
if __name__ == "__main__":

    def empty(*args):
        pass

    def show(img: Image):
        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)
        cv.createTrackbar("method", win_name, 2, 2, empty)

        dto = DTO
        dto.img_in = img
        while True:
            dto.method = cv.getTrackbarPos("method", win_name)
            res_dto = colors_auto(dto)

            ocv_img = pil_to_ocv(res_dto.img_out)

            cv.imshow(win_name, ocv_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return res_dto

    pil_image = Image.open("img_in.jpg")
    res_dto = show(pil_image)

    # convert
    ocv_img = pil_to_ocv(res_dto.img_out)

    cv.imwrite("img_out.jpg", ocv_img)
