""" 
A30
Меню «Изображение» Настроить:  
    Режим отображения: Градации серого с сохранением контраста
    ie grayscaled image with option to change brightness & contrast
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Яркость/контраст
    contrast: float = None  # [1.0-3.0]
    brightness: int = None  # [0-100]


def brightness_contrast_gray(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    # do the thing
    res = cv.convertScaleAbs(ocv_img, alpha=dto.contrast, beta=dto.brightness)
    res = grayscale(res)

    dto.img_out = ocv_to_pil(res)  # convert back
    return dto


def grayscale(img: np.array) -> np.array:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


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
        cv.createTrackbar("brightness", win_name, 0, 100, empty)
        cv.createTrackbar("contrast", win_name, 100, 300, empty)
        dto = DTO
        dto.img_in = img
        while True:
            dto.brightness = cv.getTrackbarPos("brightness", win_name)
            dto.contrast = cv.getTrackbarPos("contrast", win_name) / 100
            res_dto = brightness_contrast_gray(dto)

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
