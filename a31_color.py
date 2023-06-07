""" 
A31
Меню «Изображение» Настроить:  
    Цвет, Фильтровать цвет, Извлечь цвет
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Цвет, Фильтровать цвет, Извлечь цвет
    red: int = None  # [-128, 128]
    green: int = None  # [-128, 128]
    blue: int = None  # [-128, 128]


def colors(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    # do the thing
    (B, G, R) = cv.split(ocv_img)
    R = cv.convertScaleAbs(R, alpha=1.0, beta=dto.red)
    G = cv.convertScaleAbs(G, alpha=1.0, beta=dto.green)
    B = cv.convertScaleAbs(B, alpha=1.0, beta=dto.blue)
    res = cv.merge([B, G, R])

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
        cv.createTrackbar("red", win_name, 0, 128, empty)
        cv.setTrackbarMin("red", win_name, -128)
        cv.createTrackbar("green", win_name, 0, 128, empty)
        cv.setTrackbarMin("green", win_name, -128)
        cv.createTrackbar("blue", win_name, 0, 128, empty)
        cv.setTrackbarMin("blue", win_name, -128)

        dto = DTO
        dto.img_in = img
        while True:
            dto.red = cv.getTrackbarPos("red", win_name)
            dto.green = cv.getTrackbarPos("green", win_name)
            dto.blue = cv.getTrackbarPos("blue", win_name)
            res_dto = colors(dto)

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
