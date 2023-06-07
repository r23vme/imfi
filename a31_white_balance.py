""" 
A31
Меню «Изображение» Настроить:  
    Баланс белого
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # White balance
    white_balance: int = None  # [-255, 255]


def white_balance(dto: DTO) -> DTO:
    # convert
    ocv_img = pil_to_ocv(dto.img_in)
    img_LAB = cv.cvtColor(ocv_img, cv.COLOR_BGR2LAB)

    # do the thing
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - (
        (avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * dto.white_balance
    )
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - (
        (avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * dto.white_balance
    )

    # convert back
    res = cv.cvtColor(img_LAB, cv.COLOR_LAB2BGR)
    dto.img_out = ocv_to_pil(res)

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
        cv.createTrackbar("white_balance", win_name, 0, 255, empty)
        cv.setTrackbarMin("white_balance", win_name, -255)
        dto = DTO
        dto.img_in = img
        while True:
            dto.white_balance = cv.getTrackbarPos("white_balance", win_name) / 100
            res_dto = white_balance(dto)

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
