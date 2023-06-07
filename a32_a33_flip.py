""" 
A32
Меню «Изображение» 
    Отразить горизонтально
    Отразить вертикально
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Flip
    axis: int = None  # [-1, 0, 1] - [both, x, y]


def flip(dto: DTO) -> DTO:
    # do the thing
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    res = cv.flip(ocv_img, dto.axis)
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
        cv.createTrackbar("axis", win_name, 0, 1, empty)
        cv.setTrackbarMin("axis", win_name, -1)
        dto = DTO
        dto.img_in = img
        while True:
            dto.axis = cv.getTrackbarPos("axis", win_name)
            res_dto = flip(dto)

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
