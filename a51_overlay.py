""" 
A51
    Меню «Обработка»  
        Наложение изображения
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in1: Image = None  # background image
    img_in2: Image = None  # top image
    img_out: Image = None
    # overlay
    x: int = None  # of top image. offsets from left, top
    y: int = None
    scale: float = None
    alpha: int = None


def overlay(dto: DTO) -> DTO:
    back = dto.img_in1.copy()
    top = dto.img_in2
    top = top.resize([int(dto.scale * s) for s in top.size])

    ttop = top.copy()
    ttop.putalpha(dto.alpha)
    top.paste(ttop, top)

    back.paste(top, (dto.x, dto.y), top)

    dto.img_out = back

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

    def show(back: Image, top: Image):
        dto = DTO
        dto.img_in1 = back
        dto.img_in2 = top

        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)

        cv.createTrackbar("x", win_name, 10, 255, empty)
        cv.createTrackbar("y", win_name, 10, 255, empty)
        cv.createTrackbar("scale", win_name, 100, 1000, empty)
        cv.createTrackbar("alpha", win_name, 255, 255, empty)

        while True:
            dto.x = cv.getTrackbarPos("x", win_name)
            dto.y = cv.getTrackbarPos("y", win_name)
            dto.scale = cv.getTrackbarPos("scale", win_name) / 100
            dto.alpha = cv.getTrackbarPos("alpha", win_name)

            res_dto = overlay(dto)

            ocv_img = pil_to_ocv(res_dto.img_out)

            cv.imshow(win_name, ocv_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return res_dto

    back = Image.open("background.jpg")
    top = Image.open("top.png")
    res_dto = show(back, top)

    # convert
    ocv_img = pil_to_ocv(res_dto.img_out)

    cv.imwrite("img_out.jpg", ocv_img)
