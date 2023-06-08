""" 
A39
Меню «Изображение» 
    DPI (преобразование изображения)
    save file with provided fname & dpi

    to check dpi of an image(imagemagick):
    $ identify -format '%x,%y\n' img_out_dpi.jpg
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Глубина цветности (выбор: 24 бит, 8 бит, 4 бит, 1 бит)
    dpi_x: int = None
    dpi_y: int = None
    fname: str = "./img_out_dpi.jpg"
    quality: int = 95


def save_with_dpi(dto: DTO) -> DTO:
    dto.img_in.save(dto.fname, dpi=(dto.dpi_x, dto.dpi_y), quality=dto.quality)


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

        cv.createTrackbar("dpi_x", win_name, 900, 1200, empty)
        cv.createTrackbar("dpi_y", win_name, 900, 1200, empty)

        dto = DTO
        dto.img_in = img
        while True:
            dto.dpi_x = cv.getTrackbarPos("dpi_x", win_name)
            dto.dpi_y = cv.getTrackbarPos("dpi_y", win_name)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        res_dto = save_with_dpi(dto)
        cv.destroyAllWindows()
        return res_dto

    pil_image = Image.open("img_in.jpg")
    res_dto = show(pil_image)
