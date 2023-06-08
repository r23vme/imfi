""" 
A30
Меню «Изображение» Настроить:  
    Режим отображения: 
        Глубина цветности (выбор: 24 бит, 8 бит, 4 бит, 1 бит)
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Глубина цветности (выбор: 24 бит, 8 бит, 4 бит, 1 бит)
    depth: int = None  # [0,1,2,3] for [1,4,8,24] bits


def color_depth(dto: DTO) -> DTO:
    # do the thing
    if dto.depth == 0:  # 1 bit - black&white. = 1 layer of 1 bits
        res = dto.img_in.convert("1").convert("RGB")
    elif dto.depth == 1:  # 4 bit - 16 colors. = 1 layer of 4 bits
        res = (
            dto.img_in.convert("L")
            .convert(mode="P", palette=Image.ADAPTIVE, colors=16)
            .convert("RGB")
        )
    elif dto.depth == 2:  # 8 bit - grayscale. = 1 layer of 8 bits
        res = dto.img_in.convert("L").convert("RGB")
    elif dto.depth == 3:  # 24 bit - rgba. = 3 layers of 8 bits
        res = dto.img_in.convert("RGB")
    else:
        res = dto.img_in

    dto.img_out = res

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
        cv.createTrackbar("depth", win_name, 0, 3, empty)

        dto = DTO
        dto.img_in = img
        while True:
            dto.depth = cv.getTrackbarPos("depth", win_name)
            res_dto = color_depth(dto)

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
