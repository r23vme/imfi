""" 
A44
Меню «Обработка»  
    Удаление шумов
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # denoise
    strength: int = None
    tw_size: int = None
    sw_size: int = None


def denoise(dto: DTO) -> DTO:
    # do the thing
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    tw_size, sw_size = dto.tw_size, dto.sw_size
    if (dto.tw_size % 2) == 0:
        tw_size += 1
    if (dto.sw_size % 2) == 0:
        sw_size += 1

    res = cv.fastNlMeansDenoisingColored(
        ocv_img, None, dto.strength, dto.strength, tw_size, sw_size
    )

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
        dto = DTO
        dto.img_in = img

        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)

        cv.createTrackbar("strength", win_name, 10, 255, empty)
        cv.createTrackbar("tw_size", win_name, 7, 255, empty)
        cv.createTrackbar("sw_size", win_name, 15, 255, empty)

        while True:
            dto.strength = cv.getTrackbarPos("strength", win_name)
            dto.tw_size = cv.getTrackbarPos("tw_size", win_name)
            dto.sw_size = cv.getTrackbarPos("sw_size", win_name)

            res_dto = denoise(dto)

            ocv_img = pil_to_ocv(res_dto.img_out)

            cv.imshow(win_name, ocv_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return res_dto

    pil_image = Image.open("img_in_noisy.jpg")
    res_dto = show(pil_image)

    # convert
    ocv_img = pil_to_ocv(res_dto.img_out)

    cv.imwrite("img_out.jpg", ocv_img)
