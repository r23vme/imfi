""" 
A31
Меню «Изображение» Настроить:  
    Повысить детализацию
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # sharpening
    sharp_kernel: float = None  # >1
    detail: int = None  # [0,200]


def sharpen(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    # do the thing
    kernel = np.array(
        [
            [-dto.sharp_kernel, -dto.sharp_kernel, -dto.sharp_kernel],
            [-dto.sharp_kernel, 1 + dto.sharp_kernel * 8, -dto.sharp_kernel],
            [-dto.sharp_kernel, -dto.sharp_kernel, -dto.sharp_kernel],
        ]
    )
    res = cv.filter2D(ocv_img, -1, kernel)

    dto.img_out = ocv_to_pil(res)  # convert back
    return dto


def detail_enhance(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    res = cv.detailEnhance(ocv_img, None, dto.detail)

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
        cv.createTrackbar("kernel", win_name, 1, 1000, empty)
        cv.setTrackbarMin("kernel", win_name, 0)
        cv.createTrackbar("detail", win_name, 0, 200, empty)
        dto = DTO
        dto.img_in = img
        while True:
            dto.sharp_kernel = cv.getTrackbarPos("kernel", win_name) / 100
            dto.detail = cv.getTrackbarPos("detail", win_name)

            # res_dto = sharpen(dto)
            res_dto = detail_enhance(dto)

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
