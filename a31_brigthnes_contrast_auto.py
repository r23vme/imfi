""" 
A31
Меню «Изображение» Настроить:  
    Авто контраст
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Яркость/контраст
    clip_hist: int = None  # [1-100]
    contrast: float = None  # [1.0-3.0] # calculated automatically
    brightness: int = None  # [0-100] # calculated automatically


def brightness_contrast(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    # do the thing
    res = cv.convertScaleAbs(ocv_img, alpha=dto.contrast, beta=dto.brightness)
    dto.img_out = ocv_to_pil(res)  # convert back
    return dto


def brightness_contrast_auto(dto: DTO) -> DTO:
    ocv_img = pil_to_ocv(dto.img_in)  # convert
    gray = cv.cvtColor(ocv_img, cv.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    dto.clip_hist *= maximum / 100.0
    dto.clip_hist /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < dto.clip_hist:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - dto.clip_hist):
        maximum_gray -= 1

    # Calculate alpha and beta values
    dto.contrast = 255 / (maximum_gray - minimum_gray)
    dto.brightness = -minimum_gray * dto.contrast

    res = cv.convertScaleAbs(ocv_img, alpha=dto.contrast, beta=dto.brightness)
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
        cv.createTrackbar("cliphist", win_name, 10, 100, empty)
        cv.setTrackbarMin("cliphist", win_name, 1)
        dto = DTO
        dto.img_in = img
        while True:
            dto.clip_hist = cv.getTrackbarPos("cliphist", win_name)
            # res_dto = brightness_contrast(dto)
            res_dto = brightness_contrast_auto(dto)

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
