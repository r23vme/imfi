""" 
A37
Меню «Изображение» 
    Гистограмма
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # histogram
    histogram: np.array = None
    histogram_img: Image = None


def histogram(dto: DTO) -> DTO:
    # do the thing
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    bins = np.arange(256).reshape(256, 1)

    # h = np.zeros((300, 256, 3)) # black background
    h = np.ones((300, 256, 3))  # white background
    if len(ocv_img.shape) == 2:
        color = [(255, 255, 255)]
    elif ocv_img.shape[2] == 3:
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for ch, col in enumerate(color):
        dto.histogram = cv.calcHist([ocv_img], [ch], None, [256], [0, 256])
        cv.normalize(dto.histogram, dto.histogram, 0, 255, cv.NORM_MINMAX)
        hist = np.int32(np.around(dto.histogram))
        pts = np.int32(np.column_stack((bins, hist)))
        cv.polylines(h, [pts], False, col)

    dto.histogram_img = np.flipud(h)

    dto.img_out = dto.img_in

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

        ph = pil_to_ocv(dto.img_in)
        y, x, _ = ph.shape

        cv.createTrackbar("img_in,img_out,histogram", win_name, 2, 2, empty)

        while True:
            to_show = cv.getTrackbarPos("img_in,img_out,histogram", win_name)

            res_dto = histogram(dto)

            if to_show == 0:
                ocv_img = pil_to_ocv(res_dto.img_in)
            elif to_show == 1:
                ocv_img = pil_to_ocv(res_dto.img_out)
            elif to_show == 2:
                ocv_img = pil_to_ocv(res_dto.histogram_img)

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
