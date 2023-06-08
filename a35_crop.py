""" 
A35
Меню «Изображение» 
    Обрезать
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Crop
    # (0,0) as top left corner of image called im with left-to-right as x direction and top-to-bottom as y direction.
    # (x1,y1) as the top-left vertex
    # (x2,y2) as the bottom-right vertex of a rectangle region within that image
    y1: int = None  # left border . 0 <= y1 < y2
    y2: int = None  # right border . y1 < y2 <= max_y of img_in
    x1: int = None  # top border . 0 <= x1 < x2
    x2: int = None  # bottom border . x1 < x2 <= max_x of img_in


def crop(dto: DTO) -> DTO:
    # do the thing
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    res = ocv_img[dto.y1 : dto.y2, dto.x1 : dto.x2]

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
        dto = DTO
        dto.img_in = img

        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)

        ph = pil_to_ocv(dto.img_in)
        y, x, _ = ph.shape

        cv.createTrackbar("y1", win_name, 0, y, empty)
        cv.createTrackbar("y2", win_name, y, y, empty)
        cv.createTrackbar("x1", win_name, 0, x, empty)
        cv.createTrackbar("x2", win_name, x, x, empty)
        while True:
            dto.y1 = cv.getTrackbarPos("y1", win_name)
            dto.y2 = cv.getTrackbarPos("y2", win_name)
            dto.x1 = cv.getTrackbarPos("x1", win_name)
            dto.x2 = cv.getTrackbarPos("x2", win_name)

            if (dto.y1 < dto.y2) & (dto.x1 < dto.x2):
                res_dto = crop(dto)

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
