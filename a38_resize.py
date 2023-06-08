""" 
A38
Меню «Изображение» 
    Разрешение (преобразование изображения)
"""

import cv2 as cv
import numpy as np
from PIL import Image


class DTO:
    img_in: Image = None
    img_out: Image = None
    # Resize
    width: int = None
    height: int = None
    interpolation: int = 0  # [0,6]
    use_scale: bool = False
    scale: int = None  # if given, use this instead. Percent of original image size


def resize(dto: DTO) -> DTO:
    # do the thing
    ocv_img = pil_to_ocv(dto.img_in)  # convert

    y, x, _ = ocv_img.shape
    if dto.use_scale:
        width = int(x * dto.scale / 100)
        height = int(y * dto.scale / 100)
    else:
        width = dto.width
        height = dto.height

    if width < 1:
        width = 1
    if height < 1:
        height = 1

    dsize = (width, height)

    res = cv.resize(ocv_img, dsize=dsize, interpolation=dto.interpolation)

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

        cv.createTrackbar("interpolation_method", win_name, 0, 6, empty)

        cv.createTrackbar("x", win_name, x, 1000, empty)
        cv.createTrackbar("y", win_name, y, 1000, empty)

        cv.createTrackbar("use_scale", win_name, 0, 1, empty)
        cv.createTrackbar("scale_percent", win_name, 100, 1000, empty)
        while True:
            dto.interpolation = cv.getTrackbarPos("interpolation_method", win_name)

            dto.width = cv.getTrackbarPos("x", win_name)
            dto.height = cv.getTrackbarPos("y", win_name)

            dto.use_scale = cv.getTrackbarPos("use_scale", win_name)
            dto.scale = cv.getTrackbarPos("scale_percent", win_name)

            res_dto = resize(dto)

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
