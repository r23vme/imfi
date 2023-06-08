""" 
A52
Меню «Обработка»
    Создание мультиконтрастного изображения (focus-stacking)
"""

import cv2 as cv
import numpy as np
from PIL import Image
from typing import List


class DTO:
    img_in_list: List[any] = None
    img_out: Image = None
    # multicontrast
    lks: int = None  # > 0 and odd
    gks: int = None  # > 0 and odd


def multicontrast(dto: DTO) -> DTO:
    # do the thing
    ocv_imgs = [pil_to_ocv(img) for img in dto.img_in_list]

    if (dto.lks % 2) == 0:
        dto.lks += 1
    if (dto.gks % 2) == 0:
        dto.gks += 1

    images = _align_images(ocv_imgs)
    laplacian = _compute_laplacian(images, dto)
    res = _find_focus_regions(images, laplacian)

    dto.img_out = ocv_to_pil(res)

    return dto


def _align_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """Align the images.  Changing the focus on a lens, even if the camera remains fixed,
        causes a mild zooming on the images. We need to correct the images so they line up perfectly on top
    of each other.

    Args:
        images: list of image data
    """

    def _find_homography(
        _img1_key_points: np.ndarray, _image_2_kp: np.ndarray, _matches: List
    ):
        image_1_points = np.zeros((len(_matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(_matches), 1, 2), dtype=np.float32)

        for j in range(0, len(_matches)):
            image_1_points[j] = _img1_key_points[_matches[j].queryIdx].pt
            image_2_points[j] = _image_2_kp[_matches[j].trainIdx].pt

        homography, mask = cv.findHomography(
            image_1_points, image_2_points, cv.RANSAC, ransacReprojThreshold=2.0
        )

        return homography

    aligned_imgs = []

    detector = cv.SIFT_create()

    # Assume that image 0 is the "base" image and align all the following images to it
    aligned_imgs.append(images[0])
    img0_gray = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
    img1_key_points, image1_desc = detector.detectAndCompute(img0_gray, None)

    for i in range(1, len(images)):
        img_i_key_points, image_i_desc = detector.detectAndCompute(images[i], None)

        bf = cv.BFMatcher()
        # This returns the top two matches for each feature point (list of list)
        pair_matches = bf.knnMatch(image_i_desc, image1_desc, k=2)
        raw_matches = []
        for m, n in pair_matches:
            if m.distance < 0.7 * n.distance:
                raw_matches.append(m)

        sort_matches = sorted(raw_matches, key=lambda x: x.distance)
        matches = sort_matches[0:128]

        homography_matrix = _find_homography(img_i_key_points, img1_key_points, matches)
        aligned_img = cv.warpPerspective(
            images[i],
            homography_matrix,
            (images[i].shape[1], images[i].shape[0]),
            flags=cv.INTER_LINEAR,
        )

        aligned_imgs.append(aligned_img)

    return aligned_imgs


def _compute_laplacian(
    images: List[np.ndarray],
    dto: DTO,
) -> np.ndarray:
    """Gaussian blur and compute the gradient map of the image. This is proxy for finding the focus regions.

    Args:
        images: image data
    """
    laplacians = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(
            gray,
            (dto.gks, dto.gks),
            0,
        )
        laplacian_gradient = cv.Laplacian(blurred, cv.CV_64F, ksize=dto.lks)
        laplacians.append(laplacian_gradient)
    laplacians = np.asarray(laplacians)
    return laplacians


def _find_focus_regions(
    images: List[np.ndarray], laplacian_gradient: np.ndarray
) -> np.ndarray:
    """Take the absolute value of the Laplacian (2nd order gradient) of the Gaussian blur result.
    This will quantify the strength of the edges with respect to the size and strength of the kernel (focus regions).

    Then create a blank image, loop through each pixel and find the strongest edge in the LoG
    (i.e. the highest value in the image stack) and take the RGB value for that
    pixel from the corresponding image.

    Then for each pixel [x,y] in the output image, copy the pixel [x,y] from
    the input image which has the largest gradient [x,y]

    Args:
        images: list of image data to focus and stack.
        laplacian_gradient: the laplacian of the stack. This is the proxy for the focus region.
            Should be size: (len(images), images.shape[0], images.shape[1])

    Returns:
        np.array image data of focus stacked image, size of orignal image

    """
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)
    abs_laplacian = np.absolute(laplacian_gradient)
    maxima = abs_laplacian.max(axis=0)
    bool_mask = np.array(abs_laplacian == maxima)
    mask = bool_mask.astype(np.uint8)

    for i, img in enumerate(images):
        output = cv.bitwise_not(img, output, mask=mask[i])

    return 255 - output


# Helper: Convert img from OpenCV to PIL.Image
def ocv_to_pil(img: np.array) -> Image:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)


# Helper: Convert img from PIL.Image to OpenCV
def pil_to_ocv(img: Image) -> np.array:
    ocv_img = np.array(img)
    # return ocv_img[:, :, ::-1].copy()
    return ocv_img


# Just for testing
if __name__ == "__main__":

    def empty(*args):
        pass

    def show(images: List[any]):
        dto = DTO
        dto.img_in_list = images

        win_name = "Trackbars"
        cv.namedWindow(win_name)
        cv.resizeWindow(win_name, 500, 100)

        cv.createTrackbar("gks", win_name, 3, 100, empty)
        cv.createTrackbar("lks", win_name, 3, 100, empty)

        while True:
            dto.gks = cv.getTrackbarPos("gks", win_name)
            dto.lks = cv.getTrackbarPos("lks", win_name)

            res_dto = multicontrast(dto)

            ocv_img = pil_to_ocv(res_dto.img_out)

            cv.imshow(win_name, ocv_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv.destroyAllWindows()
        return res_dto

    image_files = [
        "001.png",
        "002.png",
        "003.png",
        "004.png",
        "005.png",
        "006.png",
        "007.png",
        "008.png",
        "009.png",
        "010.png",
    ]
    images = [Image.open(img) for img in image_files]
    res_dto = show(images)

    # convert
    ocv_img = pil_to_ocv(res_dto.img_out)

    cv.imwrite("multicontrast_img_out.jpg", ocv_img)
