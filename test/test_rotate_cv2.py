# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/2/29 6:44
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmai.com
# ------------------------------------

import cv2
import numpy as np

"""
# rotate 1

src_pts = np.int0(box).astype("float32")
w, h = map(int, rect[1])
dst_pts = np.array([[0  ,   h-1],
                    [0  ,   0  ],
                    [w-1,   0  ],
                    [w-1,   h-1]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)   # the perspective transformation matrix
rotated_img = cv2.warpPerspective(img, M, (w, h))       # directly warp the rotated rectangle to get the straightened rectangle

"""



def main():
    img = cv2.imread("big_vertical_text.jpg")
    # points for test.jpg
    contour = np.array([
            [[64, 49]],
            [[122, 11]],
            [[391, 326]],
            [[308, 373]]
        ])
    print("shape of cnt: {}".format(contour.shape))

    rect = cv2.minAreaRect(contour)
    rot_mat = cv2.getRotationMatrix2D(rect[0], rect[2] if rect[2] > -45 else rect[2] + 90, 1)
    img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    cv2.imshow("crop_img", rotated_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()