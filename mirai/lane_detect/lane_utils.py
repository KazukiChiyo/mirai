import cv2
import peakutils # Find peaks
import scipy.misc
import numpy as np
import matplotlib.image as mpimg
import time

class DetectLanes(object):
    def __init__(self, src, dst, n_windows=5, margin=40, minpix=20, color=(0, 255, 0), thickness=9, verbose=False):
        self.n_windows = n_windows
        self.margin = margin
        self.minpix = minpix
        self.color = color
        self.thickness = thickness
        self.verbose = verbose
        self.set_perspective(src, dst)

    def set_perspective(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.rev_M = cv2.getPerspectiveTransform(dst, src)

    def image_unwrap(self, image):
        warped = cv2.warpPerspective(image, self.M, (image.shape[1],image.shape[0]))
        return warped

    def hough_transform(self, image):
        rho, theta, threshold, min_line_len, max_line_gap = 10, np.pi/15, 30, 30, 10
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x2 != x1 and np.abs((y2-y1)/(x2-x1)) > 1.5:
                    cv2.line(line_image,(x1,y1),(x2,y2),1,5)
        return line_image

    def region_of_interest(self, binary_image):
        # Defining a blank mask to start with
        mask = np.zeros_like(binary_image)

        # Define mask and vertices
        ignore_mask_color = 1
        vertices = np.array([[(324,700),(136,620),(308,289),(1042,289),(1112,633),(940,700)]], dtype=np.int32)

        # Filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # Returning the image only where mask pixels are nonzero
        masked_binary = cv2.bitwise_and(binary_image, mask)
        return masked_binary

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(np.int32(binary_warped[:,:,0]), axis=0)

        peaks = peakutils.indexes(np.int32(histogram), thres=25, min_dist=200, thres_abs=True)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows

        # Create empty lists to receive left and right lane pixel indices
        x_array = []
        y_array = []

        for peak in peaks:
            # print(peak)
            lane_inds = []
            # Step through the windows one by one
            for window in range(self.n_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_x_low = peak - self.margin
                win_x_high = peak + self.margin

                # Draw the windows on the visualization image
                # cv2.rectangle(out_img,(win_x_low,win_y_low),
                # (win_x_high,win_y_high),(0,255,0), 5)

                # Identify the nonzero pixels in x and y within the window #
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

                # Append these indices to the list
                lane_inds.append(good_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > self.minpix:
                    peak = np.int(np.mean(nonzerox[good_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            lane_inds = np.concatenate(lane_inds)

            # Extract left and right line pixel positions
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds]

            # Save results to x, y arrays
            x_array.append(x)
            y_array.append(y)

        return x_array, y_array


    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
        x_array, y_array = self.find_lane_pixels(binary_warped)

        for x, y in zip(x_array,y_array):
            fit = np.polyfit(y, x, 2)
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            try:
                fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
            except TypeError:
                ('The function failed to fit a line!')
                fitx = 1*ploty**2 + 1*ploty

            cv2.polylines(img=out_img, pts=np.int32([np.stack((fitx, ploty), axis=-1)]), isClosed=False, color=self.color, thickness=self.thickness)

        return out_img


    def fit(self, image):
        start = time.time()
        unwrap= self.image_unwrap(image)
        gray = cv2.cvtColor(unwrap, cv2.COLOR_RGB2GRAY)
        adaptive_gray = cv2.adaptiveThreshold(gray, 1, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=19, C=-5)
        masked = self.region_of_interest(adaptive_gray)
        hough = self.hough_transform(masked)
        lines = self.fit_polynomial(hough)
        warp = cv2.warpPerspective(lines, self.rev_M, (image.shape[1], image.shape[0]))
        ret = cv2.addWeighted(image, 1, warp, 1, 0)

        if self.verbose:
            print("time per frame", time.time() - start)
            scipy.misc.imsave('./outputs/ipt.png', unwrap)
            scipy.misc.imsave('./outputs/gray.png', gray)
            scipy.misc.imsave('./outputs/adaptive_gray.png', adaptive_gray*255)
            scipy.misc.imsave('./outputs/masked.png', masked*255)
            scipy.misc.imsave('./outputs/hough.png', np.dstack((hough, hough, hough))*255)
            scipy.misc.imsave('outputs/lines.png', lines)
            scipy.misc.imsave('outputs/final.png', ret)

        return ret

def detect_lanes(image):
    src = np.float32([[575,464], [707,464], [258,682], [1049,682]])
    dst = np.float32([[450,0], [830,0], [450,720], [830,720]])
    obj = DetectLanes(src=src, dst=dst, verbose=True)
    return obj.fit(image)

if __name__ == '__main__':
    image = mpimg.imread('./extracted-1.0.jpg')
    detect_lanes(image)
