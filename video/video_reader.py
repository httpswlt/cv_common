# -*- coding: utf-8 -*-
import cv2
import numpy as np


class VideoReader:
    def __init__(self, video_path='', abs_diff=False, frame_interval=0):
        """

        :param video_path: type: str, the path of video
        """
        self.video_path = video_path
        self.abs_diff = abs_diff
        self.frame_interval = frame_interval

        self.cap = None
        if self.video_path != '':
            self.__init_parameters()
            self.__curr_frame = -1

    def __init_parameters(self):
        """

        :return:
        """

        self.cap = cv2.VideoCapture(self.video_path)
        assert self.cap.isOpened(), print(self.video_path)

        self.__curr_frame = -1
        self.__pre_frame = -1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dark_threshold = 50
        self.light_threshold = 240
        self.pre_img = None
        self.pre_img_pixel_diff = 0
        self.pixel_threshold = 40

    def get_total_frames(self):
        """

        :return:
        """
        return self.total_frames

    def get_fps(self):
        """

        :return:
        """
        return self.fps

    def get_resolution(self):
        """

        :return:
        """
        return int(self.height), int(self.width)

    def enable_abs_diff(self):
        """

        :return:
        """
        self.abs_diff = True
        self.frame_interval = 0

    def set_pixel_threshold(self, pixel_threshold):
        self.pixel_threshold = pixel_threshold

    def set_total_frames(self, total_frames):
        """

        :param total_frames:
        :return:
        """
        self.total_frames = total_frames

    def set_frame_interval(self, frame_interval):
        """

        :param frame_interval: how many frames interval when extract the frame from video.
        :return:
        """
        self.frame_interval = frame_interval

    def reset_video_path(self, video_path):
        """

        :param video_path: type: str, the path of video
        :return:
        """
        self.release()
        self.video_path = video_path
        self.__init_parameters()

    def get_current_frame(self):
        """

        :return:
        """
        return self.__curr_frame

    def __read_video(self):
        """

        :return:
        """
        ret, frame = self.cap.read()
        self.__curr_frame += 1
        return ret, frame

    def items_frame(self):
        """

        :return:
        """
        if self.__curr_frame >= self.total_frames:
            return False, None
        ret, frame = self.__read_video()
        return ret, frame

    def items_key_frame(self):
        """

        :return:
        """
        ret, curr_img = self.items_frame()

        if self.abs_diff:
            if curr_img is None:
                return ret, curr_img

            curr_img_avg = np.average(curr_img)

            if curr_img_avg < self.dark_threshold or curr_img_avg > self.light_threshold:
                curr_img = None
                return ret, curr_img

            curr_img = self._abs_diff(curr_img)

        else:
            curr_img = self._frame_interval(self.frame_interval, curr_img)

        return ret, curr_img

    def _abs_diff(self, curr_img):
        """

        :param curr_img:
        :return:
        """
        if self.pre_img is None:
            # first image
            self.pre_img = curr_img
            return curr_img
        else:
            diff = np.average(cv2.absdiff(curr_img, self.pre_img))
            if diff > self.pixel_threshold:
                # self.pre_img_pixel_diff = diff
                self.pre_img = curr_img
            else:
                curr_img = None
            # print('diff: {}'.format(diff))
            return curr_img

    def _frame_interval(self, frame_interval, curr_img):
        """

        :param frame_interval:
        :param curr_img:
        :return:
        """
        if self.__curr_frame == 0 or self.__curr_frame - self.__pre_frame >= frame_interval + 1:
            self.__pre_frame = self.__curr_frame
        else:
            curr_img = None
        return curr_img

    def get_img_by_frame(self, frame_num):
        """

        :param frame_num:
        :return:
        """
        if frame_num >= self.total_frames:
            return None
        self.__curr_frame = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.__curr_frame)
        ret, frame = self.__read_video()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame if ret else None

    def release(self):
        """

        :return:
        """
        if self.cap is not None:
            self.cap.release()


def main():
    """

    :return:
    """
    video_var = VideoReader('/mnt/data/sample_comparison/catchVideo/test5.mp4', abs_diff=False)
    # for i in range(100):
    #     print('i: {}'.format(i))
    #     ret, frame1 = video_var.items_key_frame()
    #     if frame1 is None:
    #         continue
    #     cv2.imwrite('./test/test_{}.jpg'.format(i), frame1)
    # frame = video_var.get_img_by_frame()
    # cv2.imwrite('./test/test.jpg', frame1)

    video_var.release()


if __name__ == '__main__':
    main()
