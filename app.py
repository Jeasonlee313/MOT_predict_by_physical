import os
import argparse
import torch
import warnings
import cv2
import numpy as np

from utils.parser import get_config
from utils.log import get_logger
from utils.draw import draw_boxes
from utils.io import write_results
from deep_sort import build_tracker

from deep_sort.sort.kalman_filter import KalmanFilter as kf

h0 = np.array([[0.176138, 0.647589, -63.412272],
               [-0.180912, 0.622446, -0.125533],
               [-0.000002, 0.001756, 0.102316]])
h1 = np.array([[0.177291, 0.004724, 31.224545],
               [0.169895, 0.661935, -79.781865],
               [-0.000028, 0.001888, 0.054634]])
h2 = np.array([[-0.104843, 0.099275, 50.734500],
               [0.107082, 0.102216, 7.822562],
               [-0.000054, 0.001922, -0.068053]])
h3 = np.array([[-0.142865, 0.553150, -17.395045],
               [-0.125726, 0.039770, 75.937144],
               [-0.000011, 0.001780, 0.015675]])


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda, h=h3)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
        else:
            assert os.path.isfile(self.video_path), "Video path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()
            self.detections = np.loadtxt(self.args.detect_file, delimiter=',')

        if self.args.save_path:
            # os.makedirs(self.args.save_path, exist_ok=True)

            self.save_video_path = os.path.join(self.args.save_path, self.args.save_name + ".avi")
            self.save_results_path = os.path.join(self.args.save_path, self.args.save_name + ".txt")

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 25, (self.im_width, self.im_height))

            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(exc_type, exc_val, exc_tb)


    def run(self):
        results = []
        frame_ids = self.detections[:, 0]
        xywhs = self.detections[:, 2:6]
        xywhs[:, 0:2] += xywhs[:, 2:4] / 2
        confs = self.detections[:, 6]

        frame_id = 0
        self.vdo.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.vdo.grab():
            frame_id += 1
            _, ori_im = self.vdo.retrieve()
            # print("frame_id: ", frame_id, " ")
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            mask = frame_ids == frame_id
            xywh = xywhs[mask]
            conf = confs[mask]
            # print("frame_id: ", frame_id, " xywh", xywh.shape)
            outputs = self.deepsort.update(xywh, conf, im)

            if len(outputs) > 0:
                tlwh = []
                xyxys = outputs[:, :4]
                ids = outputs[:, -1]
                ori_im = draw_boxes(ori_im, xyxys, ids)

                for xyxy in xyxys:
                    tlwh.append(self.deepsort._xyxy_to_tlwh(xyxy))

                results.append((frame_id, tlwh, ids))

            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) == 27:
                    break

            if self.args.save_path:
                self.writer.write(ori_im)

        write_results(self.save_results_path, results, 'mot')

        # self.logger.info("time: {:.03f}s, fps")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="./data/6p-c0.avi")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=960)
    parser.add_argument("--display_height", type=int, default=540)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--save_name", type=str, default="6p-c0")
    parser.add_argument("--detect_file", type=str, default="./data/6p-c0.txt")
    # parser.add_argument("--h", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        # with torch.no_grad():
        vdo_trk.run()