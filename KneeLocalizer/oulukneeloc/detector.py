import os
import time
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from tqdm import tqdm

from oulukneeloc import SVM_MODEL_PATH
from oulukneeloc.proposals import (read_dicom, get_joint_y_proposals,
                                   preprocess_xray)


class KneeLocalizer:
    def __init__(self, svm_model_path=SVM_MODEL_PATH, size_mm=120):
        super().__init__()
        self.win_size = (64, 64)
        self.win_stride = (64, 64)
        self.block_size = (16, 16)
        self.block_stride = (8, 8)
        self.cell_size = (8, 8)
        self.padding = (0, 0)
        self.nbins = 9
        self.scales = [3.2, 3.3, 3.4, 3.6, 3.8]
        self.step = 95

        self.size_mm = size_mm
        self.svm_w, self.svm_b = np.load(svm_model_path, encoding='bytes')

    def predict(self, fileobj, spacing=None):
        """Localize the left and the right knee joints in PA X-ray image.

        Parameters
        ----------
        fileobj: str or ndarray
            Filename of the DICOM image, or already extracted uint16 ndarray.
        spacing: float or None
            Spacing extracted from the previously read DICOM.

        Returns
        -------
        detections: list of lists
            The first list has the bbox for the left knee joint.
            The second list has the bbox for the right knee joint.
        """

        if isinstance(fileobj, str):
            tmp = read_dicom(fileobj)
            if tmp is None:
                return None
            if len(tmp) != 2:
                return None
            img, spacing = tmp
            img = preprocess_xray(img)
        elif isinstance(fileobj, np.ndarray):
            img = fileobj
            if spacing is None:
                raise ValueError
        else:
            raise ValueError

        R, C = img.shape
        split_point = C // 2
        spacing = float(spacing)
        assert spacing > 0

        right_leg = img[:, :split_point]
        left_leg = img[:, split_point:]

        sizepx = int(self.size_mm / spacing)  # Proposal size

        # We will store the coordinates of the top left and
        # the bottom right corners of the bounding box
        hog = cv2.HOGDescriptor(self.win_size,
                                self.block_size,
                                self.block_stride,
                                self.cell_size,
                                self.nbins)

        # Make proposals for the right leg
        R, C = right_leg.shape
        displacements = range(-C // 4, 1 * C // 4 + 1, self.step)
        prop = get_joint_y_proposals(right_leg)
        best_score = -np.inf

        for y_coord in prop:
            for x_displ in displacements:
                for scale in self.scales:
                    if C / 2 + x_displ - R / scale / 2 >= 0:
                        # Candidate ROI
                        roi = np.array([C / 2 + x_displ - R / scale / 2,
                                        y_coord - R / scale / 2,
                                        R / scale, R / scale], dtype=np.int)
                        x1, y1 = roi[0], roi[1]
                        x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                        patch = cv2.resize(img[y1:y2, x1:x2], (64, 64))

                        hog_descr = hog.compute(patch, self.win_stride, self.padding)
                        score = np.inner(self.svm_w, hog_descr.ravel()) + self.svm_b

                        if score > best_score:
                            jc = np.array([C / 2 + x_displ, y_coord])
                            best_score = score

        roi_R = np.array([jc[0] - sizepx // 2,
                          jc[1] - sizepx // 2,
                          jc[0] + sizepx // 2,
                          jc[1] + sizepx // 2]).round().astype(np.int)

        # Make proposals for the left leg
        R, C = left_leg.shape
        displacements = range(-C // 4, 1 * C // 4 + 1, self.step)
        prop = get_joint_y_proposals(left_leg)
        best_score = -np.inf

        for y_coord in prop:
            for x_displ in displacements:
                for scale in self.scales:
                    if split_point + x_displ + R / scale / 2 < img.shape[1]:
                        roi = np.array([split_point + C / 2 + x_displ - R / scale / 2,
                                        y_coord - R / scale / 2,
                                        R / scale, R / scale], dtype=np.int)
                        x1, y1 = roi[0], roi[1]
                        x2, y2 = roi[0] + roi[2], roi[1] + roi[3]
                        patch = np.fliplr(cv2.resize(img[y1:y2, x1:x2], (64, 64)))

                        hog_descr = hog.compute(patch, self.win_stride, self.padding)
                        score = np.inner(self.svm_w, hog_descr.ravel()) + self.svm_b

                        if score > best_score:
                            jc = np.array([split_point + C / 2 + x_displ, y_coord])
                            best_score = score

        roi_L = np.array([jc[0] - sizepx // 2,
                          jc[1] - sizepx // 2,
                          jc[0] + sizepx // 2,
                          jc[1] + sizepx // 2]).round().astype(np.int)

        return [roi_L.tolist(), roi_R.tolist()], img


def worker(fname, path_input, localizer):
    tmp = read_dicom(os.path.join(path_input, fname))
    if tmp is None:
        ret = [fname, ] + [-1, ] * 4 + [-1, ] * 4
        return ' '.join([str(e) for e in ret])

    img, spacing = tmp
    img = preprocess_xray(img)
    try:
        detections = localizer.predict(img, spacing)
    except:
        print('Error finding the knee joints')
        detections = [[-1]*4, [-1]*4]

    if detections is None:
        detections = [[-1]*4, [-1]*4]
    return ' '.join(map(str, [fname, ] + detections[0] + detections[1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', "--dir")
    parser.add_argument('--fname_output', "--output",
                        default='../detection_results.txt')

    args = parser.parse_args()
    args.path_input = os.path.abspath(args.path_input)
    args.fname_output = os.path.abspath(args.fname_output)
    return args


if __name__ == "__main__":
    args = parse_args()

    ts_start = time.time()

    localizer = KneeLocalizer()

    def worker_partial(fname):
        return worker(fname, args.path_input, localizer)

    fnames = os.listdir(args.path_input)
    
    with Pool(cpu_count()) as pool:
        res = list(tqdm(pool.imap(
            worker_partial, iter(fnames)), total=len(fnames)))
        
    with open(args.fname_output, 'w') as f:
        for entry in res:
            f.write(entry + '\n')

    ts_end = time.time() - ts_start
    print('Script execution took {} seconds'.format(ts_end))
