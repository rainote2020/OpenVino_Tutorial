import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List
import openvino as ov


@dataclass
class Result:
    label: int
    conf: float
    center_point: tuple


class Yolo11Detect:
    def __init__(self, model_path: str) -> None:
        self.ov_core = ov.Core()
        ov_model = self.ov_core.read_model(model_path)
        self.compiled_model = ov.compile_model(ov_model)

    def inference_once(self, input_img: cv.Mat) -> List[Result]:
        input_data = [self.pre_process(input_img)]
        output_data = self.compiled_model(input_data)[self.compiled_model.output(0)]
        return self.post_process(output_data)

    def pre_process(self, input_frame: cv.Mat) -> cv.Mat:
        [height, width, _] = input_frame.shape
        length = max((height, width))
        # letterbox
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = input_frame
        # hwc -> nchw

        return cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    def post_process(
        self, outputs: ov.Tensor, conf_threshold: float = 0.5, nms_iou_threshold: float = 0.7
    ) -> List[Result]:
        # (1,4+n,8400) to (8400,4+n)
        outputs = outputs.transpose(0, 2, 1)[0]
        # (8400,4)
        boxes = outputs[:, :4]
        # (8400,n)
        scores = np.max(outputs[:, 4:], axis=1).tolist()

        result_boxes = cv.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_iou_threshold, 0.5)

        results = []
        for index in result_boxes:
            (_, maxScore, _, (_, maxClassIndex)) = cv.minMaxLoc(outputs[index][4:])
            results.append(
                Result(
                    label=maxClassIndex, conf=maxScore, center_point=(outputs[index][0], outputs[index][1])
                )
            )

        return results


if __name__ == "__main__":
    pass
