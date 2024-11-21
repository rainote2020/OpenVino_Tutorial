import cv2 as cv
import numpy as np
import openvino as ov


class Yolo11Clssify:
    def __init__(self, model_path: str) -> None:
        self.ov_core = ov.Core()
        ov_model = self.ov_core.read_model(model_path)
        self.compiled_model = ov.compile_model(ov_model)

    def inference_once(self, input_img: cv.Mat):
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

    def post_process(self, outputs: ov.Tensor) -> int:

        return np.argmax(outputs[0])


if __name__ == "__main__":
    pass
