import cv2 as cv
import numpy as np
import openvino as ov
import time
from pathlib import Path
import math

t1_sum = 0
t2_sum = 0
t3_sum = 0
num = 0


class PaddleOCR:
    def __init__(self, model_path: Path, dict_path: Path) -> None:
        self.ov_core = ov.Core()
        ov_model = self.ov_core.read_model(model_path)
        self.compiled_model = ov.compile_model(ov_model, device_name="CPU")
        self.rec_input_layer = self.compiled_model.input(0)
        self.rec_output_layer = self.compiled_model.output(0)

        with open(dict_path, "r") as dict_file:
            self.dict_list = list(map(lambda x: x.strip(), dict_file.readlines()))

        self.post_process = CTCLabelDecode(character_dict_path=dict_path, use_space_char=True)

    def inference_once(self, input_img: cv.Mat) -> list | tuple[list, list]:
        input_data = [self.pre_process(input_img)]
        output_data = self.compiled_model(input_data)[self.rec_output_layer]
        result = self.post_process(output_data)
        return result

    def inference_once_perfcount(self, input_img: cv.Mat) -> list | tuple[list, list]:
        global t1_sum, t2_sum, t3_sum, num
        start = time.perf_counter()
        input_data = [self.pre_process(input_img)]
        t1 = time.perf_counter()
        t1_sum += t1 - start
        output_data = self.compiled_model(input_data)[self.rec_output_layer]
        t2 = time.perf_counter()
        t2_sum += t2 - t1
        result = self.post_process(output_data)
        t3_sum += time.perf_counter() - t2
        num += 1
        # print(t1_sum / num, t2_sum / num, t3_sum / num)
        return result

    # [h,w,c] ->[1,c,new_h,new_w]
    def pre_process(self, input_frame: cv.Mat) -> cv.Mat:
        # 填充图像
        h, w, _ = input_frame.shape
        new_w = min(320, round(w * (48 / h)))

        resize_frame = cv.resize(input_frame, (new_w, 48))
        resize_frame = resize_frame.astype(np.float32)
        resize_frame /= 255.0
        resize_frame -= 0.5
        resize_frame /= 0.5

        padding_frame = np.zeros((48, 320, 3), np.float32)
        padding_frame[:, :new_w] = resize_frame
        # 归一化 & 标准化mean
        padding_frame = cv.dnn.blobFromImage(padding_frame, swapRB=False)

        return padding_frame


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, character_type="ch", use_space_char=False):

        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []

        if character_type is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:

            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, character_type="ch", use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
