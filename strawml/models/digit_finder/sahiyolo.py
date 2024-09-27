"""
YOLO + SAHI (Slicing Aided Hyper Inference)

Inspiration from:
    https://github.com/obss/sahi
WITH CODE FROM:
    https://github.com/niconielsen32/tiling-window-detection/blob/main/tiling.py
But written into a class for easy use in the strawml pipeline
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Optional, Union
from PIL import Image

class SahiYolo:
    def __init__(self, model_path, device="cuda", verbose=False, yolo_threshold=0.1):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.verbose = verbose
        self.yolo_threshold = yolo_threshold
        self.classes = self.model.names


    def get_slice_bboxes(self,
        image_height: int,
        image_width: int,
        slice_height: Optional[int] = None,
        slice_width: Optional[int] = None,
        auto_slice_resolution: bool = True,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        ) -> List[List[int]]:

        slice_bboxes = []
        y_max = y_min = 0

        if slice_height and slice_width:
            y_overlap = int(overlap_height_ratio * slice_height)
            x_overlap = int(overlap_width_ratio * slice_width)
        else:
            raise ValueError("Compute type is not auto and slice width and height are not provided.")

        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes
    
    def slice_image(self,
        image: Union[str, Image.Image],
        slice_height: Optional[int] = None,
        slice_width: Optional[int] = None,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        auto_slice_resolution: bool = True,
        min_area_ratio: float = 0.1,
        out_ext: Optional[str] = None,
        verbose: bool = False,
        ) -> "SliceImageResult":

        image_pil = image

        image_width, image_height = image_pil.size
        if not (image_width != 0 and image_height != 0):
            raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
        slice_bboxes = self.get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            auto_slice_resolution=auto_slice_resolution,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        n_ims = 0

        sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width])

        image_pil_arr = np.asarray(image_pil)

        for slice_bbox in slice_bboxes:
            n_ims += 1

            tlx = slice_bbox[0]
            tly = slice_bbox[1]
            brx = slice_bbox[2]
            bry = slice_bbox[3]
            image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

            slice_width = slice_bbox[2] - slice_bbox[0]
            slice_height = slice_bbox[3] - slice_bbox[1]
    
            sliced_image = SlicedImage(
                image=image_pil_slice, starting_pixel=[slice_bbox[0], slice_bbox[1]]
            )
            sliced_image_result.add_sliced_image(sliced_image)

        return sliced_image_result
    

    def score_frame(self, frame):
        """
        Score a frame using the YOLO model
        :param frame: frame to score
        :return: results of the scoring
        """
        image = Image.fromarray(frame)
        height, width = image.size

        slice_height = int(height/10)
        slice_width = int(width/10)

        slice_image_result = self.slice_image(
            image=image,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
            auto_slice_resolution=False,
        )

        print(f"Number of slices: {len(slice_image_result.sliced_image_list)}")

        bboxes = []
        confs = []
        class_ids = []

        for image_slice in slice_image_result.sliced_image_list:
            window = image_slice.image
            start_x, start_y = image_slice.starting_pixel

            results = self.model(window, verbose=self.verbose)

            conf = results[0].boxes.conf.cpu().numpy()
            labels = results[0].boxes.cls.cpu().numpy()
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]

                x1 += start_x
                y1 += start_y
                x2 += start_x
                y2 += start_y

                x1, y1, x3, y3 = int(x1), int(y1), int(x2), int(y2)
                xyxy[i, 0] = x1
                xyxy[i, 1] = y1
                xyxy[i, 2] = x3
                xyxy[i, 3] = y3                

            mask = conf >= self.yolo_threshold
            # If no boxes are found, continue
            if not mask.any():
                continue
            bboxes.append(xyxy[mask])
            confs.append(conf[mask])
            class_ids.append(labels[mask])
        # combine list of arrays to single array
        # first check if there are any detections
        if len(class_ids) == 0:
            return np.array([]), np.array([]), np.array([])
        class_ids = np.concatenate(class_ids, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        confs = np.concatenate(confs, axis=0)
        return class_ids, bboxes, confs


class SlicedImage:
    def __init__(self, image, starting_pixel):
        self.image = image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    def __init__(self, original_image_size: List[int], image_dir: Optional[str] = None):
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

        self._sliced_image_list: List[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def starting_pixels(self) -> List[int]:
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> List[int]:
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i):
        def _prepare_ith_dict(i):
            return {
                "image": self.images[i],
                "starting_pixel": self.starting_pixels[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)
