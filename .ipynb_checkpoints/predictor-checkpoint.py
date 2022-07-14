import os
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer

class Predictor:
    _metadata = None
    _dataset_dict = None
    _predictor = None
    _outputs = None
    _outputs_cash = None
    _original_img = None
    _img = None
    _img_dir = "static/imgs/"
    _img_paths = []

    def __init__(self):
        # データセットを登録
        register_coco_instances(
            "coins", {}, "coins/coco-1612779490.2197058.json", "coins")
        self._metadata = MetadataCatalog.get("coins")
        self._dataset_dicts = DatasetCatalog.get("coins")
        setup_logger()

        # 設定を決める
        cfg = get_cfg()
        yamlPath = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # mac

        cfg.merge_from_file(model_zoo.get_config_file(yamlPath))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
        cfg.DATASETS.TRAIN = ("coins",)
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.DEVICE = "cpu"

        # 予測器を作成
        self._predictor = DefaultPredictor(cfg)

    @property
    def metadata(self):
        return self._metadata

    @property
    def img(self):
        return self._img

    @property
    def img_paths(self):
        return self._img_paths

    def predict(self, img):
        self._original_img = img
        self._outputs = self._predictor(img)
        data = self._outputs["instances"].to("cpu")
        v = Visualizer(img[:, :, ::-1],
                       metadata=self._metadata,
                       scale=1.0
                       )
        v = v.draw_instance_predictions(data)
        self._img = v.get_image()[:, :, ::-1]
        scores_np = data.get("scores").to('cpu').detach().numpy().copy()
        classes_np = data.get("pred_classes").to('cpu').detach().numpy().copy()
        cash_data = [1, 5, 10, 100]
        cash_sum = 0
        for i, n in enumerate(classes_np):
            if scores_np[i] > 0.8:
                cash_sum += cash_data[n]
        self._outputs_cash = str(cash_sum) + "円"
        return self._outputs
