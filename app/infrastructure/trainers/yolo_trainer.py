from ultralytics import YOLO


class YoloTrainer:
    def __init__(self, model_path: str, data_config: str, epochs: int = 50, img_size: int = 640, batch_size: int = 16):
        self.model_path = model_path
        self.data_config = data_config
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_path)

    def train(self):
        if self.model is None:
            self.load_model()

        results = self.model.train(
            data=self.data_config,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            name="yolo_custom",
            exist_ok=True
        )
        return results

    def export(self, format: str = "onnx", export_path: str = None):
        if self.model is None:
            raise RuntimeError("Error. Model not loaded or trained.")

        export_args = {"format": format}
        if export_path:
            export_args["file"] = export_path

        self.model.export(**export_args)