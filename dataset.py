from datasets import load_dataset


# ds = load_dataset("nkasmanoff/retail_detector_flattened")
ds = load_dataset("moondream/waste_detection")
centered_coords = False


class ObjectDetectionDataset:
    def __init__(self, dataset, centered_coords=centered_coords):
        self.dataset = dataset
        self.centered_coords = centered_coords

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # from the dataset  get image, convert to xmin, ymin, xmax, ymax
        sample = self.dataset[idx]
        image = sample["image"]
        label = sample["labels"][0]
        boxes = []
        for box in sample["boxes"]:
            x, y, w, h = box
            if self.centered_coords:
                x = x + w / 2
                y = y + h / 2
            boxes.append(
                {
                    "x_min": x - w / 2,
                    "y_min": y - h / 2,
                    "x_max": x + w / 2,
                    "y_max": y + h / 2,
                }
            )

        # Extract the image, query, and bounding boxes
        return image, label, boxes


def load_object_detection_dataset(split):
    return ObjectDetectionDataset(ds[split])
