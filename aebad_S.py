import os
from glob import glob
from utils.load_dataset import DatasetSplit
from datasets.mvtec import MVTecDataset  # keep this for compatibility


class AeBAD_SDataset(MVTecDataset):
    """
    Rewritten custom dataset class for AeBAD_S.
    Fully supports domain shift setups: view, illumination, background.
    """

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            class_path = os.path.join(self.source, classname, self.split.value)
            mask_path = os.path.join(self.source, classname, "ground_truth")
            if not os.path.isdir(class_path):
                continue

            anomaly_types = [a for a in os.listdir(class_path)
                             if os.path.isdir(os.path.join(class_path, a))]

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                sub_dirs = (
                    os.listdir(os.path.join(class_path, anomaly))
                    if self.split.value == "train" and anomaly == "good"
                    else [self.cfg.DATASET.domain_shift_category]
                )

                imgpaths_per_class[classname][anomaly] = []

                for sub in sub_dirs:
                    sub_path = os.path.join(class_path, anomaly, sub)
                    files = glob(os.path.join(sub_path, "*.png"))
                    imgpaths_per_class[classname][anomaly].extend(files)

                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        mask_sub_path = os.path.join(mask_path, anomaly, sub)
                        maskpaths_per_class[classname].setdefault(anomaly, [])
                        maskpaths_per_class[classname][anomaly].extend(
                            [os.path.join(mask_sub_path, os.path.basename(f)) for f in files]
                        )
                    else:
                        maskpaths_per_class[classname]["good"] = None

        # Flatten output
        data = []
        for classname in sorted(imgpaths_per_class):
            for anomaly in sorted(imgpaths_per_class[classname]):
                for idx, img_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    mask = (
                        maskpaths_per_class[classname][anomaly][idx]
                        if self.split == DatasetSplit.TEST and anomaly != "good"
                        else None
                    )
                    data.append([classname, anomaly, img_path, mask])

        return imgpaths_per_class, data
