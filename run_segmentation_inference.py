import os
from pathlib import Path

import numpy as np

import ml3d as _ml3d
import ml3d.torch as ml3d


def main() -> None:
    cfg_file = "ml3d/configs/kpconv_semantickitti.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    model = ml3d.models.KPFCNN(**cfg.model)
    cfg.dataset["dataset_path"] = "/home/ubuntu/data/larki/outdoor"
    dataset = _ml3d.datasets.Larki(cfg.dataset.pop("dataset_path", None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)

    # download the weights.
    ckpt_folder = Path("./logs/")
    ckpt_folder.mkdir(exist_ok=True)
    ckpt_path = ckpt_folder / "kpconv_semantickitti_202009090354utc.pth"
    weights_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
    if not ckpt_path.exists():
        cmd = f"wget {weights_url} -O {ckpt_path}"
        os.system(cmd)

    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    test_split = dataset.get_split("test")
    data = test_split.get_data(0)

    # run inference on a single example.
    # returns dict with 'predict_labels' and 'predict_scores'.
    result = pipeline.run_inference(data)

    print(result)

    np.save("out_labels.npy", result["predict_labels"])
    np.save("out_scores.npy", result["predict_scores"])


if __name__ == "__main__":
    main()
