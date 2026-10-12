# Attributions

Every borrowed model, piece of code, and test asset, with its license and what was used.
RoadWitness itself is MIT. No GPL-family code or weights are in this repository or its runtime.

## Models (downloaded by `python -m models.download`, never committed)
| Model | Upstream | License | Used for |
|---|---|---|---|
| ALIKED n16rot (ONNX) | Method and weights: [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED). ONNX file: [YangGuanyuhan/lightglue_opencv_project](https://github.com/YangGuanyuhan/lightglue_opencv_project), the file OpenCV's own test suite downloads (`opencv_extra/testdata/dnn/download_models.py`). | BSD-3-Clause (ALIKED) | Keypoints and descriptors |
| LightGlue for ALIKED (ONNX) | Method and weights: [cvg/LightGlue](https://github.com/cvg/LightGlue). ONNX file: same repository as above. | Apache-2.0 (LightGlue) | Feature matching |
| YuNet face detector | [OpenCV Zoo](https://huggingface.co/opencv/face_detection_yunet) | MIT | Privacy blur |
| LPD-YuNet plate detector | [OpenCV Zoo](https://huggingface.co/opencv/license_plate_detection_yunet) | Apache-2.0 | Privacy blur |
| PP-OCRv3 text detection | [OpenCV Zoo](https://huggingface.co/opencv/text_detection_ppocr), from PaddleOCR | Apache-2.0 | Privacy blur (plate fallback) |

Note on the ALIKED / LightGlue ONNX files: the repository hosting the exports carries no license file of its own. The underlying methods and weights are BSD-3-Clause and Apache-2.0, and the same files are what OpenCV's test data points to. This is stated here so a reviewer can judge it; if it is not acceptable, the exports can be regenerated from the upstream repositories.

## Code
| What | Upstream | License | Where |
|---|---|---|---|
| Plate detector prior generation and box decoding | OpenCV Zoo `lpd_yunet.py` | Apache-2.0 | `vision/privacy.py` (`PlateDetector`), re-implemented from the reference |
| Segmentation architecture used for export | torchvision `lraspp_mobilenet_v3_large` | BSD-3-Clause | offline training/export only; not in the runtime |

## Considered and rejected for license reasons
| What | License | Why rejected |
|---|---|---|
| Ultralytics YOLOv8 / YOLO11 (code and weights) | AGPL-3.0 | Not MIT-compatible. Replaced by LRASPP-MobileNetV3 (see `DECISIONS.md` D006). |
| Indian Roads semantic segmentation dataset (e-Yantra) | GPL 2.0 | GPL family. |

## Test assets
| File | Source | License |
|---|---|---|
| `tests/fixtures/box.png`, `box_in_scene.png` | `opencv_extra/testdata/cv/shared` | Apache-2.0 (OpenCV test data) |
| `tests/fixtures/astronaut.png` | NASA photo of Eileen Collins, distributed with scikit-image | Public domain (NASA) |

## Datasets
| Dataset | Source | License | Used for |
|---|---|---|---|
| Cracks and Potholes in Road Images (CPRID) | Passos et al., Mendeley Data, doi:10.17632/t576ydh9v8.4; images supplied by DNIT (Brazil) | CC BY 4.0 | Segmenter training and public benchmark |

Rejected datasets and the reasons are listed in `eval/datasets.md`.
