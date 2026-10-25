# Tea liquor classification

This project trains two independent image classifiers from tea-liquor photographs:

- **Grade:** BOP, BOPF, OP, DUST
- **Region:** DI, KA, NU, RU, SB, UP, UV

The dataset is read automatically from `/content/drive/MyDrive/TeaProject/dataset/Raw Dataset` in Colab, falling back to the local development path. Images from the same physical sample, including its two camera captures, always remain in the same train/validation split.

## Workflow

```powershell
python scripts\inspect_preprocessing.py --limit 24
python scripts\train_grade.py --model efficientnet --epochs 20
python scripts\train_region.py --model efficientnet --epochs 20
python scripts\tune.py --task grade --model efficientnet --trials 8
python scripts\compare_experiments.py --task grade
```

## Kaggle

Attach the private Kaggle dataset **tea_liquor_dataset** to the notebook using **Add Input**. The code automatically resolves its `Raw Dataset` folder, including Kaggle's nested mount form `/kaggle/input/datasets/<owner>/tea-liquor-dataset/`; no path argument is needed. The first run creates `data/manifest.csv`.

```python
# Previous Colab setup — keep commented when working in Kaggle.
# from google.colab import drive
# drive.mount('/content/drive')

!git clone --depth 1 https://github.com/Osanda-Induwara/Multimodel_chatgpt.git /kaggle/working/Multimodel_chatgpt
%cd /kaggle/working/Multimodel_chatgpt
!pip install -q -r requirements.txt
!python scripts/train_grade.py --model mobilenet --epochs 10 --batch-size 32 --lr 0.0003 --weight-decay 0.0001 --image-size 160 --patience 4
```

If cloning fails with `Could not resolve host: github.com`, enable **Settings → Session options → Internet** and restart the session. Kaggle documents Internet in the notebook Settings pane. If Internet cannot be enabled, upload this project as a second private Kaggle dataset, attach it with **Add Input**, and copy it before running:

```python
!cp -r /kaggle/input/multimodel-chatgpt-code /kaggle/working/Multimodel_chatgpt
%cd /kaggle/working/Multimodel_chatgpt
```

Model outputs are saved to `/kaggle/working/Multimodel_chatgpt/results/` and can be downloaded from the notebook's Output panel after saving a version.

Each run creates a folder in `results/` containing its configuration, learning curves, confusion matrix, classification report, and predictions. Use the same commands with `--model mobilenet`, `resnet`, `shufflenet`, or `squeezenet`.

The first epoch caches each 224px tea crop in fast local runtime storage. Later epochs avoid repeatedly decoding the large phone photographs. The best model and training history are saved after every epoch, including when a Colab run ends early.

## Colour and reflection handling

Models receive RGB images because ImageNet transfer models were trained on RGB. HSV is used only during preprocessing: it helps detect the bright white bowl/background and identifies low-saturation, high-value specular reflection pixels inside the tea circle. Those small highlight pixels are inpainted before the circular crop is resized. Training also uses rotation, horizontal flip, mild colour jitter, and random erasing, reducing reliance on a camera reflection.

`rgb -> hsv` means representing each pixel as **Hue** (colour family), **Saturation** (colour strength), and **Value** (brightness), instead of red/green/blue amounts. It is useful for the described masking operation, but converting all model inputs to HSV would discard the RGB assumptions of pretrained models and is not the default here.

## Notes

- The first training run downloads pretrained ImageNet weights if they are not already cached.
- Keep the raw dataset unchanged. Manifest and results are written inside this project.
- A validation split is a practical baseline; for a final paper, use grouped cross-validation or a separately collected camera/session test set.
