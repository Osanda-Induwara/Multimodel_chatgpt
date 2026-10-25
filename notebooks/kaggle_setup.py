# %% [markdown]
# Kaggle notebook setup
#
# Before running this cell, use Kaggle's "Add Input" button and select the private
# dataset named `tea_liquor_dataset`. Enable Internet only if cloning from GitHub.

# %%
# --- Previous Colab setup (kept as a reference; do not run in Kaggle) ---
# from google.colab import drive
# drive.mount("/content/drive")
# %cd /content/drive/MyDrive/TeaProject/Multimodel_chatgpt

# %%
# --- Kaggle setup ---
# Option A: enable Internet in Settings > Session options, then clone from GitHub.
# !git clone --depth 1 https://github.com/Osanda-Induwara/Multimodel_chatgpt.git /kaggle/working/Multimodel_chatgpt
# %cd /kaggle/working/Multimodel_chatgpt
# !pip install -q -r requirements.txt

# %%
# Option B: Internet-free setup. Upload the project code as a second private Kaggle
# dataset (suggested name: multimodel-chatgpt-code), attach it with Add Input, then run:
# !cp -r /kaggle/input/multimodel-chatgpt-code /kaggle/working/Multimodel_chatgpt
# %cd /kaggle/working/Multimodel_chatgpt
# !python -c "import torch, torchvision, cv2, pandas, sklearn, matplotlib, seaborn; print('Required packages available')"

# %%
# Verify the dataset mount and train grade classification.
# !python scripts/train_grade.py --model mobilenet --epochs 10 --batch-size 32 --lr 0.0003 --weight-decay 0.0001 --image-size 160 --patience 4
