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
# !git clone --depth 1 https://github.com/Osanda-Induwara/Multimodel_chatgpt.git /kaggle/working/Multimodel_chatgpt
# %cd /kaggle/working/Multimodel_chatgpt
# !pip install -q -r requirements.txt

# %%
# Verify the dataset mount and train grade classification.
# !python scripts/train_grade.py --model mobilenet --epochs 10 --batch-size 32 --lr 0.0003 --weight-decay 0.0001 --image-size 160 --patience 4
