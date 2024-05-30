import cv2
import glob
import json
import numpy as np
import os
import threading

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ===
# Constants
# ===

SCREEN_SIZE = (1440, 2560) # screen size as per http://www.interactionmining.org/rico.html
DEVICE = 'mps'
NUM_WORKERS = 6
LR = 3e-4
BATCH_SIZE = 16
EPOCHS = 10
PRETRAINED_MODEL = "/Users/rohan/3_Resources/ai_models/vit-base-patch16-224"

IMAGE_EXT = 'jpg'
IMAGE_DIR = '/Users/rohan/3_Resources/ai_datasets/rico/combined'
SEMANTIC_ANNOTATIONS_DIR = '/Users/rohan/3_Resources/ai_datasets/rico/semantic_annotations'

# ===
# Utils
# ===

def get_labels(item: dict) -> list[str]:
  ret = []
  if "componentLabel" in item: ret.append(item["componentLabel"])
  if "children" in item:
    for child in item["children"]:
      ret.extend(get_labels(child))
  return ret



def is_within_screen(item: dict) -> bool:
  # check to see if in screen size
  if 'bounds' not in item: return False
  bounds = item['bounds']
  if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > SCREEN_SIZE[0] or bounds[3] > SCREEN_SIZE[1]: return False

  return True

def has_area(item: dict, threshold: int = 4) -> bool:
  # check to see if has area
  if 'bounds' not in item: return False
  bounds = item['bounds']
  if bounds[2] - bounds[0] < threshold or bounds[3] - bounds[1] < threshold: return False
  return True

def extract_data(item: dict, image_id: str) -> list[dict]:
  ret = []
  if "componentLabel" in item and is_within_screen(item) and has_area(item):
    ret.append({
      "image_id": image_id,
      "label": item["componentLabel"],
      "bounds": item["bounds"],
      "screen_size": SCREEN_SIZE
    })
  if "children" in item:
    for child in item["children"]:
      ret.extend(extract_data(child, image_id))
  return ret

# ===
# Dataset
# ===

class RicDataset(Dataset):
  def __init__(self, flattened_ds, image_dir: str, image_ext: str, label2idx: dict[str, int], processor):
    self.flattened_ds = flattened_ds
    self.image_dir = image_dir
    self.image_ext = image_ext
    self.label2idx = label2idx
    self.processor = processor

  def __len__(self):
    return len(self.flattened_ds)

  def __getitem__(self, idx):
    item = self.flattened_ds[idx]

    # load image
    img = cv2.imread(os.path.join(self.image_dir, f"{item['image_id']}.{self.image_ext}"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # scaled the bounds
    screen_size = item["screen_size"]
    imageData_bounds = [
      img.shape[1],
      img.shape[0]
    ]
    interest_region_bounds = item["bounds"]
    scaled_bounds = np.array(interest_region_bounds).reshape(2, 2).T / np.array(screen_size).reshape(2, 1) * np.array(imageData_bounds).reshape(2, 1)
    scaled_bounds = scaled_bounds.astype(int).T.reshape(-1)
    left, top, right, bottom = scaled_bounds
    if left > right: left, right = right, left
    if top > bottom: top, bottom = bottom, top
    # crop the tensor
    cropped_img = img[top:bottom, left:right]
    
    assert cropped_img.shape == (bottom - top, right - left, 3), (
      f'Shape mismatch: {cropped_img.shape} != {(bottom - top, right - left, 3)}\n'
      f'Image Id: {item["image_id"]}\n'
      f'Bounds: {item["bounds"]}\n'
      f'Screen size: {screen_size}\n'
      f'Label: {item["label"]}'
    )
    h, w, _ = cropped_img.shape
    assert h > 0 and w > 0, (
      f'h should be > 0, but is {h}\n'
      f'w should be > 0, but is {w}\n'
      f'Image Id: {item["image_id"]}\n'
      f'Bounds: {item["bounds"]}\n'
      f'Screen size: {screen_size}\n'
      f'Label: {item["label"]}'
    )

    
    # apply processor
    pixel_values = self.processor(Image.fromarray(cropped_img), return_tensors='pt', image_mean=None, input_data_format="channels_last").pixel_values
    # return
    return {
      'pixel_values': pixel_values.squeeze(0),
      'label': torch.tensor(self.label2idx[item['label']]).to(torch.long)
    }


# ===
# Model
# ===

class RicModel(torch.nn.Module):
  def __init__(self, backbone, n_classes: int):
    super().__init__()
    self.backbone = backbone
    self.classifier = torch.nn.Linear(768, n_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x).last_hidden_state[:, 0].relu()
    x = self.classifier(x)
    return x



# ===
# Main
# ===

def process_file(fn: str, all_labels: set[str], flattened_ds: list[dict]) -> None:
  with open(fn) as f: data = json.load(f)
  _tmp = os.path.basename(fn).split('.')
  assert len(_tmp) == 2
  image_id = _tmp[0]
  extracted_data = extract_data(data, image_id)
  flattened_ds.extend(extract_data(data, image_id))
  all_labels.update(get_labels(data))


if __name__ == "__main__":
  # get all labels and flatten the annotations dict
  all_labels = set()
  flattened_ds = []

  all_threads = []
  for i, fn in tqdm(enumerate(glob.glob(SEMANTIC_ANNOTATIONS_DIR + "/*.json")), desc="Loading all labels"):
    thread = threading.Thread(target=process_file, args=(fn, all_labels, flattened_ds))
    thread.start()
    all_threads.append(thread)
    if i % 100 == 0:
      for thread in all_threads: thread.join()
      all_threads = []
  for thread in all_threads: thread.join()
  for thread in all_threads: assert not thread.is_alive()

  print(f'Labels: {all_labels}\nLen: {len(all_labels)}')
  print(f'Total number of annotations: {len(flattened_ds)}')
  label2idx = {label: idx for idx, label in enumerate(all_labels)}
  idx2label = {idx: label for label, idx in label2idx.items()}
  with open('label2idx.json', 'w') as f: json.dump(label2idx, f)


  # train test split
  train_ds, test_ds = train_test_split(flattened_ds, test_size=0.2, random_state=42)

  processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
  train_ds = RicDataset(train_ds, IMAGE_DIR, IMAGE_EXT, label2idx, processor)
  test_ds = RicDataset(test_ds, IMAGE_DIR, IMAGE_EXT, label2idx, processor)
  train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # define model
  backbone = AutoModel.from_pretrained(PRETRAINED_MODEL)
  model = RicModel(backbone, len(label2idx))

  # freezing backbone
  for param in backbone.parameters(): param.requires_grad = False

  # optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

  # loss_fn
  criterion = F.cross_entropy

  # training loop
  model.to(DEVICE)

  try:
    for epoch in range(EPOCHS):
      model.train()
      optimizer.zero_grad()
      train_loss = []
      pbar = tqdm(train_dl, total=len(train_dl), desc=f'Epoch {epoch}: Training')
      for batch in pbar:
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        pred = model(pixel_values)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
        pbar.set_postfix({'loss': train_loss[-1]})
      pbar.close()

      model.eval()
      test_loss = []
      with torch.no_grad():
        pbar = tqdm(test_dl, total=len(test_dl), desc=f'Epoch {epoch}: Testing')
        for batch in pbar:
          pixel_values = batch['pixel_values'].to(DEVICE)
          labels = batch['label'].to(DEVICE)
          pred = model(pixel_values)
          loss = criterion(pred, labels)
          test_loss.append(loss.item())
          pbar.set_postfix({'loss': test_loss[-1]})
        pbar.close()

      print(f'Epoch {epoch}: train loss {np.mean(train_loss):.4f}, test loss {np.mean(test_loss):.4f}')

  except KeyboardInterrupt:
    print('Interrupted by user')
  finally:
    # save model
    model.to('cpu')
    torch.save(model.state_dict(), 'model.bin')
