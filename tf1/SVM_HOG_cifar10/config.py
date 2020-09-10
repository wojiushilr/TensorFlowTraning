# -*- coding: utf-8 -*-
import configparser as cp #3.6版本开始为configparser
import json

config = cp.RawConfigParser()
config.read('./data/config/config.cfg')

orientations = json.loads(config.get("hog", "orientations"))
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")
train_feat_path = config.get("path", "train_feat_path")
test_feat_path = config.get("path", "test_feat_path")
model_path = config.get("path", "model_path")

print(orientations)
print(pixels_per_cell)
print(cells_per_block)
print(visualize)
print(normalize)
print(train_feat_path)
print(test_feat_path)
print(model_path)
