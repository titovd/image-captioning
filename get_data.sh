#!/bin/bash

# Загрузим данные с http://cocodataset.org/#download
curl http://images.cocodataset.org/zips/train2017.zip > train2017.zip
curl http://images.cocodataset.org/zips/val2017.zip > val2017.zip
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip > annotations_trainval2017.zip

unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
unzip train2017.zip > log
rm train2017.zip
unzip val2017.zip > log
rm val2017.zip