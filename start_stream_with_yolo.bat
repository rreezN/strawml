@echo off
CD /D D:/HCAI/msc/strawml
call strawenv\Scripts\activate
python strawml/visualizations/stream.py --yolo_straw
