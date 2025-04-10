@echo off
call .venv\Scripts\activate
python strawml/visualizations/stream.py --with_predictor
