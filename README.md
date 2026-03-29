# Create the virtual environment
python -m venv venv

# Then activate it
.\venv\Scripts\activate

# Then install libery
pip install -r requirements.txt

# Run
python -m train.train_loop

uvicorn inference.ai_service:app --host 0.0.0.0 --port 8000

python -m train.train_loop --episodes 5000  --save-every 500 --batch-size 128 --device cuda
python -m train.train_loop --episodes 10  --save-every 500 --batch-size 128 --device cuda


python train_loop.py \
  --episodes 30000 \
  --save-every 3000 \
  --batch-size 256 \
  --lr 3e-4 \
  --device cuda
