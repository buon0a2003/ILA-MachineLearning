python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#example
python main.py train dataset.csv model.pkl

python main.py predict model.pkl test_data.csv
