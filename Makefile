# define the name of the virtual environment directory
VENV := /Users/jan/.virtualenv/8_3d

# default target, when make executed without arguments
all: venv

#$(VENV)/bin/activate: requirements.txt
#	make -C ./libcluon
#	python3 -m venv $(VENV)
#	./$(VENV)/bin/pip install --upgrade pip
#	./$(VENV)/bin/pip install --editable .
#	./$(VENV)/bin/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/bin/activate
export PYTHONPATH=$PYTHONPATH:./

run_test:
	$(VENV)/bin/python3 ./tools/clear_results.py
	$(VENV)/bin/python3 ./main.py
	$(VENV)/bin/python3 ./tools/create_label_from_masks.py
	$(VENV)/bin/python3 ./tools/IoU.py

run_hardware:
	./$(VENV)/bin/python3 ./main/computer_vision.py
	./$(VENV)/bin/python3 ./main/training.py -l True

run_software:
	./$(VENV)/bin/python3 ./main/training.py -l False

clean:
	rm -rf ./$(VENV)
	rm -rf ./hubert.egg-info
	find -name "*.pyc" -delete