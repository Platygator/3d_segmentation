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

generate:
	$(VENV)/bin/python3 ./tools/clear_results.py
	$(VENV)/bin/python3 ./main.py
	$(VENV)/bin/python3 ./tools/IoU.py

IoU:
	$(VENV)/bin/python3 ./tools/IoU.py

txt_to_npy:
	$(VENV)/bin/python3 ./tools/txt_to_npy.py

#clean:
#	rm -rf ./$(VENV)
#	rm -rf ./hubert.egg-info
#	find -name "*.pyc" -delete