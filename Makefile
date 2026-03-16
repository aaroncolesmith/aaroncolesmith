VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: run install clean venv

# Default target
all: run

# Run the streamlit app using the venv's python
run: $(VENV)
	$(PYTHON) -m streamlit run 1_🏠_Home.py

# Install dependencies if requirements.txt changes
install: $(VENV)

# Create the virtual environment and install packages
$(VENV): requirements.txt
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)

# Manual target to recreate venv
venv:
	rm -rf $(VENV)
	$(MAKE) $(VENV)

# Clean up temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete