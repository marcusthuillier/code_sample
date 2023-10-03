#!/bin/bash
python3 -m pip install virtualenv
python3 -m virtualenv virtual
source virtual/bin/activate
pip3 install -r requirements.txt
cd Code
python3 main.py