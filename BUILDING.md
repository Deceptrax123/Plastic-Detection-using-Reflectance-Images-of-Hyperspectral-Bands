# Running the Inference Scripts
Use Bash or Zsh to run the scripts
- Clone the Repository using ```git@github.com:Deceptrax123/plastic-radiance-conversion.git ```
 - Create a virtual environment using ```python -m venv <name>```
 - Activate the virtual environment ```source <name>/bin/activate```
 - ```pip install -r requirements.txt```
 - Run ```export PYTHONPATH="path/to/repository"```
 - To run the inference scripts use ```python model_tests/inference.py``` and follow the instructions mentioned.
 
 # Training from Scratch
 - Follow the same steps as mentioned above to set up the environment
 - In ```Training_loops/train.py```, Alter the path for saving the weights.
 - Run ```python Training_loops/train.py```