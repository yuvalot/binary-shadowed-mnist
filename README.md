# Binary Shadowed MNIST
 
This repository is meant to provide a version of MNIST dataset, which is shadowed by some percentages with randomly cerated masks.

## Creation
To recreate the dataset, run 
```bash
python creation.py
```

You might need to install some dependencies (provided at requirements.txt).

## Reading
To use the dataset, read it with the following python code (or any equivalent in another language):
```python
images = np.fromfile('./data/shadow=0.9/images.bin', dtype=np.bool)
```
