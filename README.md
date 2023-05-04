# GEIA
Code for Findings-ACL 2023 paper: Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence

### Package Dependencies
* numpy
* pytorch 1.10.2
* sentence_transformers 2.2.0
* transformers 4.xx.x
* simcse 0.4
* datasets

### Data Preparation
We upload PC data under the ```data/``` folder.
The ABCD dataset we experimented can be found in https://drive.google.com/file/d/1oIo8P0Y8X9DTeEfOA1WUKq8Uix9a_Pte/view?usp=sharing.
For other datasets, we use ```datasets``` package to download and store them, so you can run our code directly.

### Baseline Attackers




### GIEA
**You need to set up arguments properly before running codes**:
```python attacker.py```

* --model_dir: Attacker model path from Huggingface or local model checkpoints.
* --num_epochs: Training epoches.
* --batch_size: Batch_size #.
* --dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
* --data_type: Train or test.
* --embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in ```attacker.py``` for more information.
* --decode: Decoding algorithm. We currently implement beam and sampling based decoding.
