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
**You need to set up arguments properly before running codes**:
```python projection.py```

* --model_dir: Attacker model path from Huggingface (like 'gpt2-large' and 'microsoft/DialoGPT-xxxx') or local model checkpoints.
* --num_epochs: Training epoches.
* --batch_size: Batch_size #.
* --dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
* --data_type: Train or test.
* --embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in ```projection.py``` for more information.
* --model_type: NN or RNN

By running:
```python projection.py```
You will train your own baseline model and evaluate it. If you want to just train or eval a certain model, check the last four lines of ```projection.py``` and disable the corresponding codes.

### GIEA

#### GPT-2 Attacker
**You need to set up arguments properly before running codes**:
```python attacker.py```

* --model_dir: Attacker model path from Huggingface (like 'gpt2-large' and 'microsoft/DialoGPT-xxxx') or local model checkpoints.
* --num_epochs: Training epoches.
* --batch_size: Batch_size #.
* --dataset: Name of the dataset including personachat, qnli, mnli, sst2, wmt16, multi_woz and abcd.
* --data_type: Train or test.
* --embed_model: The victim model you wish to attack. We currently support sentence-bert models and huggingface models, you may refer to our model_cards dictionary in ```attacker.py``` for more information.
* --decode: Decoding algorithm. We currently implement beam and sampling based decoding.

You should train the attacker on training data at first, then test your attacker on the test data to obtain test logs. Then you can evaluate attack performance on test logs by changing model_dir to your trained attcker and data_type to test.

If you want to train a randomly initialized GPT-2 attacker, after setting the arguments, run:
```python attacker_random_gpt2.py```

#### Other Attackers
Due to the fact that different decoders have different implementaions, we use separate py files for each model (the decoding implementations also differ). 

If you want to try out opt as the attacker model, run:
```python attacker_opt.py```

If you want to try out t5 as the attacker model, run:
```python attacker_t5.py```

### Evaluation
**You need to make sure the test reuslt paths is set inside the 'eval_xxx.py' files.**

To obtain classification performance, run:
```python eval_classification.py```

To obtain generation performance, run:
```python eval_generation.py```

To calculate perplexity, you need to set the LM to caluate PPL, run:
```python eval_ppl.py```


### Miscellaneous

Please send any questions about the code and/or the algorithm to hlibt@connect.ust.hk
