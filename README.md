# Digit Reader: teaching a computer to read handwritten digits

**ICT Assignment #2 — Computer Vision, Option A (improve a model on an open-source dataset)**
Bolat Rauan · Stream 2, Practice 2

🔗 **Live site + demo:** https://rauanking.github.io/digit-reader/

## Research question
How much does each model change reduce mistakes in handwritten digit recognition (MNIST)?

## Experiments
One change at a time: same data split, seed 42, 5 epochs, batch size 128. Accuracy is measured on 10 000 unseen test images.

| # | Model | Change | Test accuracy | Mistakes |
|---|---|---|---|---|
| 1 | Logistic regression | baseline | 92.36% | 764 |
| 2 | MLP (1 hidden layer) | + hidden layer | 96.97% | 303 |
| 3 | CNN, 1 conv, SGD | + convolution | 93.92% | 608 |
| 4 | CNN, 1 conv, Adam | SGD → Adam | 98.23% | 177 |
| 5 | CNN, 2 conv, Adam | + 2nd conv layer | 98.98% | 102 |
| 6 | CNN, 2 conv, Adam, Dropout | + dropout 0.3 | **99.03%** | **97** |

**Key findings**
- Mistakes fell from 764 to 97, 87% fewer than the baseline.
- A CNN trained with plain SGD was *worse* than the MLP: after 5 epochs it was still underfitting. The same CNN trained with Adam cut mistakes from 608 to 177.
- The second conv layer improved accuracy while using 3× fewer parameters.

## Files
| File | What it is |
|---|---|
| `index.html` | Landing page with results and an in-browser drawing demo (the CNN runs in plain JavaScript) |
| `ICT_A2_MNIST.ipynb` | Full notebook; open it in Google Colab and choose *Runtime → Run all* |
| `experiments.py` | The same experiments as a script |
| `make_site_data.py` | Builds `results_data.js` for the site |
| `model_weights.js`, `mnist_cnn.keras` | Final trained model |

Dataset: MNIST (LeCun, Cortes & Burges), loaded with `keras.datasets.mnist`.
AI use: an AI assistant (Claude) helped write and structure the code.
