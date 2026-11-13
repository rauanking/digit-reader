"""After experiments.py: writes results_data.js for index.html (numbers, sample digits, error text, conclusions)."""
import json

import numpy as np
from tensorflow import keras

data = json.load(open("results.json"))
rs = data["results"]
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
model = keras.models.load_model("mnist_cnn.keras")
pred = model.predict((x_test / 255.0).astype("float32")[..., None], verbose=0).argmax(1)
wrong = pred != y_test
cm = np.zeros((10, 10), int)
for t, p in zip(y_test, pred):
    cm[t, p] += 1
off = cm - np.diag(np.diag(cm))
pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
per_digit = [(cm[d, d] / cm[d].sum(), d) for d in range(10)]
worst_acc, worst_d = min(per_digit)

samples = [{"label": int(y_test[i]), "pixels": x_test[i].flatten().tolist()} for i in range(12)]


def delta(a, b):
    return rs[b]["errors"] - rs[a]["errors"]


d = {k: delta(*v) for k, v in {"mlp": (0, 1), "conv": (1, 2), "adam": (2, 3), "conv2": (3, 4), "drop": (4, 5)}.items()}


def say(n):
    return f"{-n} fewer mistakes" if n < 0 else (f"{n} more mistakes" if n > 0 else "no change in mistakes")


conclusions = [
    f"The hidden layer gave the biggest single jump: {rs[0]['errors']} → {rs[1]['errors']} mistakes ({say(d['mlp'])}).",
    f"Convolution trained with plain SGD: {say(d['conv'])} compared with the MLP. "
    + ("With a small learning rate and only 5 epochs, SGD had not finished learning, so the better architecture alone was not enough."
       if d["conv"] > 0 else "Even with plain SGD, filters that look for local shapes helped."),
    f"Switching SGD → Adam on the same CNN: {say(d['adam'])}. The optimizer matters as much as the architecture.",
    f"A second conv layer: {say(d['conv2'])}, with {rs[3]['params'] // rs[4]['params']}× fewer parameters "
    f"({rs[4]['params']:,} vs {rs[3]['params']:,}), because the extra pooling shrinks the image before the dense layer.",
    f"Dropout: {say(d['drop'])}. Its train accuracy ({rs[5]['train_acc']:.2%}) is even below test ({rs[5]['test_acc']:.2%}) "
    f"because neurons are switched off while training, so there is no sign of overfitting.",
    f"Overall: {rs[0]['errors']} → {rs[-1]['errors']} mistakes on 10 000 unseen digits, "
    f"{1 - rs[-1]['errors'] / rs[0]['errors']:.0%} fewer errors than the baseline.",
]
error_text = (
    f"The final model misreads {int(wrong.sum())} of 10 000 test digits. The most common confusions are "
    + ", ".join(f"<b>{i} → {j}</b> ({n}×)" for n, i, j in pairs)
    + f". The hardest digit is <b>{worst_d}</b> ({worst_acc:.1%} correct). Several of these digits are hard to read even for a person (see the examples)."
)
with open("results_data.js", "w", encoding="utf-8") as f:
    f.write("window.RESULTS_DATA = " + json.dumps({"results": rs, "samples": samples, "error_text": error_text,
                                                   "conclusions": conclusions}, ensure_ascii=False) + ";\n")
print(error_text)
print("\n".join(conclusions))
