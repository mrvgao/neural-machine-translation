## Introduction of New Tensorflow Pipeline.

```python
    dataset1 = [
        (img1, [1, 0, 0]),
        (img2, [0, 1, 0]),
        (img3, [0, 0, 1]),
        (img4, [0, 1, 0])
    ]
```

If `dataset1` would be a Tensorflow Dataset, then each Tuple is an `element`
consisting of two `components`.

We could use an `Iterator` to get element by element from this dataset.


