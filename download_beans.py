import tensorflow_datasets as tfds

(ds_train, ds_val, ds_test), info = tfds.load(
    "beans",
    split=["train", "validation", "test"],
    as_supervised=True,
    with_info=True
)

print(info)