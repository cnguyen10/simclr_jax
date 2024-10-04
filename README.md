[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## SimCLR in Jax
SimCLR is implemented in Jax. In addition, because Jax does not have a native data loader, the light-weight `mlx-data` is used as a drop-in replacement for the bulky `tensorflow-datasets` (which also requires to install `tensorflow`). The experiment management `aim` is also replaced by `mlflow`.

### Dataset
Instead of pointing to a folder containing samples, the data loading mechanism is designed revolved around json files. To run the code, ensure that the json files pointing to samples have the following structure:
```
[
    {
        "file": "train/19/bos_taurus_s_000507.png",
        "label": 19,
        "superclass": 11
    },
    {
        "file": "train/29/stegosaurus_s_000125.png",
        "label": 29,
        "superclass": 15
    }
]
```
The `file` path above is relative, and hence, requires `root` specified in the `conf.yaml` file to load data.