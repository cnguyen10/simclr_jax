"""The majority of this code is adopted from
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html
"""

import jax

import flax.linen as nn
from flax.training import train_state
from flax.core import FrozenDict

import optax

import dm_pix
from chex import Array, PRNGKey

from mlx import data as dx

import numpy as np

from tqdm import tqdm

import os
import json
import logging


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict


@jax.jit
def jitter_image(key: PRNGKey, img: Array) -> Array:
    """jitter the input image

    You might have noticed that we do not use the function
    `dm_pix.random_brightness` for the brightness augmentation. The reason
    is that the implementation follows TensorFlow's brightness augmentation by
    adding a constant value to each pixel, while the implementation in
    `torchvision` and `PIL` scales the pixel values by the factor. To stay
    closer to the PyTorch implementation, we implement it ourselves by scaling
    the pixel values.
    """
    # brightness
    key, _ = jax.random.split(key=key, num=2)
    img = img * jax.random.uniform(
        key=key,
        shape=(1,),
        minval=0.5,
        maxval=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # contrast
    key, _ = jax.random.split(key=key, num=2)
    img = dm_pix.random_contrast(
        key=key,
        image=img,
        lower=0.5,
        upper=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # saturation
    key, _ = jax.random.split(key=key, num=2)
    img = dm_pix.random_saturation(
        key=key,
        image=img,
        lower=0.5,
        upper=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # hue
    key, _ = jax.random.split(key=key, num=2)
    img = dm_pix.random_hue(key=key, image=img, max_delta=0.1)
    img = jax.lax.clamp(min=0., x=img, max=1.)

    return img


@jax.jit
def crop_and_resize(key: PRNGKey, x: Array) -> Array:
    image = dm_pix.random_crop(
        key=key,
        image=x,
        crop_sizes=(x.shape[0] // 2, x.shape[1] // 2, x.shape[2])
    )
    image = jax.image.resize(image=image, shape=x.shape, method='bilinear')

    return image


@jax.jit
def augment_image(key: PRNGKey, x: Array) -> Array:
    """
    """
    # random crop and resize
    key, _ = jax.random.split(key=key, num=2)
    image = jax.lax.cond(
        jax.random.bernoulli(key=key, p=0.5),
        lambda x: crop_and_resize(key, x),
        lambda x: x,
        x
    )

    # random horizontal flipping
    key, _ = jax.random.split(key=key, num=2)
    image = dm_pix.random_flip_left_right(key=key, image=image)

    key, _ = jax.random.split(key=key, num=2)
    flag_colour_jiter = jax.random.bernoulli(key=key, p=0.8)

    img = jax.lax.cond(
        flag_colour_jiter,  # if condition
        lambda x: jitter_image(key=key, img=x),  # if true
        lambda x: x,  # if false
        image  # variable
    )

    # gray scale
    key, _ = jax.random.split(key=key, num=2)
    flag_grayscale = jax.random.bernoulli(key=key, p=0.2)
    img = jax.lax.cond(
        flag_grayscale,
        lambda x: dm_pix.rgb_to_grayscale(image=x, keep_dims=True),
        lambda x: x,
        img
    )

    # Gaussian blur
    key, _ = jax.random.split(key=key, num=2)
    sigma = jax.random.uniform(key=key, minval=0.1, maxval=2.)
    img = dm_pix.gaussian_blur(image=img, sigma=sigma, kernel_size=9)

    # normalisation
    img = 2. * img - 1.

    return img


def dataset_from_json(
    json_file: str,
    root: str = None,
    resize: tuple[int, int] = None
) -> dx._c.Buffer:
    """
    """
    with open(file=json_file, mode='r') as f:
        samples = json.load(fp=f)

    for sample in tqdm(iterable=samples, desc='dataset', colour='blue', leave=False):
        if root is not None:
            filepath = os.path.join(root, sample['file'])
        else:
            filepath = sample['file']

        # encode to ascii bytes required by mlx-data
        sample['file'] = filepath.encode('ascii')

    dataset = (
        dx.buffer_from_vector(data=samples)
        .load_image(key='file', output_key='image')
    )

    if resize is not None:
        dataset = dataset.image_resize(key='image', w=resize[0], h=resize[1])

    return dataset


def prepare_dataset(
    dataset: dx._c.Buffer,
    shuffle: bool,
    batch_size: int,
    prefetch_size: int,
    num_threads: int,
    mean: tuple[int, int, int] = None,
    std: tuple[int, int, int] = None,
    random_crop_size: tuple[int, int] = None,
    prob_random_h_flip: float = None
) -> dx._c.Buffer:
    """batch, shuffle and convert from uint8 to float32 to train

    Args:
        dataset:
        shuffle:
        batch_size:
        prefetch_size:
        num_threads:
        mean: the mean to normalised input samples (translation)
        std: the standard deviation to normalised input samples (inverse scaling)
    """
    if shuffle:
        dset = dataset.shuffle()
    else:
        dset = dataset

    # region DATA AUGMENTATION
    # randomly crop
    if random_crop_size is not None:
        dset = dset.pad(key='image', dim=0, lpad=4, rpad=4, pad_value=0)
        dset = dset.pad(key='image', dim=1, lpad=4, rpad=4, pad_value=0)
        dset = dset.image_random_crop(
            key='image',
            w=random_crop_size[0],
            h=random_crop_size[1]
        )
    
    # randomly horizontal-flip
    if prob_random_h_flip is not None:
        if prob_random_h_flip < 0 or prob_random_h_flip > 1:
            raise ValueError('Probability to randomly horizontal-flip must be in [0, 1]'
                             ', but provided with {:f}'.format(prob_random_h_flip))

        dset = dset.image_random_h_flip(key='image', prob=prob_random_h_flip)
    
    # normalisation
    if (mean is None) or (std is None):
        logging.info(
            msg='mean and std must not be None. Found one or both of them are None.'
        )

        mean = 0.
        std = 1.
    
    mean = np.array(object=mean, dtype=np.float32)
    std = np.array(object=std, dtype=np.float32)
        
    dset = dset.key_transform(
        key='image',
        func=lambda x: (x.astype('float32') / 255 + mean) / std
    )
    # endregion

    # batching, converting to stream and return
    dset = (
        dset
        .to_stream()
        .batch(batch_size=batch_size)
        .prefetch(prefetch_size=prefetch_size, num_threads=num_threads)
    )

    return dset


def initialise_huggingface_resnet(
    model: nn.Module,
    sample: Array,
    num_training_samples: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    key: PRNGKey
) -> TrainState:
    """initialise the parameters and optimiser of a model

    Args:
        sample: a sample from the dataset

    Returns:
        state:
    """
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=(num_epochs + 10) * (num_training_samples // batch_size)
    )

    # pass dummy data to initialise model's parameters
    # params = model.init(rngs=key, x=sample, train=False)
    params = model.init_weights(rng=key, input_shape=sample.shape)

    # add L2 regularisation(aka weight decay)
    weight_decay = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=5e-4,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )

    # define an optimizer
    tx = optax.chain(
        weight_decay,
        # optax.add_noise(eta=0.01, gamma=0.55, seed=random.randint(a=0, b=1_000)),
        # optax.clip_by_global_norm(max_norm=10),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state