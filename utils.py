"""The majority of this code is adopted from
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial17/SimCLR.html
"""

import jax

import dm_pix
import chex


@jax.jit
def jitter_image(key: jax.random.KeyArray, img: chex.Array) -> chex.Array:
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
    key = jax.random.split(key=key.squeeze(), num=1)
    img = img * jax.random.uniform(
        key=key,
        shape=(1,),
        minval=0.5,
        maxval=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # contrast
    key = jax.random.split(key=key.squeeze(), num=1)
    img = dm_pix.random_contrast(
        key=key,
        image=img,
        lower=0.5,
        upper=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # saturation
    key = jax.random.split(key=key.squeeze(), num=1)
    img = dm_pix.random_saturation(
        key=key,
        image=img,
        lower=0.5,
        upper=1.5
    )
    img = jax.lax.clamp(min=0., x=img, max=1.)

    # hue
    key = jax.random.split(key=key.squeeze(), num=1)
    img = dm_pix.random_hue(key=key, image=img, max_delta=0.1)
    img = jax.lax.clamp(min=0., x=img, max=1.)

    return img


@jax.jit
def crop_and_resize(key: jax.random.KeyArray, x: chex.Array) -> chex.Array:
    image = dm_pix.random_crop(
        key=key,
        image=x,
        crop_sizes=(x.shape[0] // 2, x.shape[1] // 2, x.shape[2])
    )
    image = jax.image.resize(image=image, shape=x.shape, method='bilinear')

    return image


@jax.vmap
@jax.jit
def augment_image(key: jax.random.KeyArray, x: chex.Array) -> chex.Array:
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
