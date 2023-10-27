import jax
import jax.dlpack
from jax import numpy as jnp

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state, orbax_utils
import orbax.checkpoint as ocp

import optax
import chex

import clu.metrics

import tensorflow as tf

from functools import partial

import aim

import random

import os
import argparse
import logging
from typing import Any
from pathlib import Path
import subprocess
from tqdm import tqdm

from PreactResnet import ResNet18
from utils import augment_image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment-name', type=str, help='')

    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Path to the folder containing train and test sets'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='logs',
        help='Folder to store logs'
    )

    parser.add_argument(
        '--image-shape',
        type=int,
        action='append',
        help='e.g., 32 32 3 or 224 224 3'
    )

    parser.add_argument(
        '--repr-dim',
        type=int,
        default=128,
        help='The dimension of the representation'
    )
    parser.add_argument('--temperature', type=float)

    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=200)

    parser.add_argument('--jax-mem-fraction', type=float, default=0.5)

    parser.add_argument('--tqdm', dest='tqdm_flag', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm_flag', action='store_false')
    parser.set_defaults(tqdm_flag=True)

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument(
        '--run-hash-id',
        type=str,
        default=None,
        help='Hash id of the run to resume'
    )

    args = parser.parse_args()

    return args


class TrainState(train_state.TrainState):
    batch_stats: Any


@jax.jit
def info_NCE_loss_from_features(
    features: chex.Array,
    temperature: chex.Numeric
) -> chex.Numeric:
    batch_size = features.shape[0]

    # calculate cosine similarity
    cos_sim = optax.cosine_similarity(
        predictions=jnp.expand_dims(a=features, axis=1),
        targets=jnp.expand_dims(a=features, axis=0)
    )
    cos_sim = cos_sim / temperature

    # mask cosine similarity of sample itself since it is worthless to
    # maximise similarity between a sample itself
    diagonal_indices = jnp.arange(start=0, stop=batch_size, dtype=jnp.int32)
    cos_sim = cos_sim.at[diagonal_indices, diagonal_indices].set(
        values=-jnp.inf
    )

    # find positive samples (diagonals of bottom-left and top-right blocks)
    shifted_diagonal_indicess = jnp.roll(
        a=diagonal_indices,
        shift=batch_size // 2
    )
    pos_logits = cos_sim[diagonal_indices, shifted_diagonal_indicess]

    # InfoNCE loss
    loss = nn.logsumexp(a=cos_sim, axis=-1) - pos_logits
    loss = jnp.mean(a=loss)

    return loss


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: chex.Array,
    keys: jax.random.PRNGKeyArray,
    temperature: float
) -> tuple[train_state.TrainState, chex.Numeric]:
    """
    """
    # x = jax.vmap(
    #     fun=augment_image,
    #     in_axes=(0, 0)
    # )(keys, jnp.concatenate(arrays=(x, x), axis=0))
    x = augment_image(keys, jnp.concatenate(arrays=(x, x), axis=0))

    # define loss function
    def info_NCE_loss(
        params: FrozenDict,
        batch_stats: FrozenDict
    ) -> chex.Numeric:
        """
        """
        features, batch_stats_new = state.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )

        loss = info_NCE_loss_from_features(features, temperature)

        return loss, batch_stats_new

    # grad and loss
    grad_value_fn = jax.value_and_grad(
        fun=info_NCE_loss,
        argnums=0,
        has_aux=True
    )
    (loss, batch_stats_new), grads = grad_value_fn(
        state.params,
        state.batch_stats
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=batch_stats_new['batch_stats'])

    return state, loss


def train_epoch(
    state: train_state.TrainState,
    ds_train: tf.data.Dataset,
    args: argparse.Namespace
) -> tuple[train_state.TrainState, chex.Numeric]:
    """
    """
    # metrics to monitor
    loss_avg = clu.metrics.Average(
        total=jnp.array(0., dtype=jnp.float32),
        count=jnp.array(0, dtype=jnp.int32)
    )

    for imgs in tqdm(iterable=ds_train, desc='Train', leave=False, position=1):
        # move to GPU
        with tf.device(device_name='/gpu:0'):
            imgs = tf.identity(input=imgs, name='raw_image_input')

        # move to DLPack
        x_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=imgs)

        # load data from DLPack
        x = jax.dlpack.from_dlpack(x_dl)

        args.key, _ = jax.random.split(key=args.key, num=2)
        keys = jax.random.split(key=args.key, num=x.shape[0] * 2)
        state, loss = train_step(
            state=state,
            x=x,
            keys=keys,
            temperature=args.temperature
        )

        loss_avg = clu.metrics.Average.merge(
            self=loss_avg,
            other=clu.metrics.Average.from_model_output(values=loss)
        )

    return state, loss_avg.compute()


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    x: chex.Array,
    keys: jax.random.PRNGKeyArray,
    crop_size: tuple[int, int, int],
    temperature: float
) -> chex.Array:
    """
    """
    # augment images
    x = jnp.concatenate(
        arrays=[jax.vmap(
            fun=partial(augment_image, crop_size=crop_size),
            in_axes=(0, 0))(keys[i], x) for i in range(2)
        ],
        axis=0
    )

    features, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )

    loss = info_NCE_loss_from_features(features, temperature)

    return loss


def evaluate(
    state: train_state.TrainState,
    ds_test: tf.data.Dataset,
    args: argparse.Namespace
) -> chex.Numeric:
    """
    """
    # metrics to monitor
    loss_avg = clu.metrics.Average(
        total=jnp.array(0., dtype=jnp.float32),
        count=jnp.array(0, dtype=jnp.int32)
    )

    for imgs in tqdm(
        iterable=ds_test,
        desc='Evaluate',
        leave=False,
        position=1
    ):
        # move to GPU
        with tf.device(device_name='/gpu:0'):
            imgs = tf.identity(input=imgs, name='raw_image_input')

        # move to DLPack
        x_dl = tf.experimental.dlpack.to_dlpack(tf_tensor=imgs)

        # load data from DLPack
        x = jax.dlpack.from_dlpack(x_dl)

        args.key = jax.random.split(key=args.key, num=1).squeeze()
        keys = jax.random.split(key=args.key, num=x.shape[0] * 2)
        keys = jnp.split(ary=keys, indices_or_sections=2, axis=0)

        loss = evaluate_step(state, x, keys, args.crop_size, args.temperature)

        loss_avg = clu.metrics.Average.merge(
            self=loss_avg,
            other=clu.metrics.Average.from_model_output(values=loss)
        )

    return loss_avg.compute()


def preprocess_images(image: tf.Tensor) -> tf.Tensor:
    """
    """
    x = tf.cast(x=image, dtype=tf.float32) / 255.

    return x


def main() -> None:
    """
    """
    args = parse_arguments()
    args.key = jax.random.PRNGKey(seed=random.randint(a=0, b=1_000))

    # region CONFIGURATION
    # set jax memory allocation
    assert args.jax_mem_fraction < 1. and args.jax_mem_fraction > 0.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.jax_mem_fraction)

    # limit GPU memory for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        device=gpus[0],
        logical_devices=[
            tf.config.LogicalDeviceConfiguration(memory_limit=1024)
        ]
    )
    # endregion

    # region DATASETS
    assert len(args.image_shape) == 3
    args.image_shape = tuple(args.image_shape)

    ds_train = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(args.dataset_root, 'train'),
        labels=None,  # no label
        label_mode=None,
        batch_size=args.batch_size,
        image_size=args.image_shape[:-1],
        shuffle=True
    )
    ds_train = ds_train.map(map_func=preprocess_images)

    ds_test = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(args.dataset_root, 'test'),
        labels=None,
        label_mode=None,
        batch_size=args.batch_size,
        image_size=args.image_shape[:-1],
        shuffle=False
    )
    ds_test = ds_test.map(map_func=preprocess_images)
    # endregion

    # region MODEL
    model = ResNet18(num_classes=args.repr_dim)
    x = jax.random.normal(
        key=args.key,
        shape=(1,) + args.image_shape,
        dtype=jnp.float32
    )
    params = model.init(rngs=args.key, x=x, train=False)

    # optimiser
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.num_epochs * len(ds_train)
    )
    tx = optax.chain(
        optax.add_noise(eta=0.01, gamma=0.55, seed=random.randint(a=0, b=100)),
        optax.adamaxw(
            learning_rate=lr_schedule_fn,
            weight_decay=args.weight_decay
        )
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params.get('batch_stats'),
        tx=tx
    )
    # endregion

    # region EXPERIMENT TRACKING
    # create log folder if it does not exist
    logging.info(msg='Initialise AIM repository to store logs')
    if not os.path.exists(path=args.logdir):
        logging.info(
            msg=f'Logging folder not found. Make a logdir at {args.logdir}'
        )
        Path(args.logdir).mkdir(parents=True, exist_ok=True)

    if not aim.sdk.repo.Repo.exists(path=args.logdir):
        logging.info(msg='Initialize AIM repository')
        # aim.sdk.repo.Repo(path=args.logdir, read_only=False, init=True)
        subprocess.run(args=["aim", "init"])

    exp_tracker = aim.Run(
        run_hash=args.run_hash_id,
        repo=args.logdir,
        read_only=False,
        experiment=args.experiment_name,
        force_resume=False,
        capture_terminal_logs=False,
        system_tracking_interval=600  # capture every x seconds
    )
    exp_tracker['hparams'] = {key: args.__dict__[key]
                              for key in args.__dict__
                              if isinstance(args.__dict__[key],
                                            (int, bool, str, float, list))}

    # create a folder with the corresponding hash run id to store checkpoints
    args.checkpoint_dir = os.path.join(args.logdir, exp_tracker.hash)
    if not os.path.exists(path=args.checkpoint_dir):
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # endregion

    # region SETUP CHECKPOINT and RESTORE
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        save_interval_steps=1
    )
    checkpoint_mngr = ocp.CheckpointManager(
        directory=args.checkpoint_dir,
        checkpointers={'state': ocp.PyTreeCheckpointer()},
        options=checkpoint_options
    )
    # endregion

    if args.resume:
        # must be associated with a run hash id
        assert args.run_hash_id is not None

        # default restore will be in raw dictionary
        # create an example structure to restore to the dataclass of interest
        # checkpoint_example = {'state': state}

        restored = checkpoint_mngr.restore(
            step=checkpoint_mngr.latest_step(),
            # items=checkpoint_example
        )

        state = restored['state']
        # state = TrainState.create(
        #     apply_fn=model.apply,
        #     params=restored['state']['params'],
        #     batch_stats=restored['state']['batch_stats'],
        #     tx=tx
        # )

    try:
        for epoch_id in tqdm(
            iterable=range(0, args.num_epochs),
            desc='Progress',
            position=0
        ):
            state, loss_train = train_epoch(
                state=state,
                ds_train=ds_train,
                args=args
            )

            # # evaluation
            # loss_test = evaluate(
            #     state=state,
            #     ds_test=ds_test,
            #     args=args
            # )

            # tracking
            exp_tracker.track(
                value=loss_train,
                name='Loss',
                context={'subset': 'train'}
            )
            # exp_tracker.track(
            #     value=loss_test,
            #     name='Loss',
            #     context={'subset': 'test'}
            # )

            # save checkpoint
            checkpoint = {'state': state}
            checkpoint_mngr.save(
                step=epoch_id + 1,
                items=checkpoint,
                save_kwargs={
                    'save_args': orbax_utils.save_args_from_target(checkpoint)
                }
            )
            del checkpoint
    finally:
        exp_tracker.close()


if __name__ == '__main__':
    logger_current = logging.getLogger(name=__name__)
    logger_current.setLevel(level=logging.INFO)

    main()
