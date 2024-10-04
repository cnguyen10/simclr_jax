import jax
import jax.dlpack
from jax import numpy as jnp

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import orbax.checkpoint as ocp
from flax.nnx import metrics

import optax
from chex import Array, Numeric, PRNGKey

from mlx import data as dx

from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
import flatdict

import mlflow

import random

import os
from pathlib import Path
from tqdm import tqdm

from PreactResnet import ResNet18 as resnet18
from utils import (
    augment_image,
    initialise_huggingface_resnet,
    TrainState,
    dataset_from_json,
    prepare_dataset
)


@partial(jax.jit, static_argnames=('temperature',))
def info_NCE_loss_from_features(features: Array, temperature: Numeric) -> Numeric:
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


def augment_images(x: Array, keys: PRNGKey) -> Array:
    """batch version of `augment_image`
    """
    return jax.vmap(fun=augment_image, in_axes=(0, 0))(keys, x)


@partial(jax.jit, static_argnames=('temperature',), donate_argnames=('state',))
def train_step(
    x: Array,
    state: TrainState,
    keys: PRNGKey,
    temperature: float
) -> tuple[TrainState, Numeric]:
    """
    """
    x = augment_images(x=jnp.concatenate(arrays=(x, x), axis=0), keys=keys)

    # define loss function
    def info_NCE_loss(params: FrozenDict, batch_stats: FrozenDict) -> Numeric:
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


def train(
    state: TrainState,
    dataset: dx._c.Buffer,
    key: PRNGKey,
    cfg: DictConfig
) -> tuple[TrainState, Numeric, PRNGKey]:
    """
    """
    # metrics to monitor
    loss_accum = metrics.Average()

    # batching the dataset
    data_stream = prepare_dataset(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.hparams.batch_size,
        prefetch_size=cfg.hparams.prefetch_size,
        num_threads=cfg.hparams.num_threads,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    for samples in tqdm(
        iterable=data_stream,
        desc='train',
        total=len(dataset) // cfg.hparams.batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.hparams.progress_bar
    ):
        x = jnp.array(object=samples['image'], dtype=jnp.float32)

        key, _ = jax.random.split(key=key, num=2)
        keys = jax.random.split(key=key, num=x.shape[0] * 2)
        state, loss = train_step(
            x=x,
            state=state,
            keys=keys,
            temperature=cfg.hparams.temperature
        )

        loss_accum.update(values=loss)

    return state, loss_accum.compute(), key


@partial(jax.jit, static_argnames=('temperature',))
def evaluate_step(
    x: Array,
    state: TrainState,
    keys: PRNGKey,
    temperature: float
) -> Array:
    """
    """
    # augment images
    x = augment_images(x=jnp.concatenate(arrays=(x, x), axis=0), keys=keys)

    features, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )

    loss = info_NCE_loss_from_features(features, temperature)

    return loss


def evaluate(dataset: dx._c.Buffer, state: TrainState, key: PRNGKey, cfg: DictConfig) -> Numeric:
    """
    """
    # metrics to monitor
    loss_accum = metrics.Average()

    # batching the dataset
    data_stream = prepare_dataset(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.hparams.batch_size,
        prefetch_size=cfg.hparams.prefetch_size,
        num_threads=cfg.hparams.num_threads,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    for samples in tqdm(
        iterable=data_stream,
        desc='test',
        total=len(dataset) // cfg.hparams.batch_size + 1,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.hparams.progress_bar
    ):
        x = jnp.array(object=samples['image'], dtype=jnp.float32)

        key, _ = jax.random.split(key=key, num=2)
        keys = jax.random.split(key=key, num=x.shape[0] * 2)

        loss = evaluate_step(
            x=x,
            state=state,
            keys=keys,
            crop_size=cfg.dataset.crop_size,
            temperature=cfg.hparams.temperature
        )

        loss_accum.update(values=loss)

    return loss_accum.compute()


@hydra.main(version_base=None, config_path='.', config_name='conf')
def main(cfg: DictConfig) -> None:
    """
    """
    # region Jax's configuration
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )
    # endregion

    # region DATASETS
    dataset_train = dataset_from_json(
        json_file=cfg.dataset.train_file,
        root=cfg.dataset.root,
        resize=cfg.dataset.resized_shape
    )

    dataset_test = dataset_from_json(
        json_file=cfg.dataset.test_file,
        root=cfg.dataset.root,
        resize=cfg.dataset.resized_shape
    )
    # endregion

    # region MODEL
    state = initialise_huggingface_resnet(
        model=resnet18(
            num_classes=cfg.hparams.repr_dim,
            input_shape=(1,) + tuple(cfg.dataset.crop_size) + (dataset_train[0]['image'].shape[-1],),
            dtype=jnp.bfloat16
        ),
        sample=jnp.expand_dims(a=dataset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dataset_train),
        lr=cfg.hparams.lr,
        batch_size=cfg.hparams.batch_size,
        num_epochs=cfg.hparams.num_epochs,
        key=jax.random.key(seed=random.randint(a=0, b=10_000))
    )

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=50,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )
    # endregion

    # region EXPERIMENT TRACKING
    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()
    # mlflow.set_system_metrics_sampling_interval(interval=600)
    # mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)
    # endregion

    with mlflow.start_run(run_id=cfg.experiment.run_id, log_system_metrics=False) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(
            os.getcwd(),
            cfg.experiment.logdir,
            cfg.experiment.name,
            mlflow_run.info.run_id
        )

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:

            if cfg.experiment.run_id is None:
                start_epoch_id = 0

                # log hyper-parameters
                mlflow.log_params(
                    params=flatdict.FlatDict(
                        value=OmegaConf.to_container(cfg=cfg),
                        delimiter='.'
                    )
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )
            else:
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.StandardRestore(item=state)
                )

                state = checkpoint

                del checkpoint
            
            # generate a random key
            seed = random.randint(a=0, b=127)
            key = jax.random.PRNGKey(seed=seed)

            for epoch_id in tqdm(
                iterable=range(start_epoch_id, cfg.hparams.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not cfg.hparams.progress_bar
            ):
                state, loss_train, key = train(
                    dataset=dataset_train,
                    state=state,
                    key=key,
                    cfg=cfg
                )

                # wait until completing the asynchronous saving
                ckpt_mngr.wait_until_finished()

                # save parameters asynchronously
                ckpt_mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.StandardSave(state)
                )

                # evaluation
                loss_test = evaluate(dataset=dataset_test, state=state, key=key, cfg=cfg)

                # tracking
                mlflow.log_metrics(
                    metrics={
                        'loss/train': loss_train,
                        'loss/test': loss_test
                    },
                    step=epoch_id + 1,
                    synchronous=False
                )


if __name__ == '__main__':
    main()
