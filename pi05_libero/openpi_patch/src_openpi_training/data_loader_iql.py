from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IQLTargetDataset(Dataset):
    """Adds terminal sparse reward and discounted expert return.

    For an expert episode ending at T-1:

        reward_t = 1, only if t == T-1
        target_t = gamma ** (T-1-t)

    The critic target therefore becomes larger near task completion.
    """

    def __init__(
        self,
        dataset: Dataset,
        episode_data_index,
        *,
        gamma: float,
        action_horizon: int = 50,
    ):
        self._dataset = dataset
        self._gamma = float(gamma)
        self._H = int(action_horizon)

        episode_from = np.asarray(
            episode_data_index["from"],
            dtype=np.int64,
        )

        episode_to = np.asarray(
            episode_data_index["to"],
            dtype=np.int64,
        )

        self._critic_target = np.zeros(
            len(dataset),
            dtype=np.float32,
        )

        self._critic_reward = np.zeros(
            len(dataset),
            dtype=np.float32,
        )

        self._critic_done = np.zeros(
            len(dataset),
            dtype=np.bool_,
        )

        self._remaining_steps = np.zeros(
            len(dataset),
            dtype=np.int32,
        )

        self._next_index = np.arange(
            len(dataset), dtype=np.int64
        )
        self._reward_cum = np.zeros(
            len(dataset), dtype=np.float32
        )
        self._done_chunk = np.zeros(
            len(dataset), dtype=np.bool_
        )

        for start, end in zip(
            episode_from,
            episode_to,
            strict=True,
        ):
            start = int(start)
            end = int(end)

            if end <= start:
                raise ValueError(
                    f"Invalid episode range: "
                    f"[{start}, {end})"
                )

            indices = np.arange(
                start,
                end,
                dtype=np.int64,
            )

            remaining_steps = (
                end - 1 - indices
            )

            self._remaining_steps[
                indices
            ] = remaining_steps

            self._critic_target[
                indices
            ] = np.power(
                self._gamma,
                remaining_steps,
            ).astype(np.float32)

            # Sparse reward is one only on the final transition.
            self._critic_reward[
                end - 1
            ] = 1.0

            self._critic_done[
                end - 1
            ] = True

            # chunk-wise TD:  m = steps remaining,  H = action_horizon
            #   m <  H : the chunk contains the terminal step -> r_cum = gamma**m, done=1
            #   m >= H : bootstrap -> r_cum = 0, done=0, s_{t+H}
            _H = self._H
            _term = remaining_steps < _H
            self._done_chunk[indices] = _term
            self._reward_cum[indices] = np.where(
                _term,
                np.power(self._gamma, remaining_steps),
                0.0,
            ).astype(np.float32)
            self._next_index[indices] = np.minimum(
                indices + _H, end - 1
            )

    def __getitem__(
        self,
        index: SupportsIndex,
    ) -> dict:
        item_index = int(
            index.__index__()
        )

        item = dict(
            self._dataset[index]
        )

        item["critic_target"] = np.asarray(
            self._critic_target[
                item_index
            ],
            dtype=np.float32,
        )

        item["critic_reward"] = np.asarray(
            self._critic_reward[
                item_index
            ],
            dtype=np.float32,
        )

        item["critic_done"] = np.asarray(
            self._critic_done[
                item_index
            ],
            dtype=np.bool_,
        )

        item["critic_reward_cum"] = np.asarray(
            self._reward_cum[item_index],
            dtype=np.float32,
        )

        item["critic_done_chunk"] = np.asarray(
            self._done_chunk[item_index],
            dtype=np.bool_,
        )

        item["critic_next_index"] = np.asarray(
            self._next_index[item_index],
            dtype=np.int64,
        )

        # Observation at s_{t+H} for the TD bootstrap; actions are not needed for V.
        _nx = self._dataset[
            int(self._next_index[item_index])
        ]
        for _k, _v in _nx.items():
            if _k in ("actions", "action"):
                continue
            item["next_" + _k] = _v

        item["critic_remaining_steps"] = (
            np.asarray(
                self._remaining_steps[
                    item_index
                ],
                dtype=np.int32,
            )
        )

        return item

    def __len__(self) -> int:
        return len(self._dataset)


def _find_episode_data_index(
    dataset,
):
    """Recursively unwrap transformed datasets."""

    current = dataset
    visited = set()

    while current is not None:
        object_id = id(current)

        if object_id in visited:
            break

        visited.add(object_id)

        if hasattr(
            current,
            "episode_data_index",
        ):
            return current.episode_data_index

        current = getattr(
            current,
            "_dataset",
            None,
        )

    return None


class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(data_config: _config.DataConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError("Normalization stats not found. "
                             "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`.")
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    # --- offline task-id filter (LeRobot v2.1, LIBERO) ---
    _tids = os.environ.get("OFFLINE_TASK_IDS", "")
    if _tids:
        import json as _json
        import pathlib as _pl
        import numpy as _np
        from openpi.training import online_dataset as _onl
        _want = set(int(x) for x in _tids.split(",") if x.strip())
        _m = _pl.Path.home() / ".cache/huggingface/lerobot" / data_config.repo_id / "meta"
        _t2i = {}
        for _l in open(_m / "tasks.jsonl"):
            _d = _json.loads(_l)
            _t2i[_d["task"]] = int(_d["task_index"])
        _eps = [_json.loads(_l) for _l in open(_m / "episodes.jsonl")]
        _eps.sort(key=lambda d: d["episode_index"])
        _lens = [int(_d["length"]) for _d in _eps]
        _cum = [0]
        for _L in _lens:
            _cum.append(_cum[-1] + _L)
        _keep, _nf, _nt, _cur = [], [], [], 0
        for _d in _eps:
            _ti = _t2i[_d["tasks"][0]]
            if _ti not in _want:
                continue
            _e = int(_d["episode_index"])
            _keep.extend(range(_cum[_e], _cum[_e + 1]))
            _L = _lens[_e]
            _nf.append(_cur)
            _nt.append(_cur + _L)
            _cur += _L
        _edi = {"from": _np.asarray(_nf), "to": _np.asarray(_nt)}
        print(
            f"[filter] task_ids={sorted(_want)} frames={len(_keep)}/{_cum[-1]} eps={len(_nf)}",
            flush=True,
        )
        dataset = _onl.Subset(dataset, _keep, episode_data_index=_edi)

    # --- offline task-block filter (env gate) ---
    _blocks = os.environ.get("OFFLINE_TASK_BLOCKS", "")
    if _blocks:
        import json as _json
        import pathlib as _pl
        from openpi.training import online_dataset as _onl
        _bl = [int(x) for x in _blocks.split(",") if x.strip()]
        _meta = _pl.Path.home() / ".cache/huggingface/lerobot" / data_config.repo_id / "meta/episodes.jsonl"
        _lens = [_json.loads(l)["length"] for l in open(_meta)]
        _cum = [0]
        for _L in _lens:
            _cum.append(_cum[-1] + _L)
        _keep = []
        for _b in _bl:
            _keep.extend(range(_cum[_b * 100], _cum[_b * 100 + 100]))
        import numpy as _np
        _nf, _nt, _cur = [], [], 0
        for _b in _bl:
            for _e in range(_b * 100, _b * 100 + 100):
                _L = _lens[_e]
                _nf.append(_cur)
                _nt.append(_cur + _L)
                _cur += _L
        _edi = {"from": _np.asarray(_nf), "to": _np.asarray(_nt)}
        print(f"[filter] blocks={_bl} frames={len(_keep)}/{_cum[-1]} eps={len(_nf)}", flush=True)
        dataset = _onl.Subset(dataset, _keep, episode_data_index=_edi)

    use_iql = bool(
        getattr(
            config.model,
            "use_iql",
            False,
        )
    )

    episode_data_index = None

    if use_iql:
        episode_data_index = (
            _find_episode_data_index(
                dataset
            )
        )

        if episode_data_index is None:
            raise RuntimeError(
                "Could not find episode_data_index "
                "for IQL target generation."
            )

    dataset = transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
    )

    if use_iql:
        dataset = IQLTargetDataset(
            dataset,
            episode_data_index,
            gamma=float(
                getattr(
                    config.model,
                    "critic_gamma",
                    0.99,
                )
            ),
            action_horizon=int(
                getattr(config.model, "action_horizon", 50)
            ),
        )

    _online_glob = os.environ.get("ONLINE_BUFFER_GLOB", "")
    if use_iql and _online_glob:
        from openpi.training import online_dataset as _online
        _raw_on = _online.OnlineEpisodeDataset(
            _online_glob,
            gamma=float(getattr(config.model, "critic_gamma", 0.99)),
            action_horizon=int(getattr(config.model, "action_horizon", 50)),
            frame_stride=int(os.environ.get("ONLINE_FRAME_STRIDE", "25")),
        )
        _on_t = transform_dataset(_raw_on, data_config, skip_norm_stats=skip_norm_stats)
        _on = _online.OnlineWithTargets(_on_t, _raw_on)
        dataset = _online.MixedDataset(
            dataset, _on,
            offline_ratio=int(os.environ.get("OFFLINE_RATIO", "1")),
            online_ratio=int(os.environ.get("ONLINE_RATIO", "1")),
        )
        print(f"[off2on] mixed loader: offline_ratio={os.environ.get('OFFLINE_RATIO','1')}:1")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):

        def __init__(
            self,
            data_config: _config.DataConfig,
            data_loader: TorchDataLoader,
            use_iql: bool,
        ):
            self._data_config = data_config
            self._data_loader = data_loader
            self._use_iql = use_iql

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                observation = (
                    _model.Observation.from_dict(
                        batch
                    )
                )

                if self._use_iql:
                    # Observation at s_{t+H} for the TD bootstrap
                    _nb = {
                        k[len("next_"):]: v
                        for k, v in batch.items()
                        if k.startswith("next_")
                    }
                    next_observation = (
                        _model.Observation.from_dict(_nb)
                        if _nb
                        else None
                    )
                    yield (
                        observation,
                        batch["actions"],
                        batch["critic_target"],
                        next_observation,
                        batch["critic_reward_cum"],
                        batch["critic_done_chunk"],
                    )
                else:
                    yield (
                        observation,
                        batch["actions"],
                    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        use_iql,
    )


class TorchDataLoader:

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B", )),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
