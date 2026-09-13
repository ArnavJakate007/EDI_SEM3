"""Sensor-balanced batch sampling for mixed-sensor pretraining.

The pretraining pool is lopsided and stays that way: GOES-19 full-disc is archived
continuously, while Himawari-8 ISatSS exists only in a few windows, so GOES supplies
several times as many triplets. Sampling that pool uniformly means most batches are
pure GOES, the loss is dominated by GOES, and the shared weights drift towards one
sensor's radiometry and motion statistics -- which is the opposite of what a
cross-sensor pretrain is for.

`BalancedSensorBatchSampler` gives every sensor an equal share of every batch instead.
The epoch length is set by the LARGEST sensor, so the majority sensor is still seen
once per epoch and the minority is oversampled to match.

That oversampling is a real trade, not a free win: with far fewer distinct Himawari
triplets, repeating them within an epoch raises the risk of memorising them. It is
still the better default here, because the alternative -- Himawari appearing in
roughly one batch in six -- produced a model that never beat a naive blend on
Himawari at all. `epoch_length="min"` is available when the minority set is large
enough that undersampling the majority is the safer trade.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import ConcatDataset, Sampler

log = logging.getLogger(__name__)


def sensor_labels(dataset) -> list[str]:
    """One sensor label per dataset index, without loading any frame data.

    Reads the tags off the dataset objects rather than iterating samples: building
    the labels must not cost a pass over the whole cache.
    """
    if isinstance(dataset, ConcatDataset):
        labels: list[str] = []
        for part in dataset.datasets:
            labels.extend(sensor_labels(part))
        return labels

    sensor = getattr(dataset, "sensor", None)
    if sensor is None:
        raise TypeError(
            f"{type(dataset).__name__} carries no `sensor` tag; balanced sampling "
            f"needs to know which sensor each index belongs to"
        )
    return [sensor] * len(dataset)


class BalancedSensorBatchSampler(Sampler[list[int]]):
    """Yields batches holding an equal share of each sensor.

    `set_epoch` re-rolls the shuffle, mirroring `TripletDataset.set_epoch`, so a
    resumed or repeated epoch is reproducible from (seed, epoch).
    """

    def __init__(
        self,
        labels: Sequence[str],
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
        epoch_length: str = "max",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if epoch_length not in ("max", "min"):
            raise ValueError(f"epoch_length must be 'max' or 'min', got {epoch_length!r}")

        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch_length = epoch_length
        self._epoch = 0

        self.groups: dict[str, np.ndarray] = {}
        for sensor in sorted(set(self.labels)):
            idx = [i for i, s in enumerate(self.labels) if s == sensor]
            self.groups[sensor] = np.asarray(idx, dtype=np.int64)

        if not self.groups:
            raise ValueError("no samples to draw from")

        self.sensors = sorted(self.groups)
        sizes = {s: len(v) for s, v in self.groups.items()}
        log.info(
            "balanced sampler over %d sensor(s): %s",
            len(self.sensors),
            ", ".join(f"{s}={n}" for s, n in sorted(sizes.items())),
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the per-sensor shuffle for the next pass."""
        self._epoch = int(epoch)

    def _per_sensor_quota(self) -> dict[str, int]:
        """How many slots each sensor gets in one batch.

        Splits the batch as evenly as the sensor count allows and hands any remainder
        to the sensors in name order, so the split is deterministic rather than
        dependent on dict iteration.
        """
        n = len(self.sensors)
        base, extra = divmod(self.batch_size, n)
        return {
            sensor: base + (1 if i < extra else 0)
            for i, sensor in enumerate(self.sensors)
        }

    def __len__(self) -> int:
        quota = self._per_sensor_quota()
        per_sensor_batches = [
            len(self.groups[s]) / max(quota[s], 1) for s in self.sensors
        ]
        count = max(per_sensor_batches) if self.epoch_length == "max" else min(
            per_sensor_batches
        )
        return int(count) if self.drop_last else int(np.ceil(count))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self._epoch))
        quota = self._per_sensor_quota()

        # An independent shuffled stream per sensor. The minority sensor's stream is
        # re-shuffled and reused when it runs out, so oversampling never repeats the
        # same order twice within an epoch.
        streams = {s: list(rng.permutation(self.groups[s])) for s in self.sensors}
        cursors = dict.fromkeys(self.sensors, 0)

        def take(sensor: str, k: int) -> list[int]:
            out: list[int] = []
            while len(out) < k:
                stream, cur = streams[sensor], cursors[sensor]
                if cur >= len(stream):
                    streams[sensor] = list(rng.permutation(self.groups[sensor]))
                    cursors[sensor] = 0
                    continue
                want = min(k - len(out), len(stream) - cur)
                out.extend(int(i) for i in stream[cur:cur + want])
                cursors[sensor] = cur + want
            return out

        for _ in range(len(self)):
            batch: list[int] = []
            for sensor in self.sensors:
                batch.extend(take(sensor, quota[sensor]))
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                return
            yield batch

    def composition(self, n_batches: int = 0) -> Counter[str]:
        """Sensor counts over `n_batches` (0 = a whole epoch). For logging."""
        counts: Counter[str] = Counter()
        for i, batch in enumerate(self):
            if n_batches and i >= n_batches:
                break
            counts.update(self.labels[j] for j in batch)
        return counts
