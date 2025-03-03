from dataclasses import dataclass
from typing import Iterable, List, Set

import pytest
import torch.utils.data

from olmo.data import IterableDataset


def pack(values: Iterable[int]) -> List[List[int]]:
    return [[x] for x in values]


def unpack(dataset: IterableDataset) -> List[int]:
    return [x["input_ids"][0] for x in dataset]


def test_iterable_dataset_size():
    dataset = IterableDataset(pack(range(20)), 2, world_size=2, rank=0, shuffle=False)
    assert dataset.total_size == 20
    assert unpack(dataset) == list(range(0, 20, 2))

    dataset = IterableDataset(pack(range(20)), 3, world_size=3, rank=0, shuffle=False, drop_last=False)
    assert dataset.total_size == 21
    assert unpack(dataset) == list(range(0, 20, 3))

    dataset = IterableDataset(pack(range(20)), 3, world_size=3, rank=2, shuffle=False, drop_last=False)
    assert unpack(dataset) == list(range(2, 18, 3)) + [0]

    dataset = IterableDataset(pack(range(20)), 3, world_size=3, rank=0, shuffle=False, drop_last=True)
    assert dataset.total_size == 18
    assert unpack(dataset) == list(range(0, 18, 3))


def test_iterable_dataset_max_examples(tmp_path):
    device_batch_size = 2
    dataset = IterableDataset(
        pack(range(20)),
        2,
        world_size=2,
        rank=0,
        shuffle=False,
        max_examples=2 * device_batch_size * 3,
        work_dir=tmp_path,
    )
    assert unpack(dataset) == [0, 2, 4, 6, 8, 10]


def test_iterable_dataset_start_step():
    device_batch_size = 2
    dataset = IterableDataset(
        pack(range(20)), 2, world_size=2, rank=0, shuffle=False, start_index=2 * device_batch_size * 3
    )
    assert unpack(dataset) == [12, 14, 16, 18]


@pytest.mark.parametrize("world_size", [2, 4])
def test_iterable_dataset_restart_different_world_size(world_size: int):
    start_index = 4
    all_indices: Set[int] = set()
    for rank in range(world_size):
        dataset = IterableDataset(
            pack(range(20)), world_size, world_size=world_size, rank=rank, shuffle=False, start_index=start_index
        )
        indices = set(unpack(dataset))
        assert len(all_indices & indices) == 0
        all_indices.update(indices)
    assert all_indices == set(range(4, 20))


@dataclass
class MockWorkerInfo:
    id: int
    num_workers: int


@pytest.mark.parametrize("worker_id", [0, 1, 2, 3])
def test_iterable_dataset_with_workers(monkeypatch, worker_id: int):
    """
    Tests that data order is the same regardless of the number of batches.
    """
    world_size = 32
    rank = 0
    global_batch_size = 1024
    device_batch_size = global_batch_size // world_size
    all_items = list(range(1024 * 8))
    rank_items = all_items[rank::world_size]

    def patched_get_worker_info():
        return MockWorkerInfo(id=worker_id, num_workers=4)

    monkeypatch.setattr(torch.utils.data, "get_worker_info", patched_get_worker_info)
    dataset = IterableDataset(pack(all_items), global_batch_size, world_size=world_size, rank=rank, shuffle=False)
    items = unpack(dataset)
    if worker_id == 0:
        # 1st worker should get the first batch, 5th batch, etc.
        assert items[0:device_batch_size] == rank_items[0:device_batch_size]
    elif worker_id == 1:
        # 2nd worker should get the 2nd batch, 6th batch, etc.
        assert items[0:device_batch_size] == rank_items[device_batch_size : device_batch_size * 2]
    elif worker_id == 2:
        # 3rd worker should get the 3rd batch, 7th batch, etc.
        assert items[0:device_batch_size] == rank_items[device_batch_size * 2 : device_batch_size * 3]
    elif worker_id == 3:
        # 4th worker should get the 4th batch,
        assert items[0:device_batch_size] == rank_items[device_batch_size * 3 : device_batch_size * 4]


def test_iterable_dataset_with_stage():
    # Test that stage name is properly handled in indices file creation
    dataset = IterableDataset(pack(range(20)), 2, world_size=2, rank=0, shuffle=False, stage_name="stage1")
    assert dataset.stage_name == "stage1"
    assert unpack(dataset) == list(range(0, 20, 2))


def test_iterable_dataset_indices_file_with_stage(tmp_path):
    # Test that indices files are created with correct stage names
    dataset1 = IterableDataset(
        pack(range(20)),  # Same range is fine
        2,
        world_size=2,
        rank=0,
        shuffle=True,
        work_dir=tmp_path,
        stage_name="stage1",
        seed=42,  # Set different seeds to ensure different shuffling
    )
    dataset2 = IterableDataset(
        pack(range(20)),
        2,
        world_size=2,
        rank=0,
        shuffle=True,
        work_dir=tmp_path,
        stage_name="stage2",
        seed=43,  # Different seed
    )

    # Check that both indices files exist with correct names
    assert (tmp_path / "global_indices_stage1.npy").exists()
    assert (tmp_path / "global_indices_stage2.npy").exists()

    # Check that indices files are different
    indices1 = dataset1.get_global_indices()
    indices2 = dataset2.get_global_indices()

    # Check that the indices are different permutations of the same range
    assert len(indices1) == len(indices2)
    assert set(indices1) == set(indices2)  # Should contain same numbers
    assert not (indices1 == indices2).all()  # But in different order


def test_iterable_dataset_reshuffle_with_stage(tmp_path):
    # Test that reshuffling works correctly with stages
    dataset = IterableDataset(
        pack(range(20)), 2, world_size=2, rank=0, shuffle=True, work_dir=tmp_path, stage_name="stage1", seed=42
    )

    # Get initial indices
    initial_indices = dataset.get_global_indices().copy()

    # Reshuffle and verify indices changed
    dataset.reshuffle(epoch=1)
    new_indices = dataset.get_global_indices()
    assert not (initial_indices == new_indices).all()

    # Verify file still exists with correct name
    assert (tmp_path / "global_indices_stage1.npy").exists()
