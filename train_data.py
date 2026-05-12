from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.datasets as datasets
import torch

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY = True


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY,
        download=True,
        transform=get_transforms(train=True),
    )

    # Balanced sampler: each class gets equal expected frequency per batch.
    # With batch_size < 100, not every class appears in every batch, but
    # over many steps all classes are seen equally often.
    targets = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets)
    sample_weights = 1.0 / class_counts[targets].float()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    return train_dataset, train_loader
