from typing import Any, Dict, Optional, Callable

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import MushroomDataset
from src.transforms import instantiate_datamodule_transforms


class MushroomDataModule(LightningDataModule):
    """`LightningDataModule` for the Mushroom dataset. This is the
    pytorch lightning class that holds onto your train, val, and test
    datasets.

    Data source:
    https://www.kaggle.com/datasets/zlatan599/mushroom1

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to
    download, split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_transform: Callable = None,
        val_transform: Callable = None,
        test_transform: Callable = None,
        mini: int = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ) -> None:
        """Initialize a `MushroomDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_transform (callable, optional): Transform to apply
            to train images.
        :param val_transform (callable, optional): Transform to apply
            to val images.
        :param test_transform (callable, optional): Transform to apply
            to test images.
        :param mini (int, optional): If set, build a mini train dataset
            with at most `mini` samples per class. The val and test
            datasets are left untouched.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # Do not set the transforms directly, use self.set_transforms()
        # instead to parse the input configs
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

        # Instantiate the transform
        self.set_transforms()

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MushroomDataset classes.
        """
        return self.data_train.num_classes

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        MushroomDataset(
            self.hparams.data_dir,
            split='train',
            mini=self.hparams.mini)
        MushroomDataset(
            self.hparams.data_dir,
            split='val')
        MushroomDataset(
            self.hparams.data_dir,
            split='test')

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = MushroomDataset(
                self.hparams.data_dir,
                split='train',
                mini=self.hparams.mini,
                transform=self.train_transform)
            self.data_val = MushroomDataset(
                self.hparams.data_dir,
                split='val',
                transform=self.val_transform)
            self.data_test = MushroomDataset(
                self.hparams.data_dir,
                split='test',
                transform=self.test_transform)

    def set_transforms(self) -> None:
        """Parse in self.hparams in search for '*transform*' keys and
        instantiate the corresponding transforms.
        """
        t_dict = instantiate_datamodule_transforms(self.hparams)
        for key, transform in t_dict.items():
            setattr(self, key, transform)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MushroomDataModule()
