import os
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


MUSHROOM_CLASS_NAMES = [
    'Agaricus augustus',
    'Agaricus xanthodermus',
    'Amanita amerirubescens',
    'Amanita augusta',
    'Amanita brunnescens',
    'Amanita calyptroderma',
    'Amanita citrina',
    'Amanita flavoconia',
    'Amanita muscaria',
    'Amanita pantherina',
    'Amanita persicina',
    'Amanita phalloides',
    'Amanita rubescens',
    'Amanita velosa',
    'Apioperdon pyriforme',
    'Armillaria borealis',
    'Armillaria mellea',
    'Armillaria tabescens',
    'Artomyces pyxidatus',
    'Bjerkandera adusta',
    'Bolbitius titubans',
    'Boletus edulis',
    'Boletus pallidus',
    'Boletus reticulatus',
    'Boletus rex-veris',
    'Calocera viscosa',
    'Calycina citrina',
    'Cantharellus californicus',
    'Cantharellus cibarius',
    'Cantharellus cinnabarinus',
    'Cerioporus squamosus',
    'Cetraria islandica',
    'Chlorociboria aeruginascens',
    'Chlorophyllum brunneum',
    'Chlorophyllum molybdites',
    'Chondrostereum purpureum',
    'Cladonia fimbriata',
    'Cladonia rangiferina',
    'Cladonia stellaris',
    'Clitocybe nebularis',
    'Clitocybe nuda',
    'Coltricia perennis',
    'Coprinellus disseminatus',
    'Coprinellus micaceus',
    'Coprinopsis atramentaria',
    'Coprinopsis lagopus',
    'Coprinus comatus',
    'Crucibulum laeve',
    'Cryptoporus volvatus',
    'Daedaleopsis confragosa',
    'Daedaleopsis tricolor',
    'Entoloma abortivum',
    'Evernia mesomorpha',
    'Evernia prunastri',
    'Flammulina velutipes',
    'Fomes fomentarius',
    'Fomitopsis betulina',
    'Fomitopsis mounceae',
    'Fomitopsis pinicola',
    'Galerina marginata',
    'Ganoderma applanatum',
    'Ganoderma curtisii',
    'Ganoderma oregonense',
    'Ganoderma tsugae',
    'Gliophorus psittacinus',
    'Gloeophyllum sepiarium',
    'Graphis scripta',
    'Grifola frondosa',
    'Gymnopilus luteofolius',
    'Gyromitra esculenta',
    'Gyromitra gigas',
    'Gyromitra infula',
    'Hericium coralloides',
    'Hericium erinaceus',
    'Hygrophoropsis aurantiaca',
    'Hypholoma fasciculare',
    'Hypholoma lateritium',
    'Hypogymnia physodes',
    'Hypomyces lactifluorum',
    'Imleria badia',
    'Inonotus obliquus',
    'Ischnoderma resinosum',
    'Kuehneromyces mutabilis',
    'Laccaria ochropurpurea',
    'Lactarius deliciosus',
    'Lactarius torminosus',
    'Lactarius turpis',
    'Laetiporus sulphureus',
    'Leccinum albostipitatum',
    'Leccinum aurantiacum',
    'Leccinum scabrum',
    'Leccinum versipelle',
    'Lepista nuda',
    'Leratiomyces ceres',
    'Leucoagaricus americanus',
    'Leucoagaricus leucothites',
    'Lobaria pulmonaria',
    'Lycogala epidendrum',
    'Lycoperdon perlatum',
    'Lycoperdon pyriforme',
    'Macrolepiota procera',
    'Merulius tremellosus',
    'Mutinus ravenelii',
    'Mycena haematopus',
    'Mycena leaiana',
    'Nectria cinnabarina',
    'Omphalotus illudens',
    'Omphalotus olivascens',
    'Panaeolus papilionaceus',
    'Panellus stipticus',
    'Parmelia sulcata',
    'Paxillus involutus',
    'Peltigera aphthosa',
    'Peltigera praetextata',
    'Phaeolus schweinitzii',
    'Phaeophyscia orbicularis',
    'Phallus impudicus',
    'Phellinus igniarius',
    'Phellinus tremulae',
    'Phlebia radiata',
    'Phlebia tremellosa',
    'Pholiota aurivella',
    'Pholiota squarrosa',
    'Phyllotopsis nidulans',
    'Physcia adscendens',
    'Platismatia glauca',
    'Pleurotus ostreatus',
    'Pleurotus pulmonarius',
    'Psathyrella candolleana',
    'Pseudevernia furfuracea',
    'Pseudohydnum gelatinosum',
    'Psilocybe azurescens',
    'Psilocybe caerulescens',
    'Psilocybe cubensis',
    'Psilocybe cyanescens',
    'Psilocybe ovoideocystidiata',
    'Psilocybe pelliculosa',
    'Retiboletus ornatipes',
    'Rhytisma acerinum',
    'Sarcomyxa serotina',
    'Sarcoscypha austriaca',
    'Sarcosoma globosum',
    'Schizophyllum commune',
    'Stereum hirsutum',
    'Stereum ostrea',
    'Stropharia aeruginosa',
    'Stropharia ambigua',
    'Suillus americanus',
    'Suillus granulatus',
    'Suillus grevillei',
    'Suillus luteus',
    'Suillus spraguei',
    'Tapinella atrotomentosa',
    'Trametes betulina',
    'Trametes gibbosa',
    'Trametes hirsuta',
    'Trametes ochracea',
    'Trametes versicolor',
    'Tremella mesenterica',
    'Trichaptum biforme',
    'Tricholoma murrillianum',
    'Tricholomopsis rutilans',
    'Tylopilus felleus',
    'Tylopilus rubrobrunneus',
    'Urnula craterium',
    'Verpa bohemica',
    'Volvopluteus gloiocephalus',
    'Vulpicida pinastri',
    'Xanthoria parietina'
]

MUSHROOM_TO_TAXOPY = {k: k for k in MUSHROOM_CLASS_NAMES}
MUSHROOM_TO_TAXOPY['Amanita amerirubescens'] = 'Amanita rubescens'
MUSHROOM_TO_TAXOPY['Armillaria tabescens'] = 'Desarmillaria tabescens'
MUSHROOM_TO_TAXOPY['Boletus pallidus'] = 'Imleria pallida'
MUSHROOM_TO_TAXOPY['Clitocybe nuda'] = 'Clitocybe'
MUSHROOM_TO_TAXOPY['Daedaleopsis confragosa'] = 'Daedaleopsis tricolor'
MUSHROOM_TO_TAXOPY['Ganoderma tsugae'] = 'Ganoderma'
MUSHROOM_TO_TAXOPY['Gyromitra gigas'] = 'Gyromitra'
MUSHROOM_TO_TAXOPY['Gyromitra infula'] = 'Gyromitra'
MUSHROOM_TO_TAXOPY['Lepista nuda'] = 'Lepista'
MUSHROOM_TO_TAXOPY['Leucoagaricus americanus'] = 'Leucoagaricus cf. americanus'
MUSHROOM_TO_TAXOPY['Leucoagaricus leucothites'] = 'Leucoagaricus naucinus'
MUSHROOM_TO_TAXOPY['Lycoperdon pyriforme'] = 'Apioperdon pyriforme'
MUSHROOM_TO_TAXOPY['Merulius tremellosus'] = 'Phlebia tremellosa'
MUSHROOM_TO_TAXOPY['Psathyrella candolleana'] = 'Psathyrella delineata'
MUSHROOM_TO_TAXOPY['Trametes betulina'] = 'Trametes cinnabarina'
MUSHROOM_TO_TAXOPY['Trichaptum biforme'] = 'Trichaptum'

TAXOPY_TO_MUSHROOM = {v: k for k, v in MUSHROOM_TO_TAXOPY.items()}


class MushroomDataset(Dataset):
    """`Dataset` for the Mushroom dataset.

    Data source:
    https://www.kaggle.com/datasets/zlatan599/mushroom1
    """

    def __init__(
            self,
            data_dir,
            split='train',
            transform=None,
            target_transform=None,
            mini=None
    ):
        """
        :param data_dir (str): Path to the dataset root directory.
        :param split (str): 'train', 'val' or 'test'
        :param transform (callable, optional): Transform to apply to images.
        :param target_transform (callable, optional): Transform to apply to labels.
        :param mini (int, optional): If set, build a mini dataset with at most
            `mini` samples per class.
        """
        SPLITS = ['train', 'val', 'test']
        if split not in SPLITS:
            raise NotImplementedError(
                f"Unknown split '{split}'. Expected one of {SPLITS}")

        self.class_names = MUSHROOM_CLASS_NAMES
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.root = data_dir
        self.split = split
        self.csv_file = os.path.join(self.root, f"{split}.csv")
        self.mini = mini

        # Read the entire dataset
        data = pd.read_csv(self.csv_file)

        # If mini is provided, subsample mini elements for each class
        if mini is not None:
            sampled_data = []
            for cls in self.class_names:
                cls_data = data[data.iloc[:, 1] == cls]
                sampled_data.append(
                    cls_data.sample(
                        n=min(len(cls_data), mini),
                        replace=False,
                        random_state=42))
            data = pd.concat(sampled_data).reset_index(drop=True)

        # Save the data, paths, and labels as attributes
        self.data = data
        self.file_paths = [
            os.path.join(self.root, x.split('/kaggle/working/')[1])
            for x in self.data.iloc[:, 0].values]
        self.targets = [
            self.class_to_idx[label] for label in self.data.iloc[:, 1].values]

        # Transforms
        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.targets[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __repr__(self):
        head = f"Dataset {self.__class__.__name__}"
        body = [
            f"    Split: {self.split}",
            f"    Number of images: {self.__len__()}",
            f"    CSV file: {os.path.basename(self.csv_file)}",
            f"    Number of classes: {self.num_classes}",
            f"    mini: {self.mini}" * (self.mini is not None),
        ]
        return "\n".join([head] + body)

    @property
    def num_classes(self):
        return len(self.class_names)

    def show_samples(self, n=5, height=3, class_filter=None):
        """Display n random samples from the dataset with labels.

        :param n (int): Number of samples to show.
        :param class_filter (int or str, optional):
            - if int: class index
            - if str: class name
            - if None: sample from all classes
        """
        if class_filter is not None:
            if isinstance(class_filter, int):
                # ensure valid class index
                if class_filter not in self.class_to_idx.values():
                    raise ValueError(f"Invalid class index: {class_filter}")
                valid_indices = [
                    i
                    for i, t in enumerate(self.targets)
                    if t == class_filter]
                class_name = self.class_names[class_filter]
            elif isinstance(class_filter, str):
                lowercase_labels = [x.lower() for x in self.class_to_idx.keys()]
                if class_filter.lower() not in lowercase_labels:
                    raise ValueError(f"Invalid class name: {class_filter}")
                valid_indices = [
                    i
                    for i, t in enumerate(self.data.iloc[:, 1])
                    if t.lower() == class_filter.lower()]
                class_name = class_filter
            else:
                raise TypeError("class_filter must be int (index) or str (name).")
        else:
            valid_indices = list(range(len(self)))
            class_name = None

        if not valid_indices:
            raise ValueError("No samples found for the given class filter.")

        indices = random.sample(valid_indices, k=min(n, len(valid_indices)))
        plt.figure(figsize=(n * height, height))

        for i, idx in enumerate(indices):
            img_path = self.file_paths[idx]
            label = self.data.iloc[idx, 1]  # class name

            # Load image (raw, without transform for visualization)
            image = Image.open(img_path).convert("RGB")

            plt.subplot(1, n, i + 1)
            plt.imshow(image)
            plt.title(label, fontsize=10)
            plt.axis("off")

        suptitle = f"Samples from class '{class_name}'" if class_name else "Random samples"
        plt.suptitle(suptitle, fontsize=12)
        plt.tight_layout()
        plt.show()
