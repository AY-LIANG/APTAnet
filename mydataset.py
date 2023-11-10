import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem

#用于标准化SMILES
class Canonicalization():
    """Convert any SMILES to RDKit-canonical SMILES.
    Example:
        An example::

            smiles = 'CN2C(=O)N(C)C(=O)C1=C2N=CN1C'
            c = Canonicalization()
            c(smiles)

        Result is: 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'

    """

    def __init__(self, sanitize: bool = True) -> None:
        """Initialize a canonicalizer

        Args:
            sanitize (bool, optional): Whether molecule is sanitized. Defaults to True.
        """
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> str:
        """
        Forward function of canonicalization.

        Args:
            smiles (str): SMILES string for canonicalization.

        Returns:
            str: Canonicalized SMILES string.
        """
        try:
            canon = Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles, sanitize=self.sanitize), canonical=True
            )
            return canon
        except Exception:
            print(f'\nInvalid SMILES {smiles}, no canonicalization done')
            return smiles

#数据增强SMILES
class Augment():
    """Augment a SMILES string, according to Bjerrum (2017)."""

    def __init__(
        self,
        kekule_smiles: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        sanitize: bool = True,
        seed: int = -1,
    ) -> None:
        """NOTE:  These parameter need to be passed down to the enumerator."""

        self.kekule_smiles = kekule_smiles
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize
        self.seed = seed
        if self.seed > -1:
            np.random.seed(self.seed)

    def __call__(self, smiles: str) -> str:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: randomized SMILES representation.
        """
        molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
        if molecule is None:
            print(f'\nAugmentation skipped for invalid mol: {smiles}')
            return smiles
        if not self.sanitize:
            molecule.UpdatePropertyCache(strict=False)
        atom_indexes = list(range(molecule.GetNumAtoms()))
        if len(atom_indexes) == 0:  # RDkit error handling
            return smiles
        np.random.shuffle(atom_indexes)
        renumbered_molecule = Chem.RenumberAtoms(molecule, atom_indexes)
        if self.kekule_smiles:
            Chem.Kekulize(renumbered_molecule)

        return Chem.MolToSmiles(
            renumbered_molecule,
            canonical=False,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit,
        )
#代码中使用在线增强，可以按论文中使用离线增强
class ProteinSmileDataset(Dataset):
    def __init__(self,
                 affinity_filepath,
                 receptor_filepath,
                 Protein_model,
                 Protein_tokenizer,
                 Protein_padding,
                 ligand_filepath,
                 SMILES_model,
                 SMILES_tokenizer,
                 SMILES_padding,
                 SMILES_argument,
                 SMILES_Canonicalization,
                 device
                 ):
        self.affinity = pd.read_csv(affinity_filepath, sep=",", index_col=0)
        #过滤一些过长的ligand数据
        self.affinity=self.affinity[~self.affinity["ligand_name"].isin([120,131,134,137])]

        receptor = pd.read_csv(receptor_filepath, sep="\t", header=None, index_col=1)
        receptor[0] = [' '.join(list(x)) for x in receptor[0]]
        self.receptor=receptor[0].to_dict()

        ligand=pd.read_csv(ligand_filepath,index_col=0)
        self.ligand_SMILES = ligand["SMILES"].to_dict()
        self.ligand_AA= ligand["AA"].to_dict()

        #model
        self.Protein_model=Protein_model
        self.Protein_tokenizer=Protein_tokenizer
        self.SMILES_model=SMILES_model
        self.SMILES_tokenizer=SMILES_tokenizer
        #transform
        self.argument=Augment(sanitize=False)
        self.canonicalization=Canonicalization(sanitize=False)
        self.Protein_padding=Protein_padding
        self.SMILES_padding=SMILES_padding
        self.SMILES_argument=SMILES_argument
        self.SMILES_Canonicalization=SMILES_Canonicalization
        self.device = device


    def __len__(self):
        return self.affinity.shape[0]

    def __getitem__(self, index):
        selected_sample = self.affinity.iloc[index]
        affinity_tensor = torch.tensor(
            [selected_sample["label"]],
            dtype=torch.float
        )
        #根据id获取序列
        receptor_index=selected_sample["sequence_id"]
        receptor_seq=self.receptor[receptor_index]

        ligand_index=selected_sample["ligand_name"]
        ligand_SMILES_seq=self.ligand_SMILES[ligand_index]
        ligand_AA_seq=self.ligand_AA[ligand_index]

        if self.SMILES_argument:
            ligand_SMILES_seq=self.argument(ligand_SMILES_seq)
        if self.SMILES_Canonicalization:
            ligand_SMILES_seq=self.canonicalization(ligand_SMILES_seq)

        # bert开头添加 special tokens [CLS]
        ids = self.Protein_tokenizer.batch_encode_plus([receptor_seq],
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  max_length=self.Protein_padding+1)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            receptor_embedding = self.Protein_model(input_ids=input_ids, attention_mask=attention_mask)
        receptor_embedding = receptor_embedding.last_hidden_state
        receptor_embedding=receptor_embedding[0][1:]

        # bert开头添加 special tokens [CLS]
        ids = self.SMILES_tokenizer.batch_encode_plus([ligand_SMILES_seq],
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  max_length=self.SMILES_padding+1)
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
        with torch.no_grad():
            SMILES_embedding = self.SMILES_model(input_ids=input_ids, attention_mask=attention_mask)
        SMILES_embedding = SMILES_embedding.last_hidden_state
        SMILES_embedding=SMILES_embedding[0][1:]

        return receptor_seq,ligand_AA_seq,ligand_SMILES_seq,receptor_embedding,SMILES_embedding,affinity_tensor




#测试dataset的正确性
if __name__=='__main__':

    from transformers import BertModel, BertTokenizer
    from transformers import RobertaTokenizer, RobertaModel

    device = torch.device('cuda', 1)
    Protein_model_name = "Rostlab/prot_bert_bfd"
    SMILES_model_name = "DeepChem/ChemBERTa-77M-MLM"

    # AA model
    Protein_tokenizer = BertTokenizer.from_pretrained(Protein_model_name, do_lower_case=False, local_files_only=True)
    Protein_model = BertModel.from_pretrained(Protein_model_name, torch_dtype=torch.float16, local_files_only=True)
    Protein_model = Protein_model.to(device)

    # SMILES model
    SMILES_tokenizer = RobertaTokenizer.from_pretrained(SMILES_model_name, local_files_only=True)
    SMILES_model = RobertaModel.from_pretrained(SMILES_model_name, torch_dtype=torch.float16, local_files_only=True)
    SMILES_model = SMILES_model.to(device)

    dataset = ProteinSmileDataset(
        affinity_filepath='/home/xp/ATPnet/data/tcr_split/fold0/train+covid.csv',
        receptor_filepath='/home/xp/ATPnet/data/tcr_full.csv',
        Protein_model=Protein_model,
        Protein_tokenizer=Protein_tokenizer,
        Protein_padding=150,
        ligand_filepath="/home/xp/ATPnet/data/epitopes_merge.csv",
        SMILES_model=SMILES_model,
        SMILES_tokenizer=SMILES_tokenizer,
        SMILES_padding=350,
        SMILES_argument=True,
        SMILES_Canonicalization=False,
        device=device
    )
    next(iter(dataset))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    a, b, c, d, e ,f= next(iter(loader))


