# -- coding: utf-8 --
# @Time : 11/3/22 10:36 AM
# @Author : xpgege
# @File : test.py
# @Software: PyCharm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

import rdkit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar

from IPython.display import Image

#初始化画图参数
NON_ATOM_CHARACTERS = set(
    list(map(str, range(1, 10))) + list('%{}'.format(index) for index in range(10, 30)) + ['/','\\','(', ')', '#', '=', '.', ':']
)
CMAP = cm.Oranges
COLOR_NORMALIZERS = {
    'linear': colors.Normalize,
    'logarithmic': colors.LogNorm
}

ATOM_RADII = 1
PADDING_ATOM = '<PAD>'
COLOR_NORMALIZATION = 'linear'



def _get_index_and_colors(smi_att, smi_token, indexes, color_mapper_non_toxic):
    """
    Get index and RGB colors from a color map using a rule.
    The predicate acts on a tuple of (value, object).
    """

    # for atoms in not_tox_location
    indices = []
    colors = {}
    attention = {}
    for index, attention in enumerate(
            map(
                lambda t: t[0],
                filter(
                    lambda t: indexes(t),
                    zip(smi_att, smi_token)
                )
            )
    ):
        indices.append(index)
        colors[index] = color_mapper_non_toxic.to_rgba(attention)
    return colors


def draw_attention_map(epitope,molecule_smis,smiles_attentions):

    smiles_tokens = [x for x in molecule_smis]
    nontox_locations = [x for x in range(len(smiles_tokens))]
    smi_att = smiles_attentions
    normalize = COLOR_NORMALIZERS.get(
        COLOR_NORMALIZATION, colors.LogNorm
    )(
        vmin=(min(smi_att)+max(smi_att))/2,
        vmax=2*max(smi_att)
    )

    color_mapper_non_toxic = cm.ScalarMappable(
        norm=normalize, cmap=CMAP
    )

    smi_token = smiles_tokens
    mol = molecule_smis
    mol_ = Chem.MolFromSmiles(mol)
    nontox_loc = nontox_locations
    bond_cols = {}
    hit_ats_non = range(mol_.GetNumAtoms())
    atoms_colors = _get_index_and_colors(smi_att, smi_token, lambda t: t[1] not in NON_ATOM_CHARACTERS,
                                         color_mapper_non_toxic)

    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
    do = rdMolDraw2D.MolDrawOptions()
    do.bondLineWidth = 1
    d.SetDrawOptions(do)
    d.DrawMolecule(
        mol_,
        highlightAtoms=hit_ats_non,
        highlightAtomColors=atoms_colors,
        highlightBonds=None,
        highlightBondColors=None,
        highlightAtomRadii={
            index: 0.5
            for index in hit_ats_non
        }
    )
    d.FinishDrawing()
    d.WriteDrawingText('smile_attention.png',)
    # Image(filename='smile_attention.png', width=250)
    # svg = d.GetDrawingText()
    # with open('smile_attention.svg', 'w') as f:
    #     f.write(svg)
    # img_path='smile_attention.png'
    # from PIL import Image
    # img = Image.open(img_path)
    #
    # fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    # gs = fig.add_gridspec(1, 30)
    #
    # ax = fig.add_subplot(gs[0, 0:29])
    # ax.imshow(img)
    # ax.set_title(epitope)
    # ax.axis('off')
    #
    # ax2 = fig.add_subplot(gs[0, 29])
    # ax2.set_title('high', size=9, color='#71000E')
    # ax2.set_xlabel('low', size=9, color='#71000E')
    # cb2 = colorbar.ColorbarBase(ax2, cmap=CMAP,
    #                             ticks=[],
    #                             orientation='vertical')
    # cb2.set_label('Attention on atoms', size=9, color='#71000E')
    # plt.show()
    # plt.savefig(epitope+".png",dpi=400)

if __name__ == '__main__':
    # molecule_smis = "CCc1ccc(/C=C2/SC(=S)NC2=O)cc1"
    # smiles_attentions = [0.006696911528706551, 0.013492482714354992, 0.01517819706350565, 0.010640974156558514, 0.005452001933008432, 0.007403054740279913, 0.006010953802615404, 0.011763982474803925, 0.009754461236298084, 0.03150908648967743, 0.033318694680929184, 0.012895229272544384, 0.052271876484155655, 0.02710498683154583, 0.025007009506225586, 0.010064910165965557, 0.02399967424571514, 0.022340014576911926, 0.03257692977786064, 0.023365996778011322, 0.00835273414850235, 0.02631978690624237, 0.054274048656225204, 0.019341913983225822, 0.016009842976927757, 0.016858702525496483, 0.008694647811353207, 0.007494350429624319, 0.007298177573829889]
    # #nontox_locations=()
    # draw_attention_map("attention maps",molecule_smis,smiles_attentions)

    # molecule_smis = "CC(C)C[C@H](NC(=O)CNC(=O)CNC(=O)[C@H](CCCCN)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O"
    # smiles_attentions = [1]*len(molecule_smis)
    # # nontox_locations=()
    # draw_attention_map("FLKEKGGL", molecule_smis, smiles_attentions)

    smiles_attentions=[0.0068210517,
     0.008760168,
     0.00028429684,
     0.0019218221,
     0.0020139273,
     0.00023296635,
     0.00019593856,
     0.0006657781,
     0.00030404315,
     0.0024284758,
     0.0006275648,
     0.000110318666,
     0.00023316655,
     0.0022942591,
     0.004685404,
     0.0025458054,
     0.006909868,
     0.0038651526,
     0.0021327264,
     0.0036544173,
     0.0009429774,
     0.0006798329,
     0.0009943419,
     0.005767605,
     0.00037373026,
     0.000775995,
     0.00047680447,
     0.00015384972,
     0.00061929546,
     0.00043793253,
     0.0020577335,
     0.007228074,
     0.0010267079,
     0.0021513219,
     0.0015908319,
     0.0003877325,
     0.0009450579,
     0.0024030355,
     0.0017495081,
     0.0043446026,
     0.0033139922,
     0.000709042,
     0.0053850287,
     0.008426075,
     0.0003170622,
     0.0005176503,
     0.0005349114,
     0.00018424001,
     0.00010498504,
     9.345604e-05,
     0.00014486634,
     0.00013191871,
     8.591018e-05,
     0.002422088,
     0.00028912202,
     0.004832624,
     0.00617968,
     0.00050891587,
     0.00024531072,
     0.00041255474,
     0.00090263906,
     0.0012468533,
     0.0008850211,
     0.00035021434,
     0.0014084864,
     0.0013238183,
     0.0017897562,
     0.0058583426,
     0.003204999,
     0.0016413296,
     0.0021637727,
     0.0005058789,
     0.0009351627,
     0.0046449336,
     0.00047243506,
     0.0013418985,
     0.00035429717,
     0.00022987416,
     0.00017987326,
     0.0004816037,
     4.1893698e-05,
     9.6279626e-05,
     0.0033512346,
     0.0001732112,
     0.00056227087,
     0.0005457413,
     9.1625756e-05,
     6.285345e-05,
     0.0074709654,
     0.0036746885,
     0.00026820955,
     0.00010616946,
     0.07397222,
     0.006971819,
     0.0013717235,
     0.0019318953,
     0.00026995267,
     0.00049536367,
     0.002120923,
     0.010236511,
     0.0009803277,
     0.003411975,
     0.00013897837,
     0.004105957,
     0.010624728,
     0.000109190485,
     0.0059118113,
     0.00029852474,
     0.00063212716,
     0.005588778,
     0.0033180604,
     0.0031984819,
     0.0037309823,
     0.007529299,
     0.00080702774,
     0.0005161425,
     0.00094133796,
     0.00011029718,
     0.00035363022,
     0.0003529514,
     0.0008487628,
     0.0060287723,
     0.0025517535,
     0.0020263745,
     0.03308038,
     0.0019343641,
     0.014775516,
     0.04016541,
     0.00041653842,
     0.0019170649,
     0.000119482516,
     0.00027073408,
     0.0001021209,
     0.00019459445,
     4.3786516e-05,
     7.154718e-05,
     0.0019402157,
     9.9200224e-05,
     0.00024696699,
     0.0004410918,
     8.38195e-05,
     0.00012420616,
     0.020131396,
     0.0032489586,
     0.0003341127,
     0.00014571784,
     0.0010930026,
     0.008305944,
     0.003179701,
     0.008731548,
     0.003994304,
     0.011848086,
     0.0009927765,
     0.012729551,
     0.0051206504,
     0.00017992913,
     9.3632756e-05,
     0.0010029251,
     0.0004789597,
     0.0002326525,
     0.000691677,
     0.0021167458,
     0.009079334,
     0.00094471744,
     0.00011242321,
     0.00046056416,
     0.008330678,
     0.00017947861,
     0.00062088587,
     0.0007952733,
     0.00015012905,
     0.00071311125,
     0.026466122,
     0.0004087324]
    import math
    smiles_attentions=[math.log(x,2) for x in smiles_attentions]
    ligand_SMILES_seq = "CC(C)C[C@H](N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N1CCC[C@H]1C(=O)N[C@H](C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](C(=O)O)C(C)C)C(C)C"
    # ligand_SMILES_seq=[x for x in ligand_SMILES_seq]
    # import  pandas as pd
    # df=pd.DataFrame([ligand_SMILES_seq,smiles_attentions]).T
    # df[df[0]=="N"]
    draw_attention_map("attention maps", ligand_SMILES_seq, smiles_attentions)