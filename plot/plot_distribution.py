# -- coding: utf-8 --
# @Time : 12/13/22 11:05 AM
# @Author : XXXX
# @File : plot.py
# @Software: PyCharm
import pandas as pd
if __name__ == '__main__':
    # def SMILES_arguments(frequency=10):
    #     ligand_SMILES = pd.read_csv("../data/epitopes.smi", sep="\t", header=None, index_col=1)
    #     ligand_AA = pd.read_csv("../data/epitopes.csv", sep="\t", header=None, index_col=1)
    #     ligand = pd.concat([ligand_AA, ligand_SMILES], axis=1)
    #     ligand.columns = ["AA", "SMILES"]
    #     seq_list = []
    #     subid_list = []
    #     ID=[]
    #     for i in ligand.index:
    #         argument=Augment(sanitize=False,seed=i)
    #         cananical_seq=ligand["SMILES"][i]
    #         #print(cananical_seq)
    #         for j in range(0,10):
    #             ID.append(i)
    #             if j==0:
    #                 seq_list.append(cananical_seq)
    #                 subid_list.append(str(i)+"_0")
    #             else:
    #                 seq_list.append(argument(cananical_seq))
    #                 subid_list.append(str(i)+"_"+str(j))
    #     argument_ligand=pd.DataFrame([ID,subid_list,seq_list]).transpose()
    #     argument_ligand.columns=["index","subindex","argument_SMILES"]
    #     ligand=ligand.reset_index()
    #     result=pd.merge(argument_ligand,ligand,left_on="index",right_on=1)
    #     result=result[["index","subindex","SMILES","argument_SMILES","AA"]]
    #     result.to_csv("../data/"+str(frequency)+"_epitopes.csv")
    #
    # for t in [5,10,20]:
    #     SMILES_arguments(t)
    # -*- coding: utf-8 -*-
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 过滤长度分布
    ligand_SMILES = pd.read_csv("../data/epitopes.smi", sep="\t", header=None, index_col=1)
    ligand_AA = pd.read_csv("../data/epitopes.csv", sep="\t", header=None, index_col=1)
    receptor = pd.read_csv('../data/tcr_full.csv', sep="\t", header=None, index_col=1)
    ligand = pd.concat([ligand_AA, ligand_SMILES], axis=1)
    ligand.columns = ["AA", "SMILES"]
    ligand["length_SMILES"] = ligand["SMILES"].apply(len)
    ligand["length_AA"] = ligand["AA"].apply(len)
    ligand=ligand[(ligand["length_AA"]<15)]
    # ligand.to_csv("../data/epitopes_merge.csv")

    receptor["length"] = receptor[0].apply(len)

    import matplotlib

    sns.set(font_scale=1.3)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
    gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})

    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    # matplotlib.rcParams.update({'font.size': 15})  # 改变所有字体大小，改变其他性质类似


    # 画SMILES
    plt.subplot(2, 2, 1)
    sns.distplot(ligand["length_SMILES"], bins=20, kde=True, color='r', label="SMILES")

    axes[0,0].set_xlabel("SMILES length")
    import matplotlib.ticker as ticker

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    from matplotlib.pyplot import MultipleLocator

    y_major_locator = MultipleLocator(0.005)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)


    # 画AA
    # plt.subplot(3, 1, 2)
    # sns.distplot(ligand["length_AA"], bins=10, kde=True, color='g', label="AA")
    # plt.xlabel("Amino acid length")
    # import matplotlib.ticker as ticker
    #
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # from matplotlib.pyplot import MultipleLocator
    #
    # y_major_locator = MultipleLocator(0.15)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)

    # 画AA
    plt.subplot(2, 2, 2)
    sns.distplot(receptor["length"], bins=20, kde=True, color='b', label="Protein")
    plt.xlabel("TCR length")
    import matplotlib.ticker as ticker

    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    from matplotlib.pyplot import MultipleLocator

    y_major_locator = MultipleLocator(0.03)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)

    # plt.show()

    affnity=pd.read_csv("../data/VDJdb/SearchTable-2023-03-11 08_10_40.116.tsv",sep="\t")
    affnity=affnity[affnity["Gene"]=="TRB"]
    affnity=affnity[affnity["MHC class"]=="MHCI"]

    affnity=affnity[["CDR3","Epitope"]]
    affnity["CDR3_length"]=affnity["CDR3"].apply(len)
    affnity["peptide_length"]=affnity["Epitope"].apply(len)
    affnity["CDR3_length"].max()
    affnity["CDR3_length"].min()
    affnity["peptide_length"].max()
    affnity["peptide_length"].min()
    affnity=affnity[(affnity["CDR3_length"]>10) & (affnity["CDR3_length"]<=20)]
    affnity=affnity[(affnity["peptide_length"]>=8) & (affnity["peptide_length"]<15)]
    counts=affnity.groupby("Epitope")["CDR3"].count().sort_values(ascending=False)

    gs = axes[1, 0].get_gridspec()
    axes[1, 0].remove()
    axes[1, 1].remove()
    axbig = fig.add_subplot(gs[1, :])
    y=counts[0:30].to_list()
    x=counts[0:30].index.to_list()
    # counts[0]=6000
    axbig.bar(x,y,color='g',alpha=0.5)

    axbig.set_xticklabels(x,rotation=90,fontsize=13)
    # axbig.set_xticks([])
    axbig.set_ylim([0,6000])
    axbig.set_ylabel("Paired TCR Counts")
    axbig.set_xlabel("Peptides")
    # plt.tight_layout()
    # plt.show()
    plt.savefig("./peptide_distribution.png",dpi=300)
