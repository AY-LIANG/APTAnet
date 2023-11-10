# -- coding: utf-8 --
# @Time : 12/16/22 3:12 PM
# @Author : XXXX
# @File : test.py.py
# @Software: PyCharm
import torch
import matplotlib.pyplot as plt
if __name__ == '__main__':
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.01,
                                  weight_decay=0.01)
    scheduler = torch.optim. \
        lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,total_steps=1600,three_phase=False)
    lrs = []
    for i in range(10):
        for i in range(160):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

    plt.plot(lrs)
    plt.xlabel="epoch"
    plt.ylabel = "learning rate"
    plt.show()

