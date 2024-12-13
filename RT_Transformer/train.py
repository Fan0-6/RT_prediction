import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from model import MyNet, Trainer, Tester
from load_data import SMRTDataset, SMRTDatasetRetained, traindataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os
import torchmetrics
import pandas as pd

warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 30)
        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for i, data in enumerate(tqdm(data_loader)):
            data.to(self.device)
            y_hat = self.model(data)
            loss = criterion(y_hat, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return 0
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    batch_size = 64
    num_works = 8
    lr = 0.00001
    epochs = 500
    test_batch = 2048

    torch.manual_seed(1234)
    set_seed(1234)
    dataset = traindataset('./train/train_data')
    print(f'len of dataset:{len(dataset)}')
    # dataset = dataset.process()[0]
    print(len(dataset))
    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_len_2 = int(train_len * 0.9)
    dev_len = train_len - train_len_2
    train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    dev_loader = DataLoader(dev_data_set, batch_size=test_batch, shuffle=True,
                            num_workers=num_works, pin_memory=True,
                            prefetch_factor=8, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                             num_workers=num_works, pin_memory=True,
                             prefetch_factor=8, persistent_workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use\r', device)
    print('-' * 100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(train_dataset))
    print('# of test data samples:', len(test_dataset))
    print('-' * 100)
    print('Creating a model.')
    torch.manual_seed(1234)
    model = MyNet(emb_dim=512, feat_dim=512)
    # model = torch.load('./model/best_model2.pkl', map_location='cuda:0')
    # model = torch.load('best_model_epoch102_loss_32.12735703089935.pkl',map_location='cuda:0')
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    trainer = Trainer(model, lr, device)
    tester = Tester(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)
    torch.manual_seed(1234)

    model.to(device=device)
    mae_test_best = float('inf')
    no_improve_epochs = 0         # Counter for epochs with no improvement
    patience = 10                 # Number of epochs to wait before stopping

    # with open('./results/try_4.txt', 'a') as f:
        
    #     for epoch in range(epochs):
    #         print(trainer.optimizer.param_groups[0]['lr'])
    #         model.train()

    #         loss_training = trainer.train(train_loader)

    #         model.eval()
    #         # torch.cuda.empty_cache()
    #         mae_train, medAE_train, mre_train, rmse_train, r2_train, y_true, y_pred = tester.test_regressor(train_loader)
    #         mae_dev, medAE_dev, mre_dev, rmse_dev, r2_dev, y_true, y_pred = tester.test_regressor(dev_loader)
    #         mae_train_final = mae_train / train_len_2
    #         mae_dev_final = mae_dev / dev_len
    #         # print train mae, test mae, madAE, mre, rmse, r2
    #         print(f'epoch:{epoch}\ttrain_MAE:{mae_train_final}\ttest_MAE:{mae_dev_final}\ttest_MadAE:{medAE_dev}\ttest_MRE:{mre_dev}\ttest_RMSE:{rmse_dev}\ttest_R^2:{r2_dev}')
    #         # print(f'epoch:{epoch}\ttrain_loss:{mae_train_final}\ttest_loss:{mae_dev_final}\ttest_R^2:{r2_dev}')
    #         f.write(f'epoch:{epoch}\ttrain_MAE:{mae_train_final}\ttest_MAE:{mae_dev_final}\ttest_MadAE:{medAE_dev}\ttest_MRE:{mre_dev}\ttest_RMSE:{rmse_dev}\ttest_R^2:{r2_dev}\n')
    #         # f.write(f'epoch:{epoch}\ttrain_loss:{mae_train_final}\ttest_loss:{mae_dev_final}\tttest_R^2:{r2_dev}\n')
    #         f.flush()

    #         # if mae_dev < mae_test_best:
    #         #     # torch.save(model, f'./model/best_model_epoch{epoch}_loss_{mae_dev}.pkl')
    #         #     torch.save(model, f'./model/best_model3.pkl')
    #         #     mae_test_best = mae_dev
    #         if mae_dev < mae_test_best:
    #             # Improvement detected
    #             torch.save(model, f'./model/best_model4.pkl')
    #             mae_test_best = mae_dev
    #             no_improve_epochs = 0  # Reset counter
    #         else:
    #             # No improvement
    #             no_improve_epochs += 1
    #             if no_improve_epochs >= patience:
    #                 print(f'Early stopping: no improvement in validation loss for {patience} epochs.')
    #                 break  # Exit the training loop

    # model = torch.load('./model/best_model4.pkl', map_location=torch.device('cpu'))
    # tester.test_regressor(dev_loader)
    # print(tester.test_regressor(test_loader))

    # Load the best model
    model = torch.load('./model/best_model4.pkl', map_location=torch.device('cpu'))
    tester = Tester(model, device='cpu')  # Ensure the model is on CPU

    # Get metrics and predictions for test set
    mae, medAE, mre, rmse, r2, y_true, y_pred = tester.test_regressor(test_loader)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({'rt_pred': y_pred, 'rt': y_true})
    df.to_csv('./results/rt_true_pred_test.csv', index=False)

    print('Test MAE:', mae / dev_len)
    print('Test R^2:', r2)
    print('Test MadAE:', medAE)
    print('Test MRE:', mre)
    print('Test RMSE:', rmse)

    # Get metrics and predictions for training set
    mae, medAE, mre, rmse, r2, y_true, y_pred = tester.test_regressor(train_loader)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({'rt_pred': y_pred, 'rt': y_true})
    df.to_csv('./results/rt_true_pred_train.csv', index=False)

    print('Train MAE:', mae / train_len_2)
    print('Train R^2:', r2)

    # Get metrics and predictions for dev set
    mae, medAE, mre, rmse, r2, y_true, y_pred = tester.test_regressor(dev_loader)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({'rt_pred': y_pred, 'rt': y_true})
    df.to_csv('./results/rt_true_pred_dev.csv', index=False)

    print('Dev MAE:', mae / dev_len)
    print('Dev R^2:', r2)