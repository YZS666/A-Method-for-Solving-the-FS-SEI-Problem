import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from model import Encoder, Decoder, Classifier
from get_fewshot_LoRa_IQ_dataset import *
from sklearn import metrics
import random
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output)-1] = output.tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def get_accuracy_score(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype = np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    print('ACC = ', acc)
    return acc



def pre_train(encoder,
              decoder,
              dataloader,
              mask_ratio,
              optim_encoder,
              optim_decoder,
              epoch,
              device_num,
              writer
              ):
    encoder.train()
    decoder.train()
    device = torch.device("cuda:" + str(device_num))
    loss_mse = 0
    for data_label in dataloader:
        data, target = data_label
        if torch.cuda.is_available():
            data = data.to(device)

        optim_encoder.zero_grad()
        optim_decoder.zero_grad()

        bbx1, bbx2, maskdata = MaskData(data, mask_ratio)
        z = encoder(maskdata)
        data_r = decoder(z)
        mask = torch.zeros((data.shape[0],data.shape[1],data.shape[2])).cuda()
        mask[:, :, bbx1: bbx2] = torch.ones((data.size()[1],bbx2-bbx1)).cuda()
        data = data.mul(mask)
        data_r = data_r.mul(mask)
        loss_mse_batch = F.mse_loss(data_r, data)
        # loss_mse_batch = F.mse_loss(data_r, data, reduction='sum') / (bbx2-bbx1)
        loss_mse_batch.backward()
        optim_encoder.step()
        optim_decoder.step()

        loss_mse += loss_mse_batch.item()

    loss_mse /= len(dataloader)

    print('Train Epoch: {} \tMSE_Loss, {:.8f}\n'.format(
            epoch,
            loss_mse,
    )
    )

    writer.add_scalar('MSE_Loss/train', loss_mse, epoch)
    return loss_mse

def validation(encoder, decoder, test_dataloader, epoch, device_num, writer):
    encoder.eval()
    decoder.eval()
    sc = 0
    loss_mse = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            if torch.cuda.is_available():
                data = data.to(device)
            z = encoder(data)
            data_r = decoder(z)
            loss_mse += F.mse_loss(data_r, data).item()

    loss_mse /= len(test_dataloader)
    fmt = '\nValidation set: MSE loss: {:.8f}\n'
    print(
        fmt.format(
            loss_mse,
        )
    )

    writer.add_scalar('MSE_Loss/validation', loss_mse, epoch)

    X_test_embedding_feature_map, real_target = obtain_embedding_feature_map(encoder, test_dataloader)
    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    km = KMeans(n_clusters=30, n_init=30)
    km.fit(eval_tsne_embeds)
    cluster_target = km.predict(eval_tsne_embeds)
    # ac = get_accuracy_score(real_target, cluster_target)
    sc = metrics.silhouette_score(X_test_embedding_feature_map, cluster_target)

    fmt = '\nValidation set: SC: {:.8f}\n'
    print(
        fmt.format(
        sc,
        )
    )

    writer.add_scalar('SC/validation', sc, epoch)

    return sc, loss_mse

def train_and_validation(encoder,
                         decoder,
                         dataloader,
                         mask_ratio,
                         val_dataloader,
                         optim_encoder,
                         optim_decoder,
                         epochs,
                         encoder_save_path,
                         decoder_save_path,
                         device_num,
                         writer):
    current_sc_mse = 0
    gamma = 1
    loss_sc_mse_all = []
    loss_mse_all = []
    loss_sc_all = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            current_sc, current_mse = validation(encoder, decoder, val_dataloader, epoch, device_num, writer)
            current_sc_mse = current_sc - current_mse
        train_mse_loss = pre_train(encoder,
              decoder,
              dataloader,
              mask_ratio,
              optim_encoder,
              optim_decoder,
              epoch,
              device_num,
              writer)
        sc, mse = validation(encoder, decoder, val_dataloader, epoch, device_num, writer)

        sc_mse = sc - 0.3 * mse
        loss_sc_mse_all.append(sc_mse)
        loss_mse_all.append(mse)
        loss_sc_all.append(sc)

        if sc_mse > current_sc_mse:
            print("The training SC-MSE is improved from {} to {}, new model weight is saved.".format(
                current_sc_mse, sc_mse))
            current_sc_mse = sc_mse
            torch.save(encoder, encoder_save_path)
            torch.save(decoder, decoder_save_path)

        else:
            print("The training SC-MSE is not improved.")
        print("------------------------------------------------")

        writer.add_scalar('UnsupervisedLoss/train', sc_mse, epoch)

         # torch.save(encoder, encoder_save_path)
    df = pd.DataFrame(loss_sc_mse_all)
    df.to_excel(f"test_result/SimMIM_S_SC_MSE.xlsx")
    df = pd.DataFrame(loss_mse_all)
    df.to_excel(f"test_result/SimMIM_S_MSE.xlsx")
    df = pd.DataFrame(loss_sc_mse_all)
    df.to_excel(f"test_result/SimMIM_S_SC.xlsx")


class Config:
    def __init__(
            self,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            epochs: int = 300,
            mask_ratio: float = 0.3,
            lr_encoder: float = 0.001,
            lr_decoder: float = 0.001,
            n_classes: int = 30,
            encoder_save_path: str = 'model_weight/pretrain_MAE_encoder_IQ.pth',
            decoder_save_path: str = 'model_weight/pretrain_MAE_decoder_IQ.pth',
            device_num: int = 0,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.mask_ratio = mask_ratio
        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.decoder_save_path = decoder_save_path
        self.device_num = device_num

def main():
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs")
    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train, X_val, Y_train, Y_val = get_num_class_Sourcetraindata(conf.n_classes)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True)

    encoder = Encoder()
    decoder = Decoder()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        decoder = decoder.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=conf.lr_decoder)

    train_and_validation(encoder,
                         decoder,
                         train_dataloader,
                         conf.mask_ratio,
                         val_dataloader,
                         optim_encoder,
                         optim_decoder,
                         conf.epochs,
                         conf.encoder_save_path,
                         conf.decoder_save_path,
                         conf.device_num,
                         writer)

if __name__ == '__main__':
    main()




