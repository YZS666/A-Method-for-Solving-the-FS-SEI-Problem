import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from model import Classifier
from get_fewshot_LoRa_IQ_dataset import *
import random
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(encoder,
          classifier,
          dataloader,
          optim_encoder,
          optim_classifier,
          scheduler_encoder,
          scheduler_classifier,
          epoch,
          device_num,
          writer
          ):
    encoder.train()
    classifier.train()
    device = torch.device("cuda:" + str(device_num))
    loss_ce = 0
    correct = 0
    for data_label in dataloader:
        data, target = data_label
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optim_encoder.zero_grad()
        optim_classifier.zero_grad()

        z = encoder(data)
        logits = F.log_softmax(classifier(z))
        target = np.squeeze(target, axis=1)
        loss_ce_batch = F.nll_loss(logits, target)
        loss_ce_batch.backward()
        optim_encoder.step()
        optim_classifier.step()
        scheduler_encoder.step()
        scheduler_classifier.step()

        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss_ce += loss_ce_batch.item()


    loss_ce /= len(dataloader)

    fmt = 'Train Epoch: {} \tCE_Loss, {:.8f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            epoch,
            loss_ce,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )

    writer.add_scalar('CE_Loss/train', loss_ce, epoch)

def validation(encoder, classifier, test_dataloader, epoch, device_num, writer):
    encoder.eval()
    classifier.eval()
    loss_ce = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            z = encoder(data)
            logits = F.log_softmax(classifier(z))
            target = np.squeeze(target, axis=1)
            loss_ce += F.nll_loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        loss_ce /= len(test_dataloader.dataset)
        fmt = '\nValidation set: CE_loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
        print(
            fmt.format(
                loss_ce,
                correct,
                len(test_dataloader.dataset),
                100.0 * correct / len(test_dataloader.dataset),
            )
        )

        writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
        writer.add_scalar('Classifier_Loss/validation', loss_ce, epoch)

    return loss_ce

def Test(encoder, classifier, test_dataloader):
    encoder.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    device = torch.device("cuda:0")
    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            target = target.squeeze()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)

            z  = encoder(data)
            logits = F.log_softmax(classifier(z), dim=1)
            test_loss += loss(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target)-1] = pred.tolist()
            target_real[len(target_real):len(target)-1] = target.tolist()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
    test_acc = 100.0 * correct / len(test_dataloader.dataset)
    return str(test_acc)+'%'


def train_and_validation(encoder,
                         classifier,
                         dataloader,
                         val_dataloader,
                         optim_encoder,
                         optim_classifier,
                         scheduler_encoder,
                         scheduler_classifier,
                         epochs,
                         encoder_save_path,
                         classifier_save_path,
                         device_num,
                         writer):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(encoder,
              classifier,
              dataloader,
              optim_encoder,
              optim_classifier,
              scheduler_encoder,
              scheduler_classifier,
              epoch,
              device_num,
              writer)
        validation_loss = validation(encoder, classifier, val_dataloader, epoch, device_num, writer)
        if epoch == 1:
            current_min_test_loss = validation_loss
        if validation_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, validation_loss))
            current_min_test_loss = validation_loss
            torch.save(encoder, encoder_save_path)
            torch.save(classifier, classifier_save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

class Config:
    def __init__(
            self,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            epochs: int = 200,
            lr_encoder: float = 0.001,
            lr_classifier: float = 0.001,
            n_classes: int = 30,
            encoder_save_path: str = 'model_weight/MAE_encoder_IQ.pth',
            classifier_save_path: str = 'model_weight/MAE_classifier_IQ.pth',
            encoder_load_path: str = 'model_weight/pretrain_MAE_encoder_IQ.pth',
            device_num: int = 0,
            iteration: int = 100,
            k_shot: int = 10,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr_encoder = lr_encoder
        self.lr_classifier = lr_classifier
        self.n_classes = n_classes
        self.encoder_save_path = encoder_save_path
        self.classifier_save_path = classifier_save_path
        self.encoder_load_path = encoder_load_path
        self.device_num = device_num
        self.iteration = iteration
        self.k_shot = k_shot

def main(RANDOM_SEED):
    conf = Config()
    device = torch.device("cuda:" + str(conf.device_num))
    writer = SummaryWriter("logs")

    set_seed(RANDOM_SEED)
    X_train, X_val, Y_train, Y_val = get_num_class_Targettrainfinetunedata(conf.n_classes, conf.k_shot)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.train_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.train_batch_size, shuffle=True)

    encoder = torch.load(conf.encoder_load_path)
    classifier = Classifier()
    if torch.cuda.is_available():
        encoder = encoder.to(device)
        classifier = classifier.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=conf.lr_encoder)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=conf.lr_classifier)

    scheduler_encoder = CosineAnnealingLR(optim_encoder, T_max=20)
    scheduler_classifier = CosineAnnealingLR(optim_classifier, T_max=20)

    train_and_validation(encoder,
                         classifier,
                         train_dataloader,
                         val_dataloader,
                         optim_encoder,
                         optim_classifier,
                         scheduler_encoder,
                         scheduler_classifier,
                         conf.epochs,
                         conf.encoder_save_path,
                         conf.classifier_save_path,
                         conf.device_num,
                         writer)


def Test_main():
    num = [0, 30]
    X_test, Y_test = get_num_class_Targettestdata(num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    encoder = torch.load(conf.encoder_save_path)
    classifier = torch.load(conf.classifier_save_path)
    test_acc = Test(encoder, classifier, test_dataloader)
    return test_acc


if __name__ == '__main__':
    conf = Config()
    test_acc_all = []

    for i in range(conf.iteration):
        print(f"iteration: {i}-------------------------------------------")
        main(i)
        test_acc = Test_main()
        test_acc_all.append(test_acc)
        print(f"iteration={i},test_acc={test_acc}\n")
    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"test_result/AMAE_{conf.n_classes}classes_{conf.k_shot}shot_{conf.iteration}iteration.xlsx")



