import torch
from utils.utils import AverageMeter
from tqdm import tqdm
from utils.utils import IoUFindBBox, save_checkpoint
import os

def train_epoch(model=None, write_iter_num=5, trainloader=None, validloader=None, optimizer=None, scheduler=None, device=None, 
                criterion=None, start_epoch=0, end_epoch=None, log_path="./log", model_path="./weight", best_loss = 0):
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(start_epoch, end_epoch):
        is_best = False
        file = open(os.path.join(log_path, f'{epoch}_log.txt'), 'a')
        train(model=model, write_iter_num=write_iter_num, train_dataset=trainloader, optimizer=optimizer, 
                device=device, criterion=criterion, epoch=epoch, file=file)
        accuracy = valid(model=model, write_iter_num=write_iter_num, valid_dataset=validloader, criterion=criterion, 
                               device=device, epoch=epoch, file=file)
        scheduler.step()
        is_best = accuracy < best_loss
        best_loss = max(best_loss, accuracy)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_loss,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best=is_best, path=model_path)
        file.close()

def train(model=None, write_iter_num=5, train_dataset=None, optimizer=None, device=None, criterion=None, epoch=None, file=None):
    #scaler = torch.cuda.amp.GradScaler()
    assert train_dataset is not None, print("train_dataset is none")
    model.train()        
    #ave_accuracy = AverageMeter()
    #scaler = torch.cuda.amp.GradScaler()
    for idx, (Image, BBox, Label) in enumerate(tqdm(train_dataset)):
        #model input data
        Input = Image.to(device, non_blocking=True)
        label = Label.to(device, non_blocking=True)
        bbox = BBox.to(device, non_blocking=True)
        predict_cls, predict_bboxes = model(Input)
        loss_loc, loss_cls = criterion(predict_bboxes, predict_cls, bbox, label)
        loss = loss_loc+loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #accuracy = IoUFindBBox(predict_bboxes.detach(), bbox)
        #ave_accuracy.update(accuracy)
        if idx % write_iter_num == 0:
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                       f'Loss : {loss :.4f} ')
            tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(train_dataset)} '
                       f'Loss : {loss :.4f} ', file=file)

def valid(model=None, write_iter_num=5, valid_dataset=None, criterion=torch.nn.CrossEntropyLoss(), device=None, epoch=None, file=None):
    #ave_accuracy = AverageMeter()
    assert valid_dataset is not None, print("valid_dataset is none")
    model.eval()
    whole_loss = 0
    with torch.no_grad():
        for idx, (Image, BBox, Label) in enumerate(tqdm(valid_dataset)):
            #model input data
            Input = Image.to(device, non_blocking=True)
            label = Label.to(device, non_blocking=True)
            bbox = BBox.to(device, non_blocking=True)
            predict_cls, predict_bboxes = model(Input)
            loss_loc, loss_cls= criterion(predict_cls, predict_bboxes, bbox, label)
            loss = loss_loc + loss_cls
            whole_loss += loss
            #accuracy = IoUAccuracy(predict_cls, target_bboxes)
            #ave_accuracy.update(accuracy)
            if idx % write_iter_num == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(valid_dataset)} '
                        f'Loss : {loss :.4f} ')
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(valid_dataset)} '
                        f'Loss : {loss :.4f} ', file=file)
        #tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} ', file=file)
    return whole_loss/len(valid_dataset)