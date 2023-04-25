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
        assert trainloader is not None, print("train_dataset is none")
        model.train()        
        #ave_accuracy = AverageMeter()
        #scaler = torch.cuda.amp.GradScaler()
        for idx, (Image, BBox, Label) in enumerate(tqdm(trainloader)):
            #model input data
            Input = Image.to(device, non_blocking=True)
            label = Label.to(device, non_blocking=True)
            bbox = BBox.to(device, non_blocking=True)
            predict_cls, predict_bboxes = model(Input)
            loss_loc, loss_cls, loss_iou = criterion(predict_bboxes, predict_cls, bbox, label)
            loss = 0.1*loss_loc+loss_cls + loss_iou
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #accuracy = IoUFindBBox(predict_bboxes.detach(), bbox)
            #ave_accuracy.update(accuracy)
            if idx % write_iter_num == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(trainloader)} '
                        f'Loss : {loss :.4f} ')
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(trainloader)} '
                        f'Loss : {loss :.4f} ', file=file)
        assert validloader is not None, print("valid_dataset is none")
        model.eval()
        whole_loss = 0
        with torch.no_grad():
            for idx, (Image, BBox, Label) in enumerate(tqdm(validloader)):
                #model input data
                Input = Image.to(device, non_blocking=True)
                label = Label.to(device, non_blocking=True)
                bbox = BBox.to(device, non_blocking=True)
                predict_cls, predict_bboxes = model(Input)
                loss_loc, loss_cls, loss_iou = criterion(predict_cls, predict_bboxes, bbox, label)
                loss = loss_loc + loss_cls + loss_iou
                whole_loss += loss
                #accuracy = IoUAccuracy(predict_cls, target_bboxes)
                #ave_accuracy.update(accuracy)
                if idx % write_iter_num == 0:
                    tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(validloader)} '
                            f'Loss : {loss :.4f} ')
                    tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(validloader)} '
                            f'Loss : {loss :.4f} ', file=file)
            #tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} ', file=file)
        accuracy = whole_loss/len(validloader)
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