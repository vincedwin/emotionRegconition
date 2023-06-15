import torch
import config
import cv2 as cv

def save_checkpoint(model, optimizer, filename=config.CHECKPOINT):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def showText(frame, status, percent, x, y, h, w):
    x1, y1, w1, h1 = 660, 980, 170, 75

    cv.putText(frame, (f'{status} {percent}%'), (x1 + int(w1/10), y1+ int(h1/10)), cv.FONT_HERSHEY_SIMPLEX, 4, (240, 240, 240), 4) #T.L. display

    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
    cv.rectangle(frame, (x-5, y-50), (x+5 + w, y), (154,205,50), -1)
    cv.putText(frame, (f'{status}'), (x, y-12), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


def most_frequent(List):
    counter, num = 0, List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

def how_frequent(Num, List):
    percent = (List.count(Num)/ len(List)) * 100
    return int(percent)

def mask_percent(previous, current):
    percent = previous + int(current - previous / 4)
    return int(percent)