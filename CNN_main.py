import timeit
from collections import OrderedDict
import torch
import numpy as np
import random
import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CNN_submission import Net, PretrainedNet, load_dataset, get_config_dict
from torchvision import transforms,datasets
from torchvision.models import ResNet18_Weights
weight = ResNet18_Weights.DEFAULT
preprocess = weight.transforms()


torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    """
    Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

def compute_score(acc, acc_thresh):
    """
    Grades your assignment.
    """
    min_thres, max_thres = acc_thresh
    if acc <= min_thres:
        base_score = 0.0
    elif acc >= max_thres:
        base_score = 100.0
    else:
        base_score = float(acc - min_thres) / (max_thres - min_thres) \
                     * 100
    return base_score

def should_save_model(epoch, num_epochs, val_loss, best_val_loss, val_accuracy, best_val_accuracy, criteria):

    if criteria not in ['last','loss','accuracy']:
        raise ValueError("Invalid save criteria. Choose either 'loss', 'accuracy', or 'last'")
    
    if criteria == 'last' and epoch == num_epochs - 1:
        return True
    elif criteria == 'loss' and val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
        return True
    elif criteria == 'accuracy' and val_accuracy > best_val_accuracy:
        print(f"Validation accuracy increased ({best_val_accuracy:.4f} --> {val_accuracy:.4f}). Saving model...")
        return True

    return False

def save_checkpoint(model, val_loss, val_accuracy, filename='best_model.pth'):
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, filename)

    print(f"Model saved to {filename}")

def test(
        model,
        device,
        test_transform

):
    
    if test_transform:
        print("Using custom transforms for test set..")
        transformations = test_transform
    else:
        print("Using default transforms for test set..")
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transformations)

    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    model.eval()
    num_correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    acc = float(num_correct) / total
    return acc

def train(
        config: dict,
        model,
        train_dataset,
        valid_dataset,
        device,
        pretrain,
        log_interval: int = 10,

):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    objective = nn.CrossEntropyLoss()

    best_val_loss = float('inf') 
    best_val_accuracy = 0

    for epoch in range(config["num_epochs"]):
        model.train()
        print(f"Epoch: {epoch + 1}")
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(data)
            loss = objective(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f"\tbatch: {batch_idx} train Loss: {loss.item():.4f}")

        valid_acc, valid_loss = validate(valid_loader, model, device)
        print(f"\tvalidation Loss: {valid_loss:.4f} accuracy: {valid_acc:.4f}")

        if pretrain:
            if should_save_model(epoch, config["num_epochs"], valid_loss, best_val_loss, valid_acc, best_val_accuracy, config["save_criteria"]):
                if config["save_criteria"] == 'loss':
                    best_val_loss = valid_loss
                elif config["save_criteria"] == 'accuracy':
                    best_val_accuracy = valid_acc

                # Save the model
                save_checkpoint(model, valid_loss, valid_acc)


def validate(
        loader,
        model,
        device,
):
    model.eval()
    num_correct = 0
    total = 0
    valid_loss = 0.0
    objective = nn.CrossEntropyLoss(reduction="sum")
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)

            valid_loss += objective(output, targets)

            total += targets.size(0)

            num_correct += (predicted == targets).sum().item()

    valid_acc = num_correct / total
    valid_loss /= total
    return valid_acc, valid_loss


def CNN(pretrain, device, load_model):
    
    acc_thresh = {
        0: (0.55, 0.65),
        1: (0.80, 0.90),
    }
    
    config = get_config_dict(pretrain)
    
    train_dataset, valid_dataset, test_transforms = load_dataset(pretrain)

    #Part 1
    if not pretrain:
        model = Net().to(device)
    #Part 2
    else:
        model = PretrainedNet().to(device)
    
    #Trains the model
    if not load_model:
        start = timeit.default_timer()
        train(config, model, train_dataset, valid_dataset, device, pretrain)
        stop = timeit.default_timer()
        run_time = stop - start
    
    if pretrain and load_model:
        # Load the model
        start = timeit.default_timer()
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])       
        
    accuracy = test(
                model,
                device,
                test_transforms
                )
    
    if pretrain and load_model:
        stop = timeit.default_timer()
        run_time = stop - start #Calculates runtime on test set inference
            
    score = compute_score(accuracy, acc_thresh[pretrain])
    result = OrderedDict(
        accuracy=accuracy,
        score=score,
        run_time=run_time
    )
    
    part = "2" if pretrain else "1"
    print(f"Result on Part {part}:")
    for key in result:
        print(f"\t{key}: {result[key]}")

class Args:
    """
    command-line arguments
    
    pretrained: set to 1 to load a pretrained model (Part 2)
    gpu: set to 0 to run on cpu
    load_model: set to 1 to load a saved model from a .pth file. (For part 2)
    """

    pretrained = 1
    load_model = 0
    gpu = 1

def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        print("WARNING: You have not installed paramparse. Please manually edit the arguments.")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    CNN(args.pretrained, device, args.load_model)


if __name__ == "__main__":
    main()