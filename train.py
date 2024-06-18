import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data.cifar import Cifar

label_list = ["dog", "cat", "people"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar')
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=20)
    args, unknown_args = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if args.dataset == 'cifar':
        train_dataset = Cifar("./dataset/cifar", label_list, preprocess, is_train=True, device=device)
        test_dataset = Cifar("./dataset/cifar", label_list, preprocess, is_val=True, device=device)
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False)
        test_text = clip.tokenize(label_list).to(device)
    else:
        train_dataloader = None
        test_dataloader = None
        text = None
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00001)

    best_val_acc = 0.0
    best_model_path = "best_model.pt"

    train_losses = []
    val_accuracies = []
    val_losses = []

    # Train
    for epoch in range(args.epoch):
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_dataloader)):
            image, text = batch
            image = image.to(device)

            text = clip.tokenize(list(text))
            text = text.to(device)

            logits_per_image, logits_per_text = model(image, text)

            batch_size = image.shape[0]
            labels = torch.arange(batch_size, device=device).long() 
            
            loss = (criterion(logits_per_image, labels) +
                    criterion(logits_per_text, labels)
                         ) / 2
            
            loss = loss.cpu()
            train_loss += loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss.item())
        print(f'Train Epoch[{epoch}] Loss{train_loss}')

        val_loss = 0.0
        val_acc, cat_acc, dog_acc, people_acc = 0.0, 0.0, 0.0, 0.0,
        for image, label in tqdm(test_dataloader):
            with torch.no_grad():
                image = image.to(device)

                logits_per_image, logits_per_text = model(image, test_text)

                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

                max_index = np.argmax(probs)
                preb = label_list[max_index]
                acc = 1. if preb == label[0] else 0.
                cat_acc += 1. if preb == label[0] and preb == "dog" else 0.
                dog_acc += 1. if preb == label[0] and preb == "cat" else 0.
                people_acc += 1. if preb == label[0] and preb == "people" else 0.

                val_acc += acc
                val_loss += criterion(logits_per_image, torch.tensor([label_list.index(label[0])], device=device)).cpu().item()
                

        val_acc /= len(test_dataloader)
        val_loss /= len(test_dataloader)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        print(f'Val Epoch[{epoch}] Acc {val_acc * 100}%')
        print(f'Val Epoch[{epoch}] Loss {val_loss}')
        print(f'Cat Acc:{cat_acc*3 / len(test_dataloader)* 100}%')
        print(f'Dog Acc:{dog_acc*3 / len(test_dataloader)* 100}%')
        print(f'People Acc:{people_acc*3 / len(test_dataloader)* 100}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with acc {val_acc * 100}%')

    print(f'Best val acc: {best_val_acc * 100}%')

    # Plot the training and validation loss and accuracy
    epochs = range(1, args.epoch + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    print(train_losses)
    print(type(train_losses))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Val Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Val Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_val_metrics.png')