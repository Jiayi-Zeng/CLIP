import argparse
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.cifar import Cifar

label_list = ["dog", "cat", "people"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cifar')
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--model_path", type=str)
    args, unknown_args = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if args.dataset == 'cifar':
        val_dataset = Cifar("./dataset/cifar", label_list, preprocess, is_test=True, device=device)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
        test_text = clip.tokenize(label_list).to(device)
    else:
        val_dataloader = None
        test_text = None
        raise NotImplementedError

    # Load the best model
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model.eval()
    val_acc, cat_acc, dog_acc, people_acc = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for image, label in tqdm(val_dataloader):
            image = image.to(device)

            logits_per_image, logits_per_text = model(image, test_text)

            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            max_index = np.argmax(probs)
            preb = label_list[max_index]
            acc = 1.0 if preb == label[0] else 0.0
            val_acc += acc

            if preb == "dog":
                dog_acc += acc
            elif preb == "cat":
                cat_acc += acc
            elif preb == "people":
                people_acc += acc

    val_acc /= len(val_dataloader)
    cat_acc /= len([label for _, label in val_dataloader if label[0] == "cat"])
    dog_acc /= len([label for _, label in val_dataloader if label[0] == "dog"])
    people_acc /= len([label for _, label in val_dataloader if label[0] == "people"])

    print(f'Overall Accuracy: {val_acc * 100}%')
    print(f'Cat Accuracy: {cat_acc * 100}%')
    print(f'Dog Accuracy: {dog_acc * 100}%')
    print(f'People Accuracy: {people_acc * 100}%')
