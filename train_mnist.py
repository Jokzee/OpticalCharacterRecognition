import argparse
from os import path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torchvision.datasets import EMNIST, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from mnist_classifier import *

MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataloader, args):
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        i = 1
        for (inputs, labels) in train_dataloader:
            i += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model(inputs)
            loss = loss_fcn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        train_loss_list.append(loss_data)
        print("Epoch {:05d}/{:05d} | Loss: {:.4f}".format(epoch + 1, args.epochs, loss_data))

        if epoch % 5 == 0:
            scores = []
            for (inputs, labels) in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(model, inputs, labels, loss_fcn)
                scores.append(score)
                val_loss_list.append((epoch, val_loss))
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))

    plt.plot(train_loss_list, label='training loss')
    val_loss_array = np.asarray(val_loss_list)
    plt.plot(val_loss_array[:, 0], val_loss_array[:, 1], label='validation loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.title("Training of the model on MNIST dataset")
    plt.savefig("results/training_losses", dpi=500)


def evaluate(model, inputs, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        outputs = model(inputs)
        loss_data = loss_fcn(outputs, labels)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        score = f1_score(labels.cpu().numpy(), predictions, average="micro")
        return score, loss_data.item()


def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for (inputs, labels) in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(model, inputs, labels, loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def visualize_predictions(model, visualization_dataset64, device):
    viz_dataloader = DataLoader(visualization_dataset64, batch_size=1, shuffle=False)
    plt.figure()
    with torch.no_grad():
        model.eval()
        for idx, (inputs, label) in enumerate(viz_dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            prediction = torch.argmax(outputs, dim=-1).cpu().numpy()
            inputs = inputs.cpu()
            plt.subplot(8, 8, idx+1)
            img = torch.transpose(inputs[0, 0, :, :], 0, 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            # plt.title("{}, {}".format(prediction[0], label[0]))
    plt.show()


def get_roi(gray_img):
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return gray_img[max(y-10, 0):min(y+h+10, gray_img.shape[0]-1), max(x-10, 0):min(x+w+10, gray_img.shape[1])]


def main(args):
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))

    # create the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    if args.mode == 'train':
        train_dataset = EMNIST(root='./data', split='mnist', train=True, download=True, transform=transform)
        # train_dataset, _ = train_test_split(train_dataset, train_size=int(0.1*len(train_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.mode != 'custom_mnist':
        test_dataset = EMNIST(root='./data', split='mnist', train=False, download=True, transform=transform)
    if args.mode == 'custom_mnist':
        custom_transform = transforms.Compose([
            lambda x: np.asarray(x),
            lambda x: cv2.cvtColor(255 - x, cv2.COLOR_RGB2GRAY),
            lambda x: cv2.GaussianBlur(x, (3, 3), 1),
            lambda x: get_roi(x),
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            lambda x: torch.transpose(x, 1, 2)
        ])
        test_dataset = ImageFolder('./data/CustomMNIST', transform=custom_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    n_classes = len(test_dataset.classes)

    # create the model, loss function and optimizer
    model = MNISTClassifier(n_classes).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataloader, args)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    if args.mode != 'train':
        visualization_dataset64, _ = train_test_split(test_dataset, train_size=64)
        visualize_predictions(model, visualization_dataset64, device)
    return test(model, loss_fcn, device, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "custom_mnist"], default="train")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="batch size used for training, validation and test")
    args = parser.parse_args()
    main(args)
