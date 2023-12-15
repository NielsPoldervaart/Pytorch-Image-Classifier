import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Check if cuda (GPU) is available for the current machine
# torch.cuda.is_available()

# Get the MNIST dataset, which consists of handwritten numbers 0-9
train = datasets.MNIST(root="dataset", download=True, train=True, transform=ToTensor())

# DataLoader to handle batching of the dataset
# Batch size is set to 32
dataset = DataLoader(train, 32)

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # First Convolution layer: Input channels: 1 (grayscale), Output channels: 32, Kernel size: (3,3)
            nn.Conv2d(1, 32, (3,3)),
            # Apply Rectified Linear Unit (ReLU) activation function
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            # Convert the 3D tensor to a 1D tensor
            nn.Flatten(),
            
            # Flatten layer, followed by a fully connected layer
            # 28-6, MNIST images are 28px by 28px, each Conv layer above crops by 2, 2*3 = 6 making (28-6).
            # Last Conv batch size is 64
            # 10 outputs, 0-9 from MNIST dataset
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss and optimizer 
# Send our ImageClassifier() to the GPU, If cuda not available, use 'cpu'
clf = ImageClassifier().to('cuda')
# Pass the classifier parameters, and the learning rate
opt = Adam(clf.parameters(), lr=1e-3)
# Loss function
loss_fn = nn.CrossEntropyLoss()

def train_model():
    # Train for 10 epochs
    for epoch in range(10):
        for batch in dataset:
            # Unpack data
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')

            # Generate a prediction
            yhat = clf(X)

            # Calculate loss
            loss = loss_fn(yhat, y)

            # Apply backprop
            # Zero out existing gradients
            opt.zero_grad()
            # Calculate gradients
            loss.backward()
            # Apply gradient descent
            opt.step()

        # Print training data
        print(f"Epoch:{epoch} loss is {loss.item()}")
    
    # Save trained model to machine
    with open('./model/nn_model.pt', 'wb') as f:
        save(clf.state_dict(), f)

def make_prediction(test_img):
    # Load trained model into classifier
    with open('./model/nn_model.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    # Import image for prediction
    img = Image.open(test_img)
    # Convert image to grayscale, because of First Conv2d layer (input channel === 1, grayscale)
    img = img.convert('L')
    # Convert image ToTensor()
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    # Set the model to evaluation mode
    clf.eval()

    # Make prediction
    with torch.no_grad():
        output = clf(img_tensor)

    # Get the predicted class and confidence score
    predicted_class = torch.argmax(output).item()
    confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

    print(f"Predicted digit: {predicted_class}, Confidence: {confidence * 100:.2f}%")


# Call functions
if __name__ == "__main__":
    # Uncomment the line below to train the model
    # train_model()

    # Make a prediction using a sample image
    make_prediction('./img/5.jpg')