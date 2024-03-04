import json
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
############################
#                          #
#       PARAMETRES         #
#                          #
############################

# Chemin vers les jeux de données
file_path = ['dataset_drive11', 'dataset_drive8']

#Modèle
from model import Resnet18, CNNModel, LSTM
model = Resnet18()

# Fonction de perte
from loss import CustomLossFunction
loss_fn = CustomLossFunction(alpha=0.3) 

optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001) # Optimiseur

# Hyperparamètres
epochs = 5 #Nombre d'époques
batch_size = 32 #Taille des batchs
pourcentage_train = 0.8 #Pourcentage des données utilisées pour l'entrainement

schuffle = True #Mélange des données

##########################################
#                                        #
#       Récupération des données         #
#                                        #
##########################################

# Define a function to load images and corresponding labels
def load_images_with_labels(file_path=['','']):
    images = []
    throttle= []
    angle = []
    #filepath is a list of str
    for path in file_path:
        with open(path+'/labels.json', 'r') as file:
            data = json.load(file)



        for img_filename, values in data.items():
            # Load image using PIL
            img_path = path+"/images/"+img_filename  # Modify the path to your image directory
            image = Image.open(img_path)
            images.append(image)

            # Extract throttle and angle values
            throttle.append(values["user/throttle"])
            angle.append(values["user/angle"])

    return images, throttle, angle

# Load images and corresponding labels
images, throttle_values, angle_values = load_images_with_labels(file_path)

# Define transformations to apply to the images (if needed)
transform = transforms.Compose([
    #transforms.Resize((120, 160)),  # Resize the images to 120x160 pixels
    transforms.ToTensor(),  # Convert the images to tensors
])

# Transformation en tenseurs
transformed_images = [transform(img) for img in images]

values_tensor = torch.tensor([throttle_values, angle_values]).transpose(0, 1)
images_tensor = torch.stack(transformed_images)

print("Shapes - Images:", images_tensor.shape, "Throttle+angle:", values_tensor.shape)

#########################################
#                                       #
#       Préparation des données         #
#                                       #
#########################################

num_batches = len(images_tensor) // batch_size
num_images = num_batches * batch_size

# Define the number of images to use for validation
validation_indice = int(pourcentage_train * num_images)
print("Number of images:", num_images, "Validation indice:", validation_indice)

# Shuffle the images and labels
#if schuffle:
   # indices = torch.randperm(num_images)
    #images_tensor = images_tensor[indices]
    #values_tensor = values_tensor[indices]

if schuffle:
    num_groups = num_images // 3
    indices = torch.randperm(num_groups)
    
    # Créer des indices pour chaque groupe de 3 images
    shuffled_indices = indices.repeat(3) * 3 + torch.arange(3).unsqueeze(1)
    shuffled_indices = shuffled_indices.view(-1)
    
    # Vérifier si le dernier groupe a été étendu pour éviter les dépassements
    last_group_size = num_images % 3
    if last_group_size != 0:
        shuffled_indices = shuffled_indices[:-last_group_size]  # Ignorer les indices qui dépassent

    # Utiliser les indices mélangés pour réorganiser les images et les valeurs par groupe de 3
    images_tensor = images_tensor[shuffled_indices]
    values_tensor = values_tensor[shuffled_indices]

# Split the images and labels into training and validation sets
valid_images = images_tensor[validation_indice:]
valid_values = values_tensor[validation_indice:]
train_images = images_tensor[:validation_indice]
train_values = values_tensor[:validation_indice]

# Print the shapes of the training and validation sets
print("Training set - Images:", train_images.shape, "Values:", train_values.shape)
print("Validation set - Images:", valid_images.shape, "Values:", valid_values.shape)

##############################
#                            #
#       Entrainement         #
#                            #
##############################

def load_batch(images, values, batch_size, batch_number):

    start_index = batch_number * batch_size
    end_index = start_index + batch_size

    batch_images = images[start_index:end_index]
    batch_values = values[start_index:end_index]

    return batch_images, batch_values


# Define a function to train the model
def train_model(model, loss_fn, optimizer, num_epochs, batch_size, train_images, train_values):

    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1, "/", num_epochs,"...")

        # Define a list to store the losses for each batch
        train_loss=0.0

        for batch in range(len(train_images) // batch_size):

            optimizer.zero_grad()


            batch_images, batch_values = load_batch(train_images, train_values, batch_size, batch)

            batch_images = batch_images.to(device)
            batch_values = batch_values.to(device)

            batch_predictions = model(batch_images)


            batch_loss = loss_fn(batch_predictions, batch_values)


            batch_loss.backward()

            optimizer.step()

            train_loss += batch_loss.item()
        


        valid_loss=0.0

        for batch in range(len(valid_images)//batch_size):
            batch_images, batch_values = load_batch(valid_images, valid_values, batch_size, batch)

            batch_images = batch_images.to(device)
            batch_values = batch_values.to(device)
            batch_predictions = model(batch_images)

            batch_loss = loss_fn(batch_predictions, batch_values)

            valid_loss += batch_loss.item()


        train_losses.append(train_loss / len(train_images))
        valid_losses.append(valid_loss / len(valid_images))

        print('Training Loss: {0}\nValidation Loss: {1}'.format(train_losses[-1], valid_losses[-1]))
    
    return train_losses, valid_losses

if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda') 
else:
    print("Training on CPU")
    device = torch.device('cpu')

model=model.to(device)

#launch training
train_losses, valid_losses=train_model(model, loss_fn, optimizer, epochs, batch_size, train_images, train_values)

# Save the model's parameters to a file
torch.save(model.state_dict(), "model.torch")

# Plot the training and validation losses
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend()
plt.savefig(file_path[0]+"-"+type(model).__name__+"-epoch"+str(epochs)+"-batch_size"+str(batch_size)+".png")
plt.show()