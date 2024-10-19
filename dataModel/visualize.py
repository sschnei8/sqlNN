# %%
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import os

#path to training data stored in env variaable
training_data_location = os.getenv('train_csv')
#Read CSV
train_data = pd.read_csv(training_data_location)

#Drop labels column
#Pixels, 0 = black & 255 = white
x_train = train_data.drop('label', axis = 1).values

def visualize_row(index):
    img = x_train[index].reshape(28,28) #reshape back to original array size
    plt.imshow(img, cmap = 'gray')
    plt.axis('off') #remove axis , just unnecessary noise 
    plt.show()

if __name__ == '__main__':
    visualize_row(3)
# %%
