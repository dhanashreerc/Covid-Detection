import torch
import torch.nn as nn
import torchvision
import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tensorflow.keras.models import load_model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
idx_to_class = {0: 'normal', 1: 'covid', 2: 'pneumonia'}
params = {
#    "model": "densenet121",
    #"device": "cpu",
    "lr": 0.01,
    "batch_size": 32,
    "num_workers": 64,
    "n_epochs": 15,
    "image_size": 512, 
    "in_channels": 3, 
    "num_classes": 3
}

def Net(num_classes):
    model = torchvision.models.densenet121(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #model.classifier = torch.nn.Linear(in_features=1024, out_features=3)

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 512)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(512, 256)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(256, num_classes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=512),
        A.CenterCrop(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Detection')
    )
    
    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Detection':
        photo()
 

def welcome():
   # _, col2, _ = st.columns([1, 10, 1])
     st.markdown(
      """
      <style>
      .reportview-container {
        background: url("https://www.who.int/images/default-source/health-topics/coronavirus/corona-virus-getty.tmb-1200v.jpg?Culture=en&sfvrsn=217a6a68_36")
      }
     .sidebar .sidebar-content {
        background: url("https://www.who.int/images/default-source/health-topics/coronavirus/corona-virus-getty.tmb-1200v.jpg?Culture=en&sfvrsn=217a6a68_36")
    }
    </style>
    """,
     unsafe_allow_html=True
     )
   # with col2:
     st.write("")
     st.write("")
     st.write("")
     st.write("")
     st.write("")
     st.title('Covid 19 Detection using X-ray')
  #  with col2:
     st.write("")
     st.write("")
     st.write("")
     st.write("")
     st.subheader('Dhanashree Chavan')
     st.write("")
     st.subheader('Bhushan Gunjal')
     st.write("")
     st.subheader('Durvesh Talekar')
    


def photo(): 
        import torch
        if(torch.cuda.is_available() == False):
          checkpoint = torch.load('checkpoint.tar', map_location ='cpu')
        else:
          checkpoint = torch.load('checkpoint.tar')
        loaded_model = Net(params['num_classes'])

        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_criterion = checkpoint['loss']

        uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png', 'jpeg'])
        def load_image(img):
                im = Image.open(img)
                image_array = np.array(im)
                return image_array

        if uploadFile is not None:
                st.write("Original X-ray Image:")
                st.write("")
                img = load_image(uploadFile)

                final_img = cv2.resize(img, (299, 299))
                st.image(final_img)
                cv2.imwrite('1.jpg',img)
                #final_img0 = cv2.resize(img8, (255, 255))
                #im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                image = cv2.imread(r'1.jpg')

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_tensor = test_transforms(image=image)["image"]
                input_tensor = image_tensor.unsqueeze(0) 
                input_tensor = input_tensor.to(device)

                loaded_model.eval()
                prediction = np.argmax(loaded_model(input_tensor).detach().cpu().numpy())

                Predicted_Class = idx_to_class[prediction]

                model_output = loaded_model(input_tensor).detach().cpu().numpy().flatten()
                probabilities = softmax(model_output)
                if(Predicted_Class=='pneumonia'):
                    st.title("Be alert! Pneumonia is detected.")
                    a="Accuracy: "+str(max(probabilities)*100)[:4]+"%"
                    
                elif(Predicted_Class=='covid'):
                    st.title("Unfortunately Covid is detected in your Xray!")
                    a="Accuracy: "+str(max(probabilities)*100)[:4]+"%"
                    
                elif(Predicted_Class=='normal'):
                    st.title("Phew !!! Your X-ray image is normal.")
                    a="Accuracy: "+str(max(probabilities)*100)[:4]+"%"
                st.header(a)



        else:
                st.write("Make sure you image is in JPG/PNG Format.")


 

    
if __name__ == "__main__":
    main()

