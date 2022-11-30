
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import json
import torch




def process_image(img_path):

    img = Image.open(img_path)
    
    img.thumbnail((256,256))
    img = img.crop((16,16,240,240))
    
    np_img = np.array(img)
    
    trans_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])

    resultant = trans_img(np_img).float()
    
    return resultant


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet161(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image, model, c_to_i, gpu, top_k, cat_names): 
    
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    model.eval()
    classes = []
    
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    
    with torch.no_grad():
        image = image.type(torch.FloatTensor).to(device)   
        image = image.unsqueeze(0)

        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(int(top_k), dim=1)    
            
        idx_to_class = {value : key for key,value in c_to_i.items()}
        for c in top_class[0]:
            classes.append(cat_to_name[idx_to_class[int(c)]])
        top_ps = top_ps.cpu().detach().numpy()[0]

    return top_ps , classes

    
    



