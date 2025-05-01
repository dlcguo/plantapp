# plantapp, a plant-disease diagnosis app for hobby gardeners
> Built on Streamlit for the frontend and uses a pre-trained ResNet-50 CNN backbone for vision

# Description
plantapp is a lightweight app that turns a quick snapshot of a leaf into an explainable diagnosis:
- Grad-CAM heat-map overlays show where the CNN focuses on so users see the patches most impactful to the model's decision
- English treatment tips are generated in real-time with Gemini 1.5 Flash
- Confidence scores detail model uncertainty
- One-click feedback form lets users flag misclassifications where responses are saved to a local/remote SQLite
- Runs entirely in the browser tab via the Streamlit framework

# Dataset
The PlantVillage dataset, containing 61,486 images, was used for training with 39 different classes of plant leaf and background images. The dataset can be acquired [here](https://data.mendeley.com/datasets/tywbtsjrjv/1). We used a split of 0.9/0.1 for training and validation directories. See [config.json](https://github.com/dlcguo/plantapp/blob/master/plantapp/config.json) and [dataset.py](https://github.com/dlcguo/plantapp/blob/master/plantapp/data/dataset.py) for how data directories should be configured.

# Training

Once training data is setup, call ```plantapp/train.py``` via 

    python3 plantapp/train.py

# Running Locally:
    git clone https://github.com/dlcguo/plantapp.git
    cd plantapp

    # Python env
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    # Put your API keys in config.json
    nano plantapp/config.json

    # Download weights
    git lfs pull

    # Launch UI
    streamlit run app.py

# Demo:
<p align="center">
  <img src="./plantapp/docs/demo.gif" width="75%">
</p>

# Disclaimer:
Treatment guidance and plant identification are automatically generated via Google Gemini and a ResNet backbone respectively. For critical decisions always consult a qualified agronomist or plant-health professional. 

# Open Source Code:
No open-source code was used for the purposes of this app.

# Citations:
J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network”, Mendeley Data, V1, doi: 10.17632/tywbtsjrjv.1
