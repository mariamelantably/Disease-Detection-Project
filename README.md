# Disease Detection
This project aims to classify images of leaves into diseases using various AI techniques and then provide next steps for treatment using a RAG, allowing us to solve the troubling problem of disease prevention in agriculture. 

## Features
### The Dataset
For this project, three different datasets were used:
- PlantVillage (TFDS): https://www.tensorflow.org/datasets/catalog/plant_village. Used for testing.
- PlantDoc (Roboflow detection): https://public.roboflow.com/object-detection/plantdoc/. Used for few-shot examples for the LLM based object detection.
- New Plant Diseases (Kaggle): https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. Used to train the model on classification.

The testing and training datasets used 38 labels - each consisting of a crop (the main label) and the disease (the sub-label). There are 14 crops represented in the dataset, with various numbers of diseases per crop. 

### LLM based object detection 
The first step of the project pipeline is the LLM based object detection. Using the roboflow dataset, I provided 2 randomly chosen examples showing a leaf identified within a larger object, which I then passed into the GenAI API. 
Following this, the rest of the images were passed in for object detection and cropped using the PIL library. 

### Classification
This section uses the GenAI API to train the model on object classification, by passing in the training data along with their labels, and asking the model to return a structured output, consisting of each training image with a unique id and the detected label.
I also created a CNN model for the data (in the notebook saved.ipynb), with 87% accuracy. 
- 77% overall accuracy (despite running for far shorter and with far less training data than the CNN)
- 95% accuracy at predicting the crop
- accuracy per crop correlated negatively with number of sub-labels per crop
- 80% accuracy in predicting the disease, given the crop name as part of the data

### RAG
The final step of the project pipeline consists of using a hybrid RAG with an Ollama model to provide a treatment plan, with additional context from trusted agronomy docs (references provided in the directory) to prevent LLM hallucinations. 
For the sparse index, BM25s was used. FAISS was used for the dense index. The results from both indices were combined using rank reciprocal fusion.

The overall project receives the image of the leaf, identifies the leaf and crops, predicts the crop and disease and uses them for the treatment plan using RAG, and then outputs the predicted label, the confidence in the prediction, and the RAG treatment plan. For example:

The detected crop is peach and the detected disease is bacterial spot, with a confidence level of 0.94
Given the information, here are the treatment mechanisms and prevention mechanisms for bacterial spot on peaches:

**Treatment Mechanisms:**

1. **Copper-based fungicides:** Copper-based fungicides can help control bacterial spot by killing fungal pathogens that cause the disease.
2. **Boric acid fungicides:** Boric acid fungicides can also be effective in controlling bacterial spot, as they target the fungal pathogens and inhibit their growth.
3. **Systemic fungicides:** Systemic fungicides, which are absorbed by the plant through the roots, can provide long-lasting control of bacterial spot.
4. **Horticultural oil applications:** Horticultural oils, which are used to control fungal diseases on plants, can also be effective in controlling bacterial spot.

**Prevention Mechanisms:**

1. **Early detection and treatment:** Regular monitoring for signs of disease, such as spotting or yellowing of leaves, can help detect bacterial spot early and enable prompt treatment.
2. **Crop rotation:** Rotating crops with different growing conditions can reduce the risk of fungal diseases like bacterial spot, which can be spread by contaminated soil or equipment.
3. **Soil disinfection:** Disinfecting the soil before planting peaches can help reduce the risk of bacterial spores becoming airborne and infecting new plants.
4. **Biological control methods:** Introducing biological control agents, such as Trichoderma fungi, which are naturally occurring fungal pathogens that compete with bacterial spot-causing pathogens for resources, can also be effective in controlling the disease.

It's worth noting that prevention is key, especially for commercial growers and homeowners who should take proactive measures to minimize the risk of bacterial spot on peaches.

## Installation
Please ensure the following dependencies are installed:
- kagglehub
- tensorflow_datasets
- google
- PIL
- ollama
- numpy
- dotenv
- pandas
- tabulate
- matplotlib

These can be installed by running the following in your terminal
`pip install depedency`

Using Gemini API requires an API key connected to an account with sufficient tokens. Additionally, HF was used for the BM25s index, which requires an account and a token. Tokens should be loaded in using environment variables. 

## Acknowledgements 
I would like to thank the team at VAIS for giving me the opportunity to intern with them and create this project, as well as expand my knowledge of GenAI and Neural Networks

## Contact

Email: mariamantably@gmail.com 

Instagram: @mariamelantable 

Github: @mariamelantably

Linkedin: (www.linkedin.com/in/mariam-elantably-ab0559290)
