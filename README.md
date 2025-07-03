This repo contains the codebase for my masters thesis: **Semantic Segmentation And Depth Estimation Of Semi-Transparent Objects**  

You can likely run all models **except** the Depth model locally with a decent GPU.

⚠️ This code is mostly meant to be run on **DTU's HPC cluster**.  
All image files and pre-trained weights are **not included**.  
This repo exists as an illustration for employers and recruiters. If you wish to have the dataset you are free to e-mail me. 

# 📁 Structure of Directory

```text
Thesis/                         # Parent directory
│
├── Dataset/                    # Directory for input data
│   ├── Real/                   # Raw data
│   └── Synthetic/              # Processed data
│
├── Models/                     # Source code
│   ├── Logs/                   # Data logging
│   ├── models/                 # Networks and training models
│   │   └── model.py
│   └── Weights/                # Pretrained Models
│       └── New_Weights/        # Empty - Space for new models
│
├── Utility/                    # Helper Functions
│
├── README.md                   # Project Description
├── requirements.txt            # Local Dependencies
├── requirements_hpc.txt        # HPC Dependencies
├── environment.yml             # Local Conda Environment
└── hpc_env.yml                 # HPC Conda Environment
```

A requirements and conda environment file is supplied for both a local
and HPC setup.

To create a conda environment with all packages run 'conda env create -f your_yml_file.yml'  

To directly install requirements without conda env run 'pip install -r requirements.txt'

If you have issues with the CLIP package you can manually install using: pip install git+https://github.com/openai/CLIP.git


Naming conventions for the models are:  
Single - (Single views randomly chosen amongst the entire dataset)  
Multi - (All 6 views of the same setup)  
Double - (Single model into Multi model)  
Depth - (Outputs both semantic segmentation and depth predictions)  

As mentioned, you can likely run all except the Depth model locally with a decent GPU, however
this code is mostly meant to be ran from DTU's HPC cluster.

Pretrained weights are included.
If you want to train the models locally run: 'python3 -m Models.Train_Models --model='
Make sure to cd into Thesis to keep it as the top level package.

Arguments to Train_Models is:  
--model Model Name (This flag is required)  
--pretrained True/False  
--path (Absolute path to parent folder of datasets)  

**pretrained**: This flag is optional, default is True. Sets whether or not you want to use pretrained weights or train your own.
If you train your own the pretrained weights are preserved and your model will save in Models/Weights/New_Weights/.
Pretrained only retains the original weights. If you want to use your own you need to manually move and rename them.

**path**: This flag is optional. The default path is structured within the Thesis/ directory but if you want to use the HPC cluster
then you might need to move them to a scratch directory. Ensure to supply the absolute path to the parent directory ../Dataset for the datasets and keep them together:
Dataset/           
 ├── Real/     
 └── Synthetic/

If you want to run from HPC you can run 'sub < submit.sh', also from Thesis as the parent directory.
You need to edit the submit.sh file to include your desired flags if you want to change them.

The submit file is directed at the gpua100 queue. If it's busy when you want to use it, you can manually edit the file to gpuv100 instead.
If you use the voltash interactive shell instead of the a100sh you will not be able to run the Depth model, as it requires close to 30GB.
If it's really busy and you need to run it you can try to ask for 2 GPU's from the voltash node by setting:

### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 (Set this to 8 instead)
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process" (Set this to 2 instead)


If you want to run inference to visualize the predictions you need to do it locally as the cluster doesn't
have a graphical display option. You can however still look at the print statements.
Run 'python3 Models.Inference --model='

Arguments to Inference is:  
--model Model Name (This flag is required)  
--path (Absolute path to parent folder of model to load - Optional)  

Inference will run the best model, display the input image in resized form and then display a side-by-side plot of
a ground truth mask and model prediction for segmentation and depth (if available) respectively.
Sadly the best Single, Multi and Double have been lost, so some worse intermediate pretrained models are included.
A fully working model was saved in Depth which does both segmentation and depth estimation.
It's recommended to run Inference with the --Depth flag for this reason.


## Acknowledgements

This project takes inspiration and adapts code from:

- [MVTrans](https://github.com/ac-rad/MVTrans) by ac-rad
- Pre-trained backbone from OpenAi's [CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
