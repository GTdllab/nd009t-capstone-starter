**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Capstone Project: Inventory Monitoring Using Object Count Estimation from Bin Images

**TODO:** Write a short introduction to your project.
The project is to Use AWS Sagemaker to train a pretrained model that can perform object count. By using sagemaker hyperparameter tuning and debugger and other ML improvement and deployment to make a pipeline of machine learning solution to do the test. 

## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.
the data is from !(here)[https://aws.amazon.com/marketplace/pp/prodview-2v2jac3cw2cba#resources], subset of it based on this file ```file_list.json``` 

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it
use this code to help access the data in AWS:
```
def download_and_arrange_data():
    s3_client = boto3.client('s3')

    with open('file_list.json', 'r') as f:
        d=json.load(f)

    for k, v in d.items():
        print(f"Downloading Images with {k} objects")
        directory=os.path.join('train_data', k)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for file_path in tqdm(v):
            file_name=os.path.basename(file_path).split('.')[0]+'.jpg'
            s3_client.download_file('aft-vbi-pds', os.path.join('bin-images', file_name),
                             os.path.join(directory, file_name))

download_and_arrange_data()
```

Total data files downloaded is 10443. they are all jpg files, with 1 object has 1228 files, 2 objects:2299, 3 objects:2666, 4 objects:2372, 5 objects:1875 
You can find more information about the data here.

Then split data into train (75%) test (15%) and valid (15%). 

upload train, test and valid folder to AWS through this :
```
!aws s3 cp Inventory_Image s3://capstone2025inventory/ --recursive --quiet
```


## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

ResNet50 as pretained model is used, because it has proven it's ability in CV task.

### Initial training

a initial training has done with random set of parameter lr: 0.001 and batch-size: 32 
the model can acheive 33.3% accuracy in test data, however from the confusion matrix !()[confusion_matrix1.png]


however from the confusion matrix, can see that class 3 and 5 perform poorly, the biggest prediction fall to other classes.

also analyze the debugger output: !()[debugger_output1.png]


Due to time and resource limited, only two hyperparameters used to fine tune are here with shorted range:

hyperparameter_ranges = {
    "lr": ContinuousParameter(0.0001, 0.001),
    "batch-size": CategoricalParameter([32, 64, 128]),
}

the best found here is .

##


## Machine Learning Pipeline
**TODO:** Explain your project pipeline.
first train the model with a random set of parameters, and then use hyperparamter tunning to select the best set of parameters. and then look through the debugger output to find out potential problems, and train the model again with fixes. 


## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
