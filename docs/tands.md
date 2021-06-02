# Tools and Services
General information about some of the tools and services used in this project.

---
## Amazon Web Services (AWS)
Amazon Web services (AWS) is a cloud computing platform that we used to produce our trained model.

We used two AWS services: 

1. Amazon Simple Storage
2. Amazon SageMaker

Lets take a look at these services in detail. 

### Amazon Simple Storage (Amazon S3): 
Amazon Simple Storage (Amazon S3) provides cloud-based data storage.

We use Amazon S3 to store our training and validation data. After creating our processed datasets we upload them to S3. When training a model these datasets will be fetched from S3.

### Amazon SageMaker: 

Amazon SageMaker is a machine learning service that allows developers to train tune and deploy models. 

#### Notebook Instances:

*SageMaker Notebook Instances* are machine learning compute instances that run the *Jupyter Notebook App*. These instances allow us to do a variety of tasks related to the production of machine learning models. 

After creating a *Notebook Instance* we can open a *Jupyter Notebook* within it. The *Jupyter Notebook* is where we write our code to: setup, run and evaluate training jobs. 

Check out [Code Walkthrough: Training](code-walthroughs.md#code-walkthrough-training) and [Code Walkthrough: Hyperparameter Tuning](code-walthroughs.md#code-walkthrough-hyperparameter-tuning) to see how we trained models from our Jupyter notebook.


---
## MXNet
MXNet is an open source deep learning library. We used one specific tool provided by the MXNet library called im2rec. This tool allowed us to create our datasets in the RecordIO data format. 

For more information about MXNet's RecordIO data format and the im2rec tool see [Creating a Dataset Using RecordIO](https://mxnet.apache.org/versions/1.8.0/api/faq/recordio).

Also check out [Code Walkthrough: Creating Datasets](code-walthroughs.md#code-walkthrough-creating-datasets) to see how we created the datasets in the RecordIO format.

---
## Boto3
We used the Boto3 AWS SDK to connect python scripts to Amazon S3 and download images. 

Check out [Code Walkthrough: Downloading the Images](code-walthroughs.md#downloading-the-images) to see how we used boto3 to download images.

Also check out the [official boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for more information.