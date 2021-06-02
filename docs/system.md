# System Overview
Overview of the components and structures of our project. 

---
## Structure of S3 Bucket Containing Datasets

Datasets are stored in an S3 bucket named: ``sagemaker-multi-label-data``

The following shows the file structure of this bucket:

    sagemaker-multi-label-data/
        50-label-rs96/
            training/
                train.rec
            validation/
                val.rec
            output/
                training outputs
        ic-multi-label
            training/
                train.rec
            validation/
                val.rec
            output/
                training outputs

There are two datasets with different properties in this bucket. This table shows the properties of each of the datasets:

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-color:#93a1a1;border-style:solid;border-width:1px;color:#002b36;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#657b83;border-color:#93a1a1;border-style:solid;border-width:1px;color:#fdf6e3;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-1wig">Bucket </th>
    <th class="tg-1wig">Prefix</th>
    <th class="tg-1wig"># of Labels</th>
    <th class="tg-1wig"># Training Samples</th>
    <th class="tg-1wig">Image Size Pixels</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">sagemaker-multi-label-data</td>
    <td class="tg-0lax">50-label-rs96</td>
    <td class="tg-0lax">50</td>
    <td class="tg-0lax">98378</td>
    <td class="tg-0lax">96x96</td>
  </tr>
  <tr>
    <td class="tg-0lax">sagemaker-multi-label-data</td>
    <td class="tg-0lax">ic-multi-label</td>
    <td class="tg-0lax">334</td>
    <td class="tg-0lax">116945</td>
    <td class="tg-0lax">96x96</td>
  </tr>
</tbody>
</table>
<p>&nbsp;</p>

Training models with either of these datasets can be done through the files in our Jupyter Notebook.

---
##  Structure of Jupyter Notebook

The following shows the file structure of the Jupyter notebook:

    machine-learning-for-taggging-graphic-designs/
        334-label/
            resize-96/
                train.ipynb
                tune.ipynb
        50-label/
            resize-96/
                train.ipynb
                tune.ipynb

This table shows what dataset is used by what Jupyter Notebook file:

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-color:#93a1a1;border-style:solid;border-width:1px;color:#002b36;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#657b83;border-color:#93a1a1;border-style:solid;border-width:1px;color:#fdf6e3;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">Location of File in Jupyter Notebook</th>
    <th class="tg-0lax">S3 Location of Dataset Used by that file</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">334-label/resize-96/train.ipynb<br>334-label/resize-96/tune.ipynb</td>
    <td class="tg-0lax">sagemaker-multi-label-data/ic-multi-label/</td>
  </tr>
  <tr>
    <td class="tg-0lax">50-label/resize-96/train.ipynb<br>50-label/resize-96/tune.ipynb</td>
    <td class="tg-0lax">sagemaker-multi-label-data/50-label-rs96/</td>
  </tr>
</tbody>
</table>
<p>&nbsp;</p>

As you can see different files in the Jupyter Notebook use different datasets. This means that we are able to train a model with either 50-label images, or 334-label images.

---
## Data Overview
### Images

The raw data consists of ~155,000 classified images stored in S3. Each image can belong to one, or more of 334 classes. 

### List Files
List files includes the label data for each image as well as the path to the image. We use the list file to associate the images and their classes.
Each line in the list file takes the following format:

    Image-ID   label_1   label_2  ...  label_334    Image-Path  

The labels represent the classes that images can belong to. Label values can either be 0, meaning the image does not belong to that class, or 1 meaning the image does belong to that class.

For example, lets say we have 3 images that can belong to 3 classes: flower, leaf, and blue. The list file would look like this:


    0     1       0       0   image1.jpg
    1     0       1       1   image2.jpg
    2     1       1       1   image3.jpg

The first column is a unique ID for each image. The next 3 columns are the images label data and the last column is the path of the image. Lets add the column headers in for the sake of understanding our example:

    ID    flower  leaf    blue  path
    0     1       0       0     image1.jpg
    1     0       1       1     image2.jpg
    2     1       1       1     image3.jpg

**Note:** The column headers would not be included in the List file.

We can see that the first label is flower, the second is leaf and the third is blue. So, image1.jpg belongs to the class flower only, image2.jpg is belongs to both the leaf and blue class and image3.jpg belongs to all three classes.

List files are a crucial part of processing the data. They contain the label data associated with each image. Each image and its label data will be packed into the final format of the training dataset. 

---
## Data Processing

Using the raw images for training is possible, but inefficient. Processing the images allows for faster training and less resource consumption.

Processing the images involves the following steps:

1. Download images from S3.
2. Create dataset using RecordIO.

#### Downloading Images From S3
We were provided a list file containing the label data and S3 location of every image: 

        ID      labels 1-334      Path to image in S3

To download the images we used a python script to parse each line of the original list file, then write the S3 location of each image into a file called ``s3-paths.txt``. Then, using a tool called boto3, we connect to S3 and download every image to a local machine.

For more information on Boto3 see the [Tools and Services: Boto3](tands.md#boto3).

Check out [Code Walkthrough: Downloading the Images](code-walthroughs.md#downloading-the-images) to see how we downloaded the images.

#### Create Dataset Using RecordIO

We use MXNet's RecordIO data format for the dataset. This provides many benefits such as: reducing training time, reducing overall size of the dataset and more.

We have to convert the raw images and the label data into the RecordIO data format. To do this we use a tool called im2rec.py provided by MXNet. This tool takes our list file and the root directory containing our images and produces a .rec dataset. 

The .rec file that im2rec.py generates can be used as training data. We upload the dataset back to S3 and we are ready to use it for training.

Check out [Code Walkthrough: Creating Datasets](code-walthroughs.md#code-walkthrough-creating-datasets) to see how we created the datasets in the RecordIO format.

---
## Model Training
Training a machine Learning model with SageMaker is done through the SageMaker Notebook instance. Within the instance we write code to setup and run training jobs.

Training jobs in SageMaker include the following information:

- Location of training data.
    - The training data is stored in an Amazon Simple Storage bucket.
    - The URL of the bucket containing the training data is provided to the training job before launching it.
- Type of compute instance to use for training.
    - Training models is done in a separate compute instance.
    - There are several types of compute instances that can be used for training. 
    - The desired type of compute instance is provided to the training job before launching it.
- Location of where the output should be stored.
    - The URL of the bucket where outputs from a training job should be stored.
- Path of EC2 container registry where training code is stored.
    - This path contains an image of the actual training algorithm.
    - SageMaker provides several built-in training images that can be used.
    - The path of the container where the training image is stored is provided to the training job before launching it.

We write code in our notebook instance to specify all of this information before launching the training job. Additionally, we specify some model parameters that describe how the training should take place.

Check out [Code Walkthrough: Training](code-walthroughs.md#code-walkthrough-training) and [Code Walkthrough: Hyperparameter Tuning](code-walthroughs.md#code-walkthrough-hyperparameter-tuning) to see how we trained models from our Jupyter notebook.


---
## Model Deployment
Trained models can be deployed to endpoints with sagemaker. The endpoint can then receive input data from a client application and pass it to the trained model. The models output is then returned back to the client application through the endpoint. 

For more information on model deployment read [Deploy a Model to an Endpoint in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html).


