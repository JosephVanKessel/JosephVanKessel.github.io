# Code Walkthroughs
Lets look at how the code works.

---
## Code Walkthrough: Training
This walkthrough takes you through the code used to train a single model. We will look at the files in our Jupyter Notebook called ``334-label/resize-96/train.ipynb`` and ``50-label/resize-96/train.ipynb``. 

**Note:** These two files use different datasets for training, however the structure of the code is exactly the same.

The training code follows these steps:

### Import required libraries. Set role and session.
First we import some libraries that we need to use to create the training job.

    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.amazon.amazon_estimator import image_uris
    from sagemaker.image_uris import retrieve, config_for_framework

    role = get_execution_role()
    sess = sagemaker.Session()

### Set Data Channels.
Next we specify the training inputs. In other words we tell the training job where to get our training and validation data as well as where it should send its output.

First we format the S3 paths to the training and validation data. We also format the S3 path that will be used to store output. 

    bucket = 'sagemaker-multi-label-data'
    prefix = 'ic-multi-label'

    training = 's3://{}/{}/training/'.format(bucket, prefix)
    validation = 's3://{}/{}/validation/'.format(bucket, prefix)
    output = 's3://{}/{}/output'.format(bucket, prefix)

Next create create the training inputs and set the data channels. In other words we are telling the training job a little bit about the data it will use for training.

For example we have to set ``content_type='application/x-recordio'`` because our dataset uses the RecordIO data format.

    train_data = sagemaker.inputs.TrainingInput(
        training, 
        distribution='FullyReplicated', 
        content_type='application/x-recordio', 
        s3_data_type='S3Prefix'
    )

    validation_data = sagemaker.inputs.TrainingInput(
        validation, 
        distribution='FullyReplicated', 
        content_type='application/x-recordio', 
        s3_data_type='S3Prefix'
    )

    data_channels = {'train': train_data, 'validation': validation_data}

### Get Training Image 

We also need to provide a machine learning algorithm to our training job this is called a training image. We use SageMakers built-in Image Classification algorithm for our training image.

    training_image = retrieve('image-classification', sess.boto_region_name)

### Create Estimator

Next, we create an estimator object. Here we chose what type of instance the training job will use and how instances we want to run in parallel.

    multilabel_ic = sagemaker.estimator.Estimator(
        training_image,
        role, 
        instance_count = 1, 
        instance_type = 'ml.p3.2xlarge',
        output_path = output,
        sagemaker_session = sess
        )

Then we set the estimators hyper-parameters. The values we chose for these hyper-parameters can significantly effect our models accuracy. We may want to adjust them and re-train our models to see if we can get better accuracy.

Its important to make sure the ``num_classes`` is set to the number of classes in our dataset or else the training will fail. Similarly ``num_training_samples`` must be set to the number of samples in the training data, or else the training will fail.

For more information on hyper-parameters read [Image Classification Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/IC-Hyperparameter.html).

    multilabel_ic.set_hyperparameters(
        num_classes = 334,
        num_training_samples = 116945,
        augmentation_type = 'crop_color_transform',
        epochs=5,
        image_shape = "3,96,96",  
        learning_rate = 0.001,
        mini_batch_size = 256,
        multi_label = 1,
        use_weighted_loss = 1,
        optimizer = 'adam'
        )

### Run Training
Finally we can begin the training. We specify our data channels as the inputs for training then run the training job and wait for it to finish.

    multilabel_ic.fit(inputs = data_channels, logs = True)

---
## Code Walkthrough: Hyperparameter Tuning

This walkthrough takes you through the code used to automatically tune a models hyperparameters. We will look at the files in our Jupyter Notebook called 334-label/resize-96/tune.ipynb and 50-label/resize-96/tune.ipynb. 

Creating a hyperparameter tuning job is very similar to creating a normal training job. One difference is instead of hard-coding the values of hyperparameters, we set ranges of hyper-parameter values. The hyperparameter tuning job will automatically train and re-train models adjusting the hyperparameters within the ranges we set.

### Set Initial Hyperparameters
We provide the hyperparameters that cant be changed

    multilabel_ic.set_hyperparameters(
        num_classes=334,
        num_training_samples=116945,
        augmentation_type = 'crop_color_transform',
        epochs=5,
        image_shape = "3,96,96",
        multi_label=1,
        use_weighted_loss=1
        )

### Set Hyperparameter Ranges and Create Tuner
There are 3 types of parameters:

1. Continuous: all values in the specified range can be used.
2. Integer: all integers in specified range can be used.
3. Categorical: any of the values in a list can be used.

Here we set the parameter ranges and create the tuner object. 

The ``max_jobs`` property of the tuner tells the tuning job how many different models to train with different parameters.

The ``max_parallel_jobs`` property of the tuner tells the tuning job how many training jobs to run at once. 

    tuning_job_name = "imageclassif-job-{}".format(strftime("%d-%H-%M-%S", gmtime()))

    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(0.0001, 0.05),
        'mini_batch_size': IntegerParameter(126, 256),
        'optimizer': CategoricalParameter(['sgd', 'adam', 'rmsprop', 'nag'])}

    objective_metric_name = 'validation:accuracy'

    tuner = HyperparameterTuner(
        multilabel_ic,
        objective_metric_name,
        hyperparameter_ranges,
        objective_type='Maximize',
        max_jobs=2,
        max_parallel_jobs=1) 

### Run Tuning
Now we run the tuner and wait for it to finish.

    tuner.fit(data_channels, job_name=tuning_job_name, include_cls_metadata=False)
    tuner.wait()

### Tuner Output
The models created by the tuner can be viewed and deployed from the SageMaker dashboard. The hyperparameters used for each training job can also be viewed through the SageMaker dashboard.

---
## Code Walkthrough: Creating Datasets
This walkthrough will take you through the process of creating a new dataset.

### Requirements
- Local machine with at least 250 GB of free space.
- AWS credentials (aws_access_key_id and aws_secret_access_key)
- The file s3-paths.txt
- MXNet installed on local machine
- train.lst and val.lst files.

### Downloading the Images
The first step in creating the dataset is downloading all of the images. I did this with the following files and python script:

**Note:** aws_access_key_id and aws_secret_access_key will be omitted for privacy.

Files: ``s3-paths.txt`` contains all of the images S3 locations.

The script I used to download the images from S3 takes the following steps:

1. Connect to Amazon S3 with Boto3
2. Read each image location from s3-paths.txt
3. Download the image saving it in the ``images/`` directory.
4. Logging which images were downloaded successfully and which images were not.

Downloading images script: 

    import boto3
    import botocore
    from botocore.retries import bucket

    s3_paths = open('s3-paths.txt', 'r')

    error_logs = open('logs.txt', 'w+')

    no_download = open('no-download.txt', 'w+')

    downloaded = open('downloaded.txt', 'w+')

    session = boto3.Session(
        aws_access_key_id=,
        aws_secret_access_key=
    )
    s3 = session.resource('s3')

    bucket = 'pixelscrapper-user-content'

    for path in s3_paths:
        local = 'images/' + path.split('/')[-1].split('\n')[0]
        try:
            s3.Bucket(bucket).download_file(path.split('\n')[0], local)
            downloaded.write(path)
        except botocore.exceptions.ClientError as error:
            error_logs.write(str(error) + path)
            no_download.write(path)

After this script completes we will have all of the images stored in a directory called ``images``. 

### Generating the Dataset
After downloading the images, we need to process them to create our dataset.

First we get the path of the im2rec.py tool by running the following:

    import mxnet
    path = mxnet.test_utils.get_im2rec_path()
    print(path)

Copy the printed path and save it somewhere. We will use this path later on to run the im2rec tool.

At this point we should have the following:

- A file called: ``train.lst`` containing the label data and local paths to training images.
- A file called: ``val.lst`` containing the label data and local paths to validation images.
- The path to ``im2rec.py`` 
- ``images`` directory containing all downloaded images

The following shows the output of running ``im2rec.py -h``. You can see there are many options and arguments, so lets narrow down the most important ones. 

    usage: im2rec.py [-h] [--list] [--exts EXTS [EXTS ...]] [--chunks CHUNKS]
                    [--train-ratio TRAIN_RATIO] [--test-ratio TEST_RATIO]
                    [--recursive] [--no-shuffle] [--pass-through]
                    [--resize RESIZE] [--center-crop] [--quality QUALITY]
                    [--num-thread NUM_THREAD] [--color {-1,0,1}]
                    [--encoding {.jpg,.png}] [--pack-label]
                    prefix root

    Create an image list or make a record database by reading from an image list

    positional arguments:
    prefix                prefix of input/output lst and rec files.
    root                  path to folder containing images.

    optional arguments:
    -h, --help            show this help message and exit

    Options for creating image lists:
    --list                If this is set im2rec will create image list(s) by
                            traversing root folder and output to <prefix>.lst.
                            Otherwise im2rec will read <prefix>.lst and create a
                            database at <prefix>.rec (default: False)
    --exts EXTS [EXTS ...]
                            list of acceptable image extensions. (default:
                            ['.jpeg', '.jpg', '.png'])
    --chunks CHUNKS       number of chunks. (default: 1)
    --train-ratio TRAIN_RATIO
                            Ratio of images to use for training. (default: 1.0)
    --test-ratio TEST_RATIO
                            Ratio of images to use for testing. (default: 0)
    --recursive           If true recursively walk through subdirs and assign an
                            unique label to images in each folder. Otherwise only
                            include images in the root folder and give them label
                            1. (default: False)
    --no-shuffle          If this is passed, im2rec will not randomize the image
                            order in <prefix>.lst (default: True)

    Options for creating database:
    --pass-through        whether to skip transformation and save image as is
                            (default: False)
    --resize RESIZE       resize the shorter edge of image to the newsize,
                            original images will be packed by default. (default:
                            0)
    --center-crop         specify whether to crop the center image to make it
                            rectangular. (default: False)
    --quality QUALITY     JPEG quality for encoding, 1-100; or PNG compression
                            for encoding, 1-9 (default: 95)
    --num-thread NUM_THREAD
                            number of thread to use for encoding. order of images
                            will be different from the input list if >1. the input
                            list will be modified to match the resulting order.
                            (default: 1)
    --color {-1,0,1}      specify the color mode of the loaded image. 1: Loads a
                            color image. Any transparency of image will be
                            neglected. It is the default flag. 0: Loads image in
                            grayscale mode. -1:Loads image as such including alpha
                            channel. (default: 1)
    --encoding {.jpg,.png}
                            specify the encoding of the images. (default: .jpg)
    --pack-label          Whether to also pack multi dimensional label in the
                            record file (default: False)

We already have the training and validation list files, so we are not interested in using im2rec to create those. We are primarily concerned with using im2rec to create the databases.

The most important arguments for creating our dataset using im2rec are:

    --resize RESIZE       resize the shorter edge of image to the newsize,
                            original images will be packed by default. (default:
                            0)
    --num-thread NUM_THREAD
                            number of thread to use for encoding. order of images
                            will be different from the input list if >1. the input
                            list will be modified to match the resulting order.
                            (default: 1)

    --pack-label          Whether to also pack multi dimensional label in the
                            record file (default: False)

**Important:** you can change the resize value and number of threads in the following commands if you want. These are just examples.

Create the training record by running:

    $ path/to/im2rec.py --resize 256 --num-thread 4 --pack-label train.lst images

This will generate the files ``train.rec`` and ``train.idx``.

Next, create the validation record by running:

    $ path/to/im2rec.py --resize 256 --num-thread 4 --pack-label val.lst images

This will generate the files ``val.rec`` and ``val.idx``.

``train.rec`` and ``val.rec`` are now ready to be uploaded to Amazon S3 where they can be used as training inputs.

---