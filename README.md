# Sentiment-Analysis-On-Juvenile-Delinquency-Using-BERT-Embeddings
Final Thesis Project - MSc Data Analytics

Criminal Juvenile Delinquency is one of the most prevalent problems in the modern
society. It generally occurs because of poverty, poor education, lack of awareness,
insufficient social infrastructure, and unstable family conditions. Although
there are juvenile justice systems in place, few offenders are penalized due to their
young age under the guise of teenage expression for less severe crimes. However,
sometimes benign crimes eventually propagate over the years into full-fledged antisocial
behavior recovering from which becomes virtually impossible. Such offenders
can be corrected with effective policy making. But it is difficult to draw policies
balancing social conduct and human rights in case of juveniles due to which identifying
public opinion in such cases is helpful in decision making. Hence, this project
aims at performing sentiment analysis by fine-tuning BERT transformer models on
twitter posts dataset to gauge public sentiment towards juvenile delinquency. We
also propose a contrast between a BERT fine-tuned model and Machine Learning
(Random Forest and Support Vector Classifier) model with BERT embeddings for
sentiment classification to determine whether fine-tuning is worth the effort for this
problem domain. The feature engineering approach using BERT features (100%
Class Accuracy) outperformed the fine-tuned BERT model (77% Class Accuracy)
on a benchmark of twitter post dataset on juvenile delinquency proving that in
terms of design complexity, execution time, and performance, features engineered
directly from the primary dataset using BERT has higher utility when compared
with fine-tuned transfer learning approach.

## Methodology 

![image](https://user-images.githubusercontent.com/98535942/218256719-0b90c6d1-ec49-4411-8001-53f97823eba4.png)
Figure 1: Project Methodology Framework

<img width="396" alt="image" src="https://user-images.githubusercontent.com/98535942/218256745-fe1da701-3aa0-4644-aade-74210bb5a18a.png">
Table 1: Twitter API Search Parameters

<img width="517" alt="image" src="https://user-images.githubusercontent.com/98535942/218256765-c1e11cb3-c370-40f6-a4be-d4f7891dfdd0.png">

<img width="408" alt="image" src="https://user-images.githubusercontent.com/98535942/218256779-b6d375d5-eddb-4731-ad6e-b19014d5a215.png">

<img width="306" alt="image" src="https://user-images.githubusercontent.com/98535942/218256802-de7fee3a-a472-4a69-81fb-0eb13531f923.png">

<img width="435" alt="image" src="https://user-images.githubusercontent.com/98535942/218256811-b0997d3b-08a6-4518-8a1a-d38bd0523ae1.png">

## Results

Implementation and evaluation of selective experimental models were managed using
available computational infrastructure to perform sentiment analysis on juvenile delinquency
and results were evaluated to reveal that, although the overall classification accuracy
of the fine-tuned models (77 %, highest for Model 7 and lowest 70% for Model 6)
increased with the increase in number of training epochs, its performance lagged compared
to the standard machine learning classifiers (RF- 98% and SVC- 100%) employed
with BERT embedding features. Not only did the accuracy performance suffer but the
precision and recall scores were not satisfactory compared to the RF and SVC models
(Models 9-10 in Table 5,6). It was also observed that increase training epochs and
parameter tuning for fine-tuned models indeed improved overall performance, it also
contributed to higher model complexity, exponential computational time, and resource
utilization. However, performing sentiment analysis using BERT embeddings as features
extracted directly from the concerned training data and then feeding it to the standard
classification models (RF, SVC, NB, KNN) yielded much better results and provided
faster conversion time.

Due to the computational limitation of the local system and cloud GPUs in this study,
we downsized the primary (Twitter Benchmark Dataset) and secondary datasets (Sentiment
140) from over 900,000 records to 14000 records and 1.6 million records to 100,000
records respectively and pre-processed for training and testing on BERT models. We also
had to control the parameters for model training phase to avoid system crashes and late
runtimes. An exhaustive evaluation of the related models such as DistilBERT, XLNet,
NB, KNN etc. and their respective combinations could have provided more concrete
grounds for analysis and results. Provided the available computational resources further
exploration of the research question can be extended to other NLP tasks such as Emotion
analysis and Stance Analysis. Nevertheless, evaluation results from this controlled
study proves that the implementation of fine-tuning BERT model approach (approach 1)
lags when compared with BERT embeddings used with ML models (approach 2) when it
comes to medium to large datasets and incurs high computational resource demand for
experimentation, hence it is preferable to use BERT embeddings with an ML model in
ensemble to perform sentiment analysis in this problem domain.

<img width="334" alt="image" src="https://user-images.githubusercontent.com/98535942/218256855-d35c8062-7b7f-421b-8004-e594e3de0c6a.png">

<img width="327" alt="image" src="https://user-images.githubusercontent.com/98535942/218256869-3a234494-e694-43b5-9c66-2018586e0778.png">

<img width="312" alt="image" src="https://user-images.githubusercontent.com/98535942/218256877-9d0d7c32-aa82-4408-a662-0a2fab55255f.png">

<img width="371" alt="image" src="https://user-images.githubusercontent.com/98535942/218256904-b7cde89d-a203-43bb-b843-8ab6a4c6cc78.png">

<img width="322" alt="image" src="https://user-images.githubusercontent.com/98535942/218256917-bc078d44-8a8d-471e-89e8-318c12263042.png">

## Conclusion

Sentiment analysis is one of the most powerful fields of study that has immense potential
for knowledge extraction and utilization in various fields, especially social domain.
Businesses and governments can utilize this technology to inspire social change for the
better. Social organizations must consider and explore this field of research to gain
valuable insights into policy and its impact on society on a regular basis. We implemented
a state-of-the-BERT model and fine-tuned it for our tasks and contrasted it with
a feature engineered BERT fed into an ML model for classification. We observed that
in terms of model complexity, training time, computational resource demand, and model
performance, the latter approach (RF-BERT: 98%, SVC-BERT: 100%) outperformed the fine-tuned approach (77%).

Owing to the limited computational resources, the research was done in a controlled
environment, however, with the necessary infrastructure larger model design and ensemble
techniques can be explored. This research can not only be extended to other domains
but also to different social media platforms such as facebook and Instagram for opinion
mining. For future work, other state-of-the-art models including roBERTa, DistilBERT,
XLNet, XLM etc. will be explored and an exhaustive evaluation will be conducted with
the sufficient infrastructure on expansive datasets covering larger demographics to yield
results for different age groups and sections of society.

## Code Links and primary dataset here :

### Codes: 
Part1_Final_code_Thesis_demo_Sentiment140.ipynb
Part2_Final_code_Thesis_demo_IMDB.ipynb

### Primary Input Dataset: 
Twitter_dataset_932k_Input.json

### Secondary Datasets:
1. Sentiment140_train.csv,
(https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets)
2. IMDB_Train.csv,
(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Link for Artifacts access:
https://drive.google.com/drive/folders/1XY3VK7rOR992FW1ajOAUpl6XDjB0-fzf?usp=share_link
Config Manual: x20259409_MSc_Thesis_Project_Config_Manual_Final.pdf

## Configuration Manual

1 Access the Primary Dataset and necessary files
1. The given link to Google drive is shared here for access to relevant project artifacts.
[Click me!](https://drive.google.com/drive/folders/1XY3VK7rOR992FW1ajOAUpl6XDjB0-fzf)
![image](https://user-images.githubusercontent.com/98535942/218256606-fe92f932-ceaa-42a2-b9ae-32dd8aa5a74e.png)
Figure 1: Google Drive location for relevant project files
2. Explore the shared drive location for Primary dataset (’Twitter dataset 932k Input˙json’)
and relevant files. .
3. Models and predictions were saved for reuse in Saved Models folder.
4. Colab notebooks starting with names Part1 and Part2 are the main project development
codes. You can start here.
2 Google Colab Pro+ IDE for development
2.1 Choose the right Colab Membership for you
For this development project Pro+ membership was subscribed however, regular plans
may allow execution (although execution time may vary).
![image](https://user-images.githubusercontent.com/98535942/218256619-a709a020-b23e-4df0-b926-e8c6a0270f57.png)
Figure 2: Google Colab Subscription for development and testing
2.2 Setup Google Colab IDE for Code Testing
1. Download the project files from the above mentioned drive link onto your google drive
and update the drive paths in the code.
2. Open collab notebooks ’Part1 Final code Thesis demo Sentiment140ipynb and
Part2 Final code Thesis demo IMDB.ipynb for code setup. Both the code can be run
independently for results. Sections have been designed properly to help with code understanding.
![image](https://user-images.githubusercontent.com/98535942/218256626-87f014ee-89a0-45f8-970c-d7e5044f0f35.png)
Figure 3: Google Colab Subscription for development and testing
3. Install Dependencies using 1st code cell to mount drive and necessary dependencies for testing.
![image](https://user-images.githubusercontent.com/98535942/218256630-7afbd5a0-c36a-4f44-b9d2-605c52f41ea5.png)
Figure 4: Google Colab Subscription for development and testing
3 Code Execution Steps
1. The whole code has been designed to run with ’Run all’ feature in Google Colab,
however it is advised to skip some steps to avoid long execution cycles. Comments and
Hints have been marked throughout the code for testing.
2. The commented read commands can be uncommented to boost execution time
instead of running section 3 Benchmarking step in each code.
3. All is set! You can now run the code section by section for Testing.
