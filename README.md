1. The model they are created will be a classifier from github issue they have to finish quickly in 5 minutes
2. Create a semantic search for coding question
	1. Github Issue
	2. Add github discussion
	3. add open source forum
	4. Find Code snippet for a code example of the question
	5. Add stack overflow for the question.


- The requirement will be using low abstraction for the project for very basic stuff
- We will use PyTorch no lightning. In the future I wished there was a faster way they can use lightning or composer 

1. The first part of the project is to get the dataset
	1. You will have to use the python_graphql_client to download the github issue using the graphiql api call.
		1. Create a new directory called multi_label_classification
		2. install miniconda ( if you have window install install wsl 2 and get familiar with vim)
		3. Create a new conda environment called aim
			```bash
				conda create -n AIM python==3.7 -y
				conda activate AIM
			```
		4.  know do you first conda install with python graphql
			```python
				pip install python-graphql-client
			```
	2. To explore the github graphiql  go to this link https://docs.github.com/en/graphql/overview/explorer
		1. Press the button with the big triangle pointing to the right
		2. You should see something like 
			```json
						{
			  "data": {
				"viewer": {
				  "login": "Rami-Ismael" ## Github Profile
				}
			  }
			}
			```
		3. Click the explore button
			1. just play
		4. Create your first graphiql query for github that will grab data about issue and labels from one repository
			```json
			{
			  "data": {
				"repository": {
				  "diskUsage": 101531,
				  "issue": {
					"id": "MDU6SXNzdWU1MDI4MTY1MzA=",
					"title": "Add TPU support",
					"labels": {
					  "edges": [
						{
						  "node": {
							"id": "MDU6TGFiZWwxMjk3MDkwNjg4",
							"name": "feature"
						  }
						},
						{
						  "node": {
							"id": "MDU6TGFiZWwxMjk3MDkwNjg5",
							"name": "help wanted"
						  }
						}
					  ]
					}
				  }
				}
			  }
			}
			
			```
	1. You will download the github issue over this github repository
		1. Download the first only 10 PyTorch Lightning ( This is your first challenge) 
		2. Create a github access key. This allowed to request more  from github from a single ip-address
			1. Go to setting on github
			2. Go developer setting in the github settings
			3. Select Personal access token
		3. Download the rest of PyTorch Lightning issues
	2. Dockerize the  code follow this tutorial
		1. https://www.youtube.com/watch?v=Gjnup-PuquQ&t=45s ( What is docker)
		2. https://www.youtube.com/watch?v=bi0cKgmRuiA&t=1024s9 ( How to create a docker)
	3. Modify the code to run through this multiple github repository
		3. PyTorch
		4. Optuna
		5. Pandas
		6. Numpy
		7. Zarr
		8. Hugging Face Transformers
		9. Pinecone
		10. Weavite
		11. Torch Metrics
		12. Ray Tune
		13. Weight and Bias 
		... ( You can add more if you want)
	4. redo the docker environment
	5. Run the docker image
	7.  The dataset will be store in json format. You must find a python library that can compress the library
	8.  The dataset will be store in azure.
2. The second part will be exploring the dataset  using scikit-learn , seaborn 
	1. to explore the dataset
		1. Target distribution
		2. The lengtht of the text
		3. Word Counts
		4. Word Legth
		5. Most Common Word
		6. Calculate the tf-idf of the most common word in that doesn't exist in common parlar with other respository
https://www.kaggle.com/code/datafan07/disaster-tweets-nlp-eda-bert-with-transformers/notebook
[[Data Science]]
	2. Explain the dataset for me
3. Learn about NLP
	1. Grooking Deep Learning Chapter 11 on NLP
	2. Go through Hugging Face Course
4. Create the multi-label classfication model with only pytorch and hugging face
5. Add Infrastructure tooling Weight and Bias
6. Add your model to the cloud 
7. Test your model (https://fullstackdeeplearning.com/)
	1. Monitor your model 

We should a text classfication model in 4 week.

Create a semantic search function that will produce an answer to people question about coding documentation
Example of Semantic Search being used and providing value 
- [[ZetaAlpha]](https://www.zeta-alpha.com/)
- [[Ranking YC Companies with a Neural Net]](https://evjang.com/2022/04/02/yc-rank.html)

8. Go through [[Haystack]] documentation
9. Go through [[Pinecone]] documentation
	# Learn 
	[README.md](https://github.com/Rami-Ismael/Toward-Functional-Safety#readme)

	## NLP For Semantic Search
	- [ ] Dense Vector
	- [ ] Sentence Transformers with MNR Loss
	- [ ] Domain Transfer
	- [ ] Data Augmentation with BERT
	- [ ] the Art of Asking Question with GenQ
	- [ ] DOmain Adaption with Generative Pseudo-Labeling

	## Documentation
	(https://www.pinecone.io/docs/)
	- [ ] Image Similarity
	- [ ] [[Haystack]] Integration
10. go through [[Weavite]] documentation
13. Building the semantic search pipeline
14. I want the pipeline find question and answer pair
15. add chat in the model
16. Add reinforcement learning to increase the model performance

Codex- Clone that can produce code from a propmpt to code 


Create an semantic search for for meme

Generate meme with diffusion model

Make Money

## Make a list a YouTube they have to watch
Week 1: https://www.youtube.com/watch?v=aircAruvnKk



Week 2 : Deep Learning Fundamentals ( https://www.youtube.com/watch?v=fGxWfEuUu0w&list=PL1T8fO7ArWlcWg04OgNiJy91PywMKT2lv&index=1)



Week 3: ML Project (https://www.youtube.com/watch?v=pxisK6RMn1s&list=PL1T8fO7ArWlcWg04OgNiJy91PywMKT2lv&index=11)

**I have some feedback**

We want to improve and update the course iteratively with your feedback. If you have some, please contact


**How much background knowledge is needed?**

Some prerequisites:

Good skills in **Python** 🐍
Basics in **Deep Learning and Pytorch**

If it's not the case yet, you can check these free resources:

- Python: [https://www.udacity.com/course/introduction-to-python--ud1110](https://www.udacity.com/course/introduction-to-python--ud1110)
- Intro to Deep Learning with PyTorch: [https://www.udacity.com/course/deep-learning-pytorch--ud188](https://www.udacity.com/course/deep-learning-pytorch--ud188)
- PyTorch in 60min: [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html