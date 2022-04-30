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
		1. PyTorch Lightning
		2. PyTorch
		3. Optuna
		4. Pandas
		5. Numpy
		6. Zarr
		7. Hugging Face Transformers
		8. Pinecone
		9. Weavite
		10. Torch Metrics
		11. Ray Tune
		12. Weight and Bias 
		... ( You can add more if you want)
	4.  The dataset will be store in json format. You must find a python library that can compress the library
	5.  The dataset will be store in azure.
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

	# Documentation
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
