# Multi-Modal-Retriever

## Description
This is the search model that ranks the users images based on relevance

## Installation
1. Clone the repository:
2. Create a .env file in the root directory with the following parameters:
	AWS_ACCESS_KEY_ID=Your Access Key ID
	AWS_SECRET_ACCESS_KEY=Your Secret Access Key 
	AWS_DEFAULT_REGION=Your AWS Region
	S3_BUCKET_NAME=Your S3 Bucket Name
	AWS_SQS_QUEUE = Your SQS Queue URL

	PINECONE_API_KEY= Your Pinecone API key
	PINECONE_ENVIRONMENT= Your Pinecone Environment
3. Install dependencies 
	pip install -r requirements.txt
4. Run app.py using uvicorn on port 3000: python -m uvicorn app:app --port 3000
