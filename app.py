from fastapi import FastAPI
import torch
import clip
from PIL import Image
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import boto3
from io import BytesIO
from PIL import Image
import os
from pinecone import Pinecone, ServerlessSpec
import json

load_dotenv()

s3 = boto3.client('s3')

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = FastAPI()


class ImageArray(BaseModel):
    image: List[List[float]]

class Textformat(BaseModel):
    text: str

class ImagePathRequest(BaseModel):
    image_path: str

@app.post("/getfeatures/v1/image")
def image_features_using_clip(data: ImageArray):
    image = data.image
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    return {"image_features":image_features}

@app.post("/getfeatures/v1/text")
def text_features_using_clip(data: Textformat):
    text = data.text
    text = clip.tokenize([text]).to(device)
    print(text)
    text_features = model.encode_text(text).squeeze(dim = 0)
    # print(text_features.shape)
    indexes = pc.list_indexes()
    print(indexes)
    names = [ index['name'] for index in indexes]


    if 'upscaler1' not in names:
        pc.create_index('upscaler1', dimension=512 , metric="cosine",
                    spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                    ))
    index = pc.Index('upscaler1')

    indexes = pc.list_indexes()
    print(indexes)
    names = [ index['name'] for index in indexes]


    if 'upscaler1' not in names:
        pc.create_index('upscaler1', dimension=512 , metric="cosine",
                    spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                    ))
    index = pc.Index('upscaler1')

    result = index.query(
    vector=text_features.tolist(),  # Convert the embedding to a list
    top_k=10000,  # Retrieve top 10 most similar embeddings
    include_values=True,  # Whether to return the original embeddings
    include_metadata=True,  # Whether to return the metadata associated with the embeddings
    namespace=''  # Optionally specify a namespace if you're using one
    )

    ids = []
    for match in result['matches']:
        ids.append(match['id'])

    json_string = json.dumps(ids)
    # print(text_features)
    print(ids)
    return {"id" : json_string}

@app.post("/getfeatures/v1/storedimage")
def image_from_s3(request: ImagePathRequest):
    response = s3.get_object(Bucket=os.getenv('S3_BUCKET_NAME'), Key=request.image_path)
    png_data = response['Body'].read()
    image = Image.open(BytesIO(png_data))
    image.show()
    image = preprocess(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    print(image_features.shape)

    indexes = pc.list_indexes()
    print(indexes)
    names = [ index['name'] for index in indexes]


    if 'upscaler1' not in names:
        pc.create_index('upscaler1', dimension=512 , metric="cosine",
                    spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                    ))
    index = pc.Index('upscaler1')

    index.upsert(
        vectors=zip([request.image_path], [image_features.squeeze(dim=0)])
    )
    

