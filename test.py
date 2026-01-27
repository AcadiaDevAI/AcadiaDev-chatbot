# import json
# import boto3

# session = boto3.Session(profile_name="bedrock-dev", region_name="us-east-1")
# client = session.client("bedrock-runtime")

# resp = client.invoke_model(
#     modelId="amazon.titan-embed-text-v1",
#     body=json.dumps({"inputText": "This is a test log entry"}),
#     contentType="application/json",
#     accept="application/json",
# )

# print(resp["body"].read().decode())

import json
import boto3

PROFILE = "bedrock-dev"
REGION = "us-east-1"
MODEL_ID = "mistral.mistral-7b-instruct-v0:2"  # change if your catalog shows a different one

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
client = session.client("bedrock-runtime")

prompt = "<s>[INST] You are a helpful assistant. Say hello in one line. [/INST]"

resp = client.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps({
        "prompt": prompt,
        "max_tokens": 120,
        "temperature": 0.2,
        "top_p": 0.9
    }),
    contentType="application/json",
    accept="application/json"
)

print(resp["body"].read().decode("utf-8"))

