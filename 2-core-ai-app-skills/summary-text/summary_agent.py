import boto3
import json

BEDROCK_REGION = "us-west-2"
BEDROCK_TEXT_MODEL = "amazon.titan-text-express-v1"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v1"
BEDROCK_SERVICE = "bedrock-runtime"

client = boto3.client(service_name=BEDROCK_SERVICE, region_name=BEDROCK_REGION)

def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    points = event["queryStringParameters"]["points"]
    if text and points:
        llm_config = get_model_config(text, points)
        response = client.invoke_model(
            body=llm_config,
            modelId=BEDROCK_TEXT_MODEL,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        result = response_body.get("results")[0]
        return{
            "statusCode": 200,
            "body": json.dumps({"summary": result.get("outputText")})
        }
    else:
        return{
            "statusCode": 400,
            "body": json.dumps({"error": "Required Text and Points."})
        }

def get_model_config(text: str, points: str) -> json:
    prompt = f"""Text: {text}
    From the text above, summarize the story in {points} key points."""
    
    return json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig":{
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        }
    )