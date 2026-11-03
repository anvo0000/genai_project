import json

from summary_agent import handler
event = {
    "body": json.dumps(
        {
            "text": """Meet Twin Labs, a Paris-based startup that wants to build an automation product for repetitive tasks, such as onboarding new employees to all your internal services, reordering items when you’re running out of stock, downloading financial reports across several SaaS products, reaching out to potential prospects and more.
“Twin’s starting point is a science-fiction idea. We saw the development of the technical capabilities of LLMs — foundation models. And the question we asked ourselves was whether we’d be able to duplicate ourselves by training an AI agent on the way we perform our tasks,” Twin Labs co-founder and CEO Hugo Mercier told me.
In Twin Labs’ case, the most interesting thing isn’t what they’re doing — improving internal processes — but how they’re doing it. The company relies on multimodal models with vision capabilities, such as GPT-4 with Vision (GPT-4V), to replicate what humans usually do.
Before landing on multimodal models, Twin Labs first tried to develop autonomous agents using traditional LLMs. “We’ve tested lots of things, we’ve implemented research papers, we’ve tested open source GitHub repositories. Overall, the conclusion is that LLMs are completely unreliable. This means that LLMs are making the wrong decisions,” Mercier said. “In the end, the task isn’t done.”
According to him, GPT-4V has been trained on a lot of different software interfaces and the underlying code bases, which unlocked new possibilities. “When you show an interface, it understands the feature behind the button,” Mercier said.
Unlike Zapier and other automation products, Twin Labs doesn’t rely on APIs and designing complicated multi-step processes. Instead, Twin Labs is more like a web browser. The tool can automatically load web pages, click on buttons and enter text."""
        }
    ),
    "queryStringParameters":{"points": "3"}
}
response = handler(event,{})
print(response)