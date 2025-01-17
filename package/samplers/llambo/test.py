import asyncio

import openai


# Azure OpenAI setup
openai.api_type = "azure"
openai.api_base = "https://optuna-pe.openai.azure.com/"  # Replace with your Azure OpenAI endpoint
openai.api_version = "2024-10-21"  # Use the version supported by Azure
openai.api_key = input(
    "Please enter your Azure OpenAI API key: "
)  # Replace with your Azure OpenAI API key

# Chat engine and user message
chat_engine = (
    "gpt-4o-mini"  # Replace with the deployment name of the model in your Azure OpenAI resource
)

message = [{"role": "user", "content": "Hello"}]
# message = [{"role": "user", "content": "Generate an essay, make it as long as possible"}]

# Retry configuration
retry_frequency = 1 / 2.5  # Number of retries per second
max_retries = 50  # Maximum number of retries
n = 1  # Number of responses per API call

# the limit seems to be based on retry_frequency*n/second, rather than input or output token length / second
# n*retry_frequency has to be smaller than a certain number X. the number gets smaller as n gets smaller.
# e.g., for n=10, X is (1/6) and retry_frequency has to be lower than 1/60
# for n=1, X is between (1/2) and (1/3)


# Async function to call Azure OpenAI Chat Completion with retries
async def get_chat_completion_with_retries():
    retry_interval = 1.0 / retry_frequency  # Calculate interval in seconds
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Sending request...")
            # Call Azure OpenAI
            resp = await openai.ChatCompletion.acreate(
                engine=chat_engine,
                messages=message,
                temperature=0.8,
                max_tokens=500,
                top_p=0.95,
                n=n,  # Generate multiple responses per attempt
                request_timeout=10,
            )
            # Print all responses in this trial
            for i, choice in enumerate(resp["choices"]):
                print(f"Attempt {attempt + 1}, Response {i + 1}: {choice['message']['content']}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        # Wait for the next attempt even if the request was successful
        if attempt < max_retries - 1:
            print(f"waiting for {retry_interval} seconds")
            await asyncio.sleep(retry_interval)


# Example usage
async def main():
    await get_chat_completion_with_retries()


# Run the main function
asyncio.run(main())
