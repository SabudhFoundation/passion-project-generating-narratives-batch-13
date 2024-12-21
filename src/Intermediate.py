#! pip install --upgrade transformers
#These are run and tested on Google Collab

from huggingface_hub import login
login("Your_API_KEY") # Enter your Api Key

def inter(x):
    import torch
    from transformers import pipeline

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    story= f"{x}"

    messages = [
            {
                "role": "system",
                "content": "Convert complex narrative into a single, powerful visual metaphor for image generation.Ensure the description avoids using text, numbers, or written symbols in the imagery."
            },
            {
                "role": "user",
                "content": f"""Transform this story into a precise, single-line image concept:

    Story Key Elements:
    - use graphs or tug-of-war to show contrast between actual claimed trends where needed

    Story Context: {story}

    Generate a concise, symbolic visual representation in one line."""
            }
        ]


    outputs = pipe(
        messages,
        max_new_tokens=77,
        do_sample=False,
        temperature=0.7,
        return_full_text=False

    )
    return outputs[0]["generated_text"]
