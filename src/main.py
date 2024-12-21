#These are run and tested on Google Collab
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from Image import generate_image
from TTS import generate_voice
from Intermediate import inter

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

prompt="""<|system|>
You are a persuasive and factually accurate story-generating assistant with a journalistic rigor.
<|end|>
<|user|>
Given a claim and corresponding facts, generate a summarized well-structured story that meets the following criteria:

- Refute the claim using every provided fact, along with additional context from historically accurate trends and statistics.
- Limit the response to a maximum of 250 words.
- Use a strongly critical tone, directly highlighting the flaws in the claim and emphasizing its misleading nature.
- Include relevant background information, citing historically significant economic events or definitions (e.g., major inflation periods or economic milestones
- When writing about public figures, always use their correct title based on their current or former status:
     Example: Use “former President” for Donald Trump, who no longer holds office, and “President” for Joe Biden, who is currently in office.
     Similarly, for any retired officials, use titles like “former” or “ex-” as appropriate.
- The narrative-based summarized story should be coherent and persuasive, leaving no room for doubt about the validity of the refutation.
- The summarized story must follow a clear 3-paragraph structure:

  Paragraph1- Begin with historical background or context around the event or term in question.
  Paragraph2- Progress to opposing the claim in a detailed manner, referencing the provided facts and relevant historical events to refute it.
  Paragraph3- Conclude with a strong, fact-based analysis that highlights the flaws in the claim and clarifies the reality of inflation trends.

Claim: Trump said, “I think we have the worst inflation we've had in 100 years. They say it’s 48 years, I don’t believe it.”
Facts: Trump framed this as an opinion, but it's baseless nonetheless- wrong in two different ways. First, even when the inflation rate hit its Biden-era peak of 9.1% in June 2022, that 9.1% rate was the highest since 1981 – between 40 and 41 years prior, certainly not “100 years” and not even “48 years.” Second, inflation has declined sharply since the June 2022 peak, and the most recent available rate at the time he spoke, for July 2024, was 3.2% – a rate that, the Biden presidency aside, was exceeded as recently as 2011.
<|end|>
<|assistant|>
"""
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 2000,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": False,
}

output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])

#Generates Instruction for Image Generation
s=inter(output[0]['generated_text'])
# Generates Images
generate_image(s)
#Generates Speech
generate_voice(output[0]['generated_text'])
