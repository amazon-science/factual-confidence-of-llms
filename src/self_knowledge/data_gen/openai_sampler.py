from openai import OpenAI

# Usage of this requires you set up your OpenAI API credentials
# used for sampling


class gpt:
    """a dummy model that calls on the openai API. The generate function is replaced by an API call, nothing else is implemented."""

    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.client = OpenAI()

    def generate(self, input_text, **kwargs):
        max_new_tokens = (
            kwargs.get("max_new_tokens", 10) if hasattr(kwargs, "max_new_tokens") else 10
        )
        temperature = kwargs.get("temperature", 0.7) if hasattr(kwargs, "temperature") else 0.7
        messages = [
            {
                "role": "system",
                "content": "provide a short completion to the following sentence:",
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text
