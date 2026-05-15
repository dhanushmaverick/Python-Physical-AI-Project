#### Copy everything below this line into your app.py file ####
from openai import OpenAI

def main():
    # Make sure you have set OPENAI_API_KEY in your environment
    client = OpenAI()

    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            response = client.responses.create(
                model="gpt-5.4-nano",
                input=user_input
            )

            # Extract text safely
            output_text = ""
            if response.output and len(response.output) > 0:
                for item in response.output:
                    if hasattr(item, "content"):
                        for c in item.content:
                            if hasattr(c, "text"):
                                output_text += c.text

            print("ChatGPT:", output_text or "[No response text]")

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()