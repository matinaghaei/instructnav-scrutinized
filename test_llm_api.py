import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

def main():
    client = OpenAI() 
    model_name = os.environ['GPT_API_DEPLOY']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("Interactive OpenAI Chat. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Add user's message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Call the ChatCompletion endpoint
        response = client.chat.completions.create(model=model_name, messages=messages)

        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()
        print(f"Assistant: {assistant_reply}\n")

        # Add assistant's reply to history so context is preserved
        messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
