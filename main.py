from app.infrastructure.llm_model import LLMModel


def main():
    model = LLMModel()

    response = model.client.chat.completions.create(
        model=model.model,
        messages=[
            {
                "role": "user",
                "content": "Say hello in one word.",
            }
        ],
    )

    print("\n===== FULL RESPONSE =====")
    print(response.model_dump())

    print("\n===== USAGE =====")
    print(response.usage)

    print("\n===== USAGE DICT =====")
    if response.usage is not None:
        print(response.usage.model_dump())


if __name__ == "__main__":
    main()