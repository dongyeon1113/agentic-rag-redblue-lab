import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(override=False)
    uvicorn.run("agentdojo_benchmark.gui:app", host="127.0.0.1", port=19020)


if __name__ == "__main__":
    main()
