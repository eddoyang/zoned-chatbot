import sys

def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "(no question given)"
    print(f"you asked: {question}")


if __name__ == "__main__":
    main()