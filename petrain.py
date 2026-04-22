import torch
from model import AIO
from datasets import load_dataset

def start_training():
    print("Iniciando arquitectura A IO...")
    model = AIO()
    # Streaming del corpus de Wikimedia
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    print("Conexión con Wikimedia exitosa. Listo para computación masiva.")

if __name__ == "__main__":
    start_training()
