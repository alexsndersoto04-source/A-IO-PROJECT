from tokenizers import Tokenizer
from tokenizers.models import BPE

def load_aio_tokenizer():
    # Diccionario inicial para A IO
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    print("Diccionario A IO configurado correctamente.")
    return tokenizer

if __name__ == "__main__":
    load_aio_tokenizer()
