# Import nécéssaire
from transformers import AutoTokenizer,  BitsAndBytesConfig, LlamaForCausalLM
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

# Permet de séparer un texte (format str) en différents chunk selonde des seprators (de gauche à droite)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])

# Quantization en 8bits
nf8_config = BitsAndBytesConfig(load_in_8bit=True)

# Path des modèles utilisées
llama_path = "/workspace/Meta-Llama-3-8B-Instruct"


def parametre_Llama3():
    """
    Parametrage LLama3:
        
        - model:
        
            `LlamaForCausalLM` vient du module `transformers`. Permet de paramétrer llama3.
        
        - tokenizer:
        
           `AutoTokenizer`  permets de choisier le tokenizer adapté au model.
        

        \u23ce Retourne:

        model: `transformers.models.llama.modeling_llama.LlamaForCausalLM` (via `LlamaForCausalLM`)

        tokenizer: `transformers.tokenization_utils_fast.PreTrainedTokenizerFast` (via `AutoTokenizer`)

        True est seulement là pour être récupéré dans une variable afin de ne pas paramêtrer de nouveau inutilement

        

    """

    # attn_implementation permet d'accélerer l'inférence avec flash attentiton
    model = LlamaForCausalLM.from_pretrained(llama_path, attn_implementation="flash_attention_2",
                                                device_map="cuda", torch_dtype="auto", 
                                                quantization_config=nf8_config)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer, True



def generation_llama3(document:str, llama_model, tokenizer):
    """
    Prend en input un `str` contenant tout le texte avec les descriptions d'image afin de permettre à llama de générer le QCM.

    \u23ce Retourne une liste contenant toutes les générations faite
    """
    
    # Parametrage llama3
    

    # Séparer le document en plusieurs chunk afin de repndre plus simple son analyse par llama3
    # RecursiveCharacterTextSplitter est une méthode permettant de défnir comment sont découpé les chunks (cf. variable text_splitter)
    chunked_documents = text_splitter.split_text(document)

    # Récupère le nombre de chunk
    length = len(chunked_documents)

    liste = []

    # Pour chaque chunk, génère une réponse
    for i in range(length):

        # Juste pour vérifier l'avancement
        print("Texte", i+1, "sur", length)

        # Prompt pour Llama3 contenant le chunk
        f_prompt = f"""
        
        A partir de ce texte: {chunked_documents[i]}, 


            Fait 3 questions avec 4 choix de réponse.
            Vérifie que les réponses soit correctes.
            Les réponses doivent être en langage naturel.
        Ne rajoute pas de ligne de code à la fin. Contente toi de donner 3 questions.
        Sert toi uniquement du document, aucune autre connaissance
        Suis le format suivant:

        ##Question: [La question]

        ##Options:
        A. [Option A]
        B. [Option B]
        C. [Option C]
        D. [Option D]
        
        ##Reponse: [lettre de la bonne réponse]
        """

        # Encodage  de l'entrée
        inputs = tokenizer(f_prompt, return_tensors="pt").to("cuda")

        # Generation de la réponse
        reponse = llama_model.generate(inputs.input_ids, max_new_tokens=1000, repetition_penalty=1.2, temperature=0.7)

        # Décodage de la réponse
        reponse = tokenizer.batch_decode(reponse, skip_special_tokens=True)[0]

        # Pour enlever l'input de la réponse et supprime les espaces inutiles
        reponse = reponse[len(f_prompt):].strip()

        # Sauvegarde la page sur l'ordi pour vérification uniquement
        filename = f"/workspace/je_sais_pas/generation{i+1}.txt"

        with open(filename, "w") as file:
            file.write(reponse)

        # Ajoute la réponse dans une liste
        liste.append(reponse)

        # Vide de le cache pour ne pas gaspiller de la mémoire
        torch.cuda.empty_cache()

    # Retourne la liste des générations du ducoments
    return liste