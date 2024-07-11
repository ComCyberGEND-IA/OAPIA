# Import nécéssaire
import fitz
import io
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor,  BitsAndBytesConfig
import torch


# Vide le cache afin dêtre sur que toute la mémoire disponible est utilisé pour la génération
torch.cuda.empty_cache()


# Définition des limites
# limite minimale des images à extraire (en pixels)
dimlimit_width = 0
dimlimit_height = 0

# Ratio minimum de la taille de l'image par rapport à la taille de la page
size_min = 0

# Taille absolue minimale des images à extraire (en octets)
abssize = 0


def parametre_phi3_vision():
    """
    Paramétrage du modèle phi 3 Vision:

    - para:

        `False` permet de paramétrer le modèle.

        `True` empêche de reparamétrer à chaque fois (ce qui fait gagner du temps).

    \u26A0 para doit être utilisé dans une boucle en dehors de la fonction pour ne pas reparamétrer à chaque fois. \u26A0

    - processor:

        `AutoProcessor` vient du module `transformers`. Permet de gérer à la fois les images et le texte.

    - model:

        `AutoModelForCausalLM` vient du module `transformers`. Permet de paramétrer automatiquement le modèle utilisé localement.

    - messages:

        Prompt envoyé au modèle.

    - prompt:

        `processor.tokenizer.apply_chat_template` permet de rendre le prompt compréhensible par le modèle.

    \u23ce Retourne:

            -processor: `transformers_modules.Phi-3-vision-128k-instruct.processing_phi3_v.Phi3VProcessor` (via `AutoProcessor`)

            -model : `transformers_modules.Phi-3-vision-128k-instruct.modeling_phi3_v.Phi3VForCausalLM` (via `AutoModelForCausalLM`)

            -prompt: `str`

            -True est seulement là pour être récupéré dans une variable afin de ne pas paramêtrer de nouveau inutilement.

    """
    # Quantization en 8bits
    nf8_config = BitsAndBytesConfig(load_in_8bit=True)

    # Path des modèles utilisées
    phi_path = "/workspace/Phi-3-vision-128k-instruct"

    processor = AutoProcessor.from_pretrained(phi_path, trust_remote_code=True, use_fast=True)

    # Modèle avec quantisation 
    model = AutoModelForCausalLM.from_pretrained(phi_path, attn_implementation="flash_attention_2",
                                                    device_map="cuda", trust_remote_code=True, torch_dtype="auto",
                                                    quantization_config=nf8_config)

    # Prompt utilisé pour la description d'image
    messages = [{"role": "user", "content": "\Décris l'image. Tu ne dois rien inerprêter\n<|image_1|>"},
                {"role": "system", 'content': 'français'}]

    # Tokenization
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False)


    
    return processor, model, prompt, True
    


def generation_phi3_vision(liste_image:list, doc:fitz , xreflist:list, text:str, processor, model, prompt) :
    """ 
    Fonction qui génère une description d'image

    - liste_image: 

        Liste d'image récupérée avant

    - doc: 
    
        Récupère le document ouvert avec le module `fitz` afin d'extraire les images grace au tag (xref)
    
    - xreflist:
    
        liste des tags précédemment utilisés.
    
    - text: 

        Permet de faire rejoindre le texte extrait avec la description faite dans la fonction
    
    processor_phi, phi_model, prompt_phi récupère le paramêtrage fait dans parametre_phi3_vision

    \u23ce Retourne la description générée pour toutes les images de la page
    """

    # Boucle pour gêrer chaque image de la liste
    for img in liste_image:

        # Récupère le tag de l'image
        xref = img[0]
        
        # Ignorer l'image si elle a déjà été extraite pour éviter les doublons (continue passe à la prochaine itération)
        if xref in xreflist:  
            continue

        width = img[2]
        height = img[3] 

        # Ignorer les petites images
        if width <= dimlimit_width or height <= dimlimit_height:  
            continue

        # Extraire l'image avec le tag 
        image = doc.extract_image(xref)  

        # Récupère le sinformations pour pouvoir lire l'image 
        imgdata = image["image"] 

        # Ignorer les images avec un faible ratio de taille
        if width * height <= size_min:  
            continue

        # Permettre d'utiliser l'image avec le module Image
        img_io = io.BytesIO(imgdata)

        # Ajouter la référence de l'image à la liste des images extraites
        xreflist.append(xref)

        # "Ouvre" l'image avec le module Image afin de pouvoir l'utiliser avec phi3 sans avoir besoin de l'enregistrer sur l'ordi
        image = Image.open(img_io)

        # Process prompt and image for model input
        inputs = processor(prompt, [image], return_tensors="pt").to('cuda')

        # Generate text response using model
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=500, do_sample=False, repetition_penalty=1.2)

        # Remove input tokens from generated response
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

        # Decode generated IDs to text
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Ajoute la description de l'image au texte si le modèle ne renvoie pas une description vide 
        if response != '':
            text += "Texte remplaçant une image: " + response + '\n\n'
        
        # Vide le cache afin dêtre sur que toute la mémoire disponible est utilisé pour la génération
        torch.cuda.empty_cache()
    
    return text
