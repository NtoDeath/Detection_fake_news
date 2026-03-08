from style_extractor import StyleExtractor

def print_results(raw_text):
    print("ANALYSE DE LA NEWS:\n")
    print(f"Texte brut : \n{raw_text}\n")
    
    extractor = StyleExtractor()
    
    clean_text = extractor._normalize_text(raw_text)
    print(f"Texte normalisé : \n{clean_text}\n")
    
    style_vector = extractor._extract_metrics(clean_text)
    
    print("Vecteur de style (FEATURES):\n")
    
    for key, value in style_vector.items():
        if isinstance(value, float):
            print(f"{key:<30} : {round(value, 3)}")
        else:
            print(f"{key:<30} : {value}")
    
    print("\n")

text_test = "ALERT! You must absolutely read this: https://www.google.com. The government is hiding 10,000 terrible and monstrous things! Wake up immediately @user! #news 😡"
# text_test = "ALERTE !!! Vous devez absolument lire ceci : https://www.google.com. Le gouvernement nous cache 10000 choses terribles et monstrueuses ! Réveillez-vous immédiatement @user ! #news 😡"

print_results(text_test)