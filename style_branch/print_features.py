from style_extractor import StyleExtractor

def print_results(raw_text):
    print("NEWS ANALYSIS:\n")
    print(f"Raw text:\n{raw_text}\n")
    
    extractor = StyleExtractor()
    
    if not raw_text or not isinstance(raw_text, str):
        print("Error: The provided text is empty or invalid.")
        return

    clean_text = extractor._normalize_text(raw_text)
    print(f"Normalized text:\n{clean_text}\n")
    
    style_vector = extractor._extract_metrics(clean_text)
    
    print("Style vector (FEATURES):\n")
    
    for key, value in style_vector.items():
        if isinstance(value, float):
            print(f"{key:<30} : {round(value, 3)}")
        else:
            print(f"{key:<30} : {value}")
    
    print("\n")

text_test = "ALERT! You must absolutely read this: https://www.google.com. The government is hiding 10,000 terrible and monstrous things! Wake up immediately @user! #news 😡"

print_results(text_test)