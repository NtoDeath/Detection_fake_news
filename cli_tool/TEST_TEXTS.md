# Test Texts for Fake News Detection CLI

Use these texts to test the fake news detector. Copy and paste them when prompted.

## Very Short Texts (1-2 sentences)

### Short Fake 1
```
The moon landing was faked in a Hollywood studio.
```

### Short Fake 2
```
5G towers cause COVID-19.
```

### Short Fake 3
```
Drinking bleach cures cancer.
```

### Short Real 1
```
The Earth orbits around the Sun.
```

### Short Real 2
```
Water boils at 100 degrees Celsius at sea level.
```

### Short Real 3
```
Paris is the capital of France.
```

---

## Medium Texts (3-5 sentences)

### Medium Fake 1
```
Scientists have discovered that vaccines contain tracking devices. These microchips can monitor your location and thoughts through 5G networks. The government has been hiding this technology for decades to control the population. Bill Gates personally invested billions into this project to reduce global population.
```

### Medium Fake 2
```
Recent studies prove that the Earth is actually flat and governments have been lying about it. NASA photoshops all images to hide the truth. Anyone who has gone to space is part of a massive conspiracy. The Ice Wall surrounds our disc-shaped planet to prevent people from falling off.
```

### Medium Fake 3
```
Elvis Presley is still alive and living in secret bunkers. He never actually died in 1977. The government replaced him with a clone. He has been spotted multiple times in Las Vegas casinos by credible witnesses.
```

### Medium Real 1
```
The COVID-19 pandemic was caused by a novel coronavirus (SARS-CoV-2) that first emerged in Wuhan, China, in late 2019. Scientists around the world worked together to develop vaccines using mRNA technology. These vaccines have proven to be safe and effective in reducing severe illness and death. Millions of people worldwide have received vaccinations with good safety profiles.
```

### Medium Real 2
```
Climate change is primarily caused by greenhouse gases like carbon dioxide and methane released by human activities. Scientific consensus is overwhelming among climate researchers who study this phenomenon. Global temperatures have been rising consistently over the past century. Melting polar ice and rising sea levels are observable consequences of this warming trend.
```

### Medium Real 3
```
Photosynthesis is the process by which plants convert sunlight into chemical energy. Plants absorb carbon dioxide from the air and water from the soil. They use chlorophyll in their leaves to capture light energy. This process produces glucose for plant growth and releases oxygen as a byproduct.
```

---

## Long Texts (Full paragraphs)

### Long Fake 1
```
Ancient astronaut theorists believe that extraterrestrials visited Earth thousands of years ago and genetically modified humans to serve as their slaves. The evidence for this can be found in ancient texts like the Sumerian clay tablets and Egyptian hieroglyphics, which allegedly describe spacecraft and advanced technology. The Great Pyramids of Giza were built by aliens using laser technology and anti-gravity devices, not by ancient Egyptians with simple tools. Modern governments have been in contact with extraterrestrials for centuries and have reverse-engineered their technology to create everything from smartphones to aircraft. Area 51 in Nevada is the secret base where captured alien spacecraft are studied, and numerous eyewitnesses have reported seeing UFOs in the surrounding desert. NASA's space program is actually a front for a deeper space agency that has already established colonies on Mars and the Moon.
```

### Long Fake 2
```
A global elite group of billionaires and politicians are secretly running a shadow government that controls world events from the shadows. This cabal includes billionaire tech founders, royal families, and influential media moguls who meet in secret on private islands to plan economic crashes and wars. They are deliberately spreading misinformation to keep the general population confused and docile. The United Nations is their tool for establishing a "New World Order" with a single world government and cryptocurrency-based surveillance system. COVID-19 was intentionally created in a laboratory as a bioweapon to depopulate the planet and consolidate power. All mainstream media outlets knowingly spread propaganda to support this agenda, and any journalists who try to expose the truth are silenced or killed.
```

### Long Fake 3
```
Medical doctors have been trained by pharmaceutical companies to prescribe unnecessary drugs that make people sick so they can sell more treatments. Vaccines are actually designed to sterilize people and reduce population growth. The pharmaceutical industry makes trillions of dollars by deliberately keeping cures secret while promoting treatments that never cure the disease. Cancer was cured in the 1950s but the cure was suppressed because cancer treatment is too profitable. Chemotherapy and radiation are toxic treatments designed to kill patients faster while hospitals make money from extended treatments. Natural remedies like essential oils and crystals are the real cures but are banned by the FDA to protect pharmaceutical profits. Doctors who promote natural medicine are arrested and their licenses revoked by the medical establishment.
```

### Long Real 1
```
The human immune system is a complex biological defense mechanism that protects the body against pathogens like bacteria, viruses, fungi, and parasites. It consists of physical barriers such as skin and mucous membranes, as well as specialized cells like white blood cells, lymphocytes, and macrophages. When a pathogen invades the body, the innate immune system provides a rapid initial response to contain the threat. The adaptive immune system then develops a targeted response by producing antibodies and activating T cells specific to the invading pathogen. Vaccines work by introducing a weakened or inactivated form of a pathogen, allowing the immune system to develop antibodies and memory cells without causing severe illness. This creates immunity so that if exposure to the actual pathogen occurs in the future, the immune system can rapidly mount a response to prevent infection or minimize symptoms. Extensive clinical trials and ongoing monitoring have demonstrated that vaccines are safe and effective at reducing disease burden globally.
```

### Long Real 2
```
The process of photosynthesis in plants occurs primarily in the leaves, specifically in structures called chloroplasts. These organelles contain a green pigment called chlorophyll that absorbs light energy from the sun across different wavelengths. When light energy is absorbed, water molecules are split into hydrogen and oxygen through a process called photolysis. The oxygen is released as a waste product, while the hydrogen is used to power the Calvin cycle, a series of biochemical reactions that fix carbon dioxide into glucose. This glucose serves as an energy source and building block for plant growth and development. The overall reaction can be summarized as: carbon dioxide plus water, in the presence of light energy, produces glucose and oxygen. Different plants have adapted variations of this basic process, including C3 photosynthesis, C4 photosynthesis, and CAM photosynthesis, depending on their environmental conditions.
```

### Long Real 3
```
The Internet was developed in the late 1960s and 1970s as a research project funded by the United States Department of Defense's Advanced Research Projects Agency (ARPA). The early network, known as ARPANET, connected research institutions and universities to share computing resources and information. A major breakthrough came in 1974 when researchers Vint Cerf and Bob Kahn developed the TCP/IP protocol, which became the foundational communication standard for the Internet. In 1989, Tim Berners-Lee, a British physicist working at CERN in Switzerland, invented the World Wide Web and the HTTP protocol. This allowed documents to be linked together through hypertext, creating what we know today as websites and web pages. The commercialization of the Internet began in 1991, and by the 1990s, public dial-up access services like America Online brought the Internet into homes worldwide. Today, the Internet connects billions of devices globally and has become essential infrastructure for communication, commerce, education, and entertainment.
```

---

## Mixed/Ambiguous Texts

### Ambiguous 1 (Partially True)
```
Climate change is real and proven by scientists, but it's actually good for plants because they need more carbon dioxide. Some scientists say global warming will increase agricultural productivity in northern regions. We don't need to reduce emissions because the Earth's climate has changed naturally before.
```

### Ambiguous 2 (Partially True)
```
Masks can help reduce transmission of respiratory viruses during pandemics, but some people claim they cause oxygen deprivation or contain tracking devices. Studies show mask effectiveness varies by mask type and proper usage, but many reject mask recommendations based on false information about health effects.
```

### Ambiguous 3 (Partially True)
```
Artificial intelligence and automation are changing the job market, but some claim this means all jobs will disappear imminently. While job displacement is a real concern that economists study, extreme predictions about total economic collapse from AI are not supported by current evidence. Like previous technological revolutions, new jobs typically emerge even as some older positions become obsolete.
```

---

## Test Strategy

1. **Start with short texts** to see if the model catches obvious fake claims
2. **Test medium texts** to see how it handles interconnected false claims
3. **Try long texts** to test performance on detailed narratives
4. **Use mixed texts** to see how it handles partially true or ambiguous information

Record the model's predictions and confidence scores to compare performance across different text lengths and content types.
