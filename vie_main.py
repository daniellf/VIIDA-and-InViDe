import vie_comp

import torch
from PIL import Image as PIL_Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_image(image, image_size, device, dataset_path, file_format):

    img = dataset_path + str(image) + file_format  
    
    img = PIL_Image.open(img)
    raw_image = img.convert("RGB") 

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def vqa_captions(img, question, model, threshold_yes, device, dataset_path, file_format):

    captions = []
    image = load_image(img, 480, device, dataset_path, file_format)

    with torch.no_grad():
        question = question + "?"
        answer, confidence = model(image, question, train=False, inference='generate')

        if answer[0] == "yes":
          if confidence[0].cpu().numpy() >= threshold_yes: return True

    return False 

def vqa_basic(img, question, model, device, dataset_path, file_format):

    answers = []
    image = load_image(img, 480, device, dataset_path, file_format)

    with torch.no_grad():
        for i, q in enumerate(question):

          if i == 5 or i == 6:
            if answers[4] == 'yes':
              answers.append('None')
              continue

          elif i == 8:
            if answers[7] == 'no':
              answers.append('None')
              continue

          elif i == 10 or i == 11:
            if answers[9] == 'no':
              answers.append('None')
              continue

          elif i == 14 or i == 15:
            if answers[13] == 'no':
              answers.append('None')
              continue    
            elif i == 15: q = "%s colors?" % answers[14]

          elif i == 18: q = "does the %s have more than one color?" % answers[17]

          elif i == 19:
            if answers[18] == 'yes': word = 'colors are'
            else: word = 'color is'
            
            q = "what %s the %s?" % (word, answers[17])
            
          answer, confidence = model(image, q+"?", train=False, inference='generate')

          if i == 1:
            is_int = represents_int(answer[0])
            
            if is_int: answers.append(answer[0])  
            else:
              if answer[0] == "middle-aged" or answer[0] == "middle aged":
                age = 55
                answers.append(age)
                
              else:
                age = 33
                answers.append(age)
                
          else:
            answers.append(answer[0])
    return answers

def represents_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def age_group(apparent_age):

  if apparent_age < 10:
    group = 'child'
  elif apparent_age < 20:
    group = 'young'
  elif apparent_age < 45:
    group = 'adult'
  elif apparent_age < 65:
    group = 'midlife'
  else:
    group = 'elderly'

  return group

def check_is_vowel(text, phoney, next_word=False):

    if text == "":
      text = next_word

    text = text.strip()
    text = phoney.phonize(text).lower()
    first_char = text.split()

    vowel = ["a", "e", "i", "o", "u"]
    
    if first_char[0][0] in vowel: return 'an'
    else: return 'a'

def is_singular(word, inflect, phoney):

  if inflect.singular_noun(word) == False:
    res = check_is_vowel(word, phoney)
    return res
    
  else: return ''

def join_separate(answer):

  new =  ''
  
  if len(answer.split()) > 1:
    for i, words in enumerate(answer.split()):
      if i < len(answer.split())-1: new += words + "_"
      else: new += words
      
    answer = new
    
  return answer
  
def generate_dense_captions_vqa(answers, phoney, inflect):

  subject = answers[0]

  age = age_group(int(answers[1]))
  
  if age == 'adult': age = ''
  elif age == 'child':
    if subject == 'man': subject = 'boy'
    else: subject = 'girl'
    
    age = ''

  place = answers[20]
  
  if place == 'home': at_in = 'at'
  else: at_in = 'in the'

  place = join_separate(place)

  objects = answers[21]
  objects = join_separate(objects)

  dense_captions = [
  "%s %s %s %s with %s expression." % (check_is_vowel(age, 
                                                      phoney, 
                                                      answers[2]).capitalize(), 
                                                      age, 
                                                      answers[2], 
                                                      subject, 
                                                      answers[16]),
  "The %s is wearing %s %s %s." % (subject, 
                                   check_is_vowel(answers[19], phoney), 
                                   answers[19], 
                                   answers[17]),
  "The %s has %s eyes." % (subject, answers[3]),
  "The %s is bald." % (subject),
  "The %s has %s hair." % (subject, answers[6]),
  "The %s has %s hair." % (subject, answers[5]),
  "The %s has %s mustache." % (subject, answers[8]),
  "The %s has %s beard." % (subject, answers[10]),
  "The %s has %s beard." % (subject, vie_comp.lemmatize_subject(answers[11])),
  "The %s is wearing makeup." % (subject),
  "The %s is wearing %s %s." % (subject, answers[15], answers[14]),
  "The %s is %s %s, with %s %s in the background." % (subject, 
                                                             at_in, 
                                                             place, 
                                                             is_singular(answers[21], inflect, phoney), 
                                                             answers[21])]

  sentences = []
  sentences_not_fisio = []
  
  is_bald = False
  use_acessory = False

  for qi, caption in enumerate(dense_captions):
    if 0 <= qi < 10:
      if qi == 1: sentences_not_fisio.append(caption)
      elif qi == 2:
        if answers[22] == "yes": sentences.append(caption)
      elif qi == 3:
        if answers[4] == "yes":
          is_bald = True
          sentences.append(caption)
      elif qi == 4:
        if is_bald == False: sentences.append(caption)
      elif qi == 5:
        if is_bald == False: sentences.append(caption)
      elif qi == 6:
        if answers[7] == "yes": sentences.append(caption)
      elif qi == 7:
        if answers[9] == "yes": sentences.append(caption)
      elif qi == 8:
        if answers[9] == "yes": sentences.append(caption)
      elif qi == 9:
        if answers[12] == "yes": sentences_not_fisio.append(caption)
      else:
        sentences.append(caption)
        
    else:
      if qi > 9:
        if qi == 10:
          if answers[13] == "yes": sentences_not_fisio.append(caption)
        else: sentences_not_fisio.append(caption)
      else: sentences.append(caption)

  return sentences, sentences_not_fisio, subject  
