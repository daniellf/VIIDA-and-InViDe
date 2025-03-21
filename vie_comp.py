import vie_main

import os
import re
import json
import math
import nltk
import spacy
import torch
import logging
import inflect 
import warnings
import sng_parser 

import numpy as np
import tensorflow as tf

import torch.nn as nn
import torchvision.models as models_trch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TVTF

from string import punctuation
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm
from string import punctuation
from PIL import Image as PIL_Image
from statistics import median, mean
from collections import Counter, defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from BLIP.models.blip_vqa import blip_vqa
from BLIP.models.blip import blip_decoder
from big_phoney.big_phoney import BigPhoney

from torch.nn import functional as F
from torch.autograd import Variable as V

from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import read_image, ImageReadMode
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import convert_image_dtype, InterpolationMode

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

punctuation = punctuation.replace("'","")

try:
  nltk.data.find('corpora/wordnet.zip')
except LookupError:
  nltk.download('wordnet')

punkt_path = os.path.join(nltk.data.find('tokenizers').path, 'punkt')
if not os.path.exists(punkt_path):
    nltk.download('punkt')
 
try:
  nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
  nltk.download('averaged_perceptron_tagger')
    
try:
  nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
  nltk.download('omw-1.4')

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def is_body_part(word, nlp): 

  part = wordnet.synsets('body_part')[0]

  for j,ss in enumerate(wordnet.synsets(word)):
    if j < 5:
      name = ss.name().split(".", 1)[0]
      if name != word:
          continue

      hit = part.lowest_common_hypernyms(ss)

      if hit and hit[0] == part:
          doc = nlp(name)
          if doc[0].pos_== 'ADJ': return False
          else: return True

  return False

def sentences_linguistically_acceptable_classify(captions, tokenizer_rob, model_rob):
  class_id = []
  probs = []

  for txt in captions:
    inputs = tokenizer_rob(txt, return_tensors="pt")

    with torch.no_grad():
        logits = model_rob(**inputs).logits

    predicted_class_id = logits.argmax().item()
    class_id.append(predicted_class_id)
    predictions = tf.nn.softmax(logits)
    max_prob = max(tf.keras.backend.get_value(predictions)[0])
    probs.append(max_prob)

  return class_id, probs

def check_densecap_similarity(regions, nlp, threshold_similar, tokenizer_rob, model_rob, stop_word):

  check_sim = []

  for region in regions:
    description = region.lower()
    description = description.strip(" ")
    description = ''.join(c for c in description if c not in punctuation)
    description = ' '.join(re.split("\s+", description, flags=re.UNICODE))
    parsedData = nlp(description)

    if stop_word:
      parsedData = nlp(' '.join([str(t) for t in parsedData if not t.is_stop]))

    if not check_sim:
      if len(parsedData) <= 2:
        short = False
        for word in parsedData:
          anwser = is_body_part(lemmatize_subject(str(word)), nlp)
          if anwser:
            short = True
            break
              
        if short != True:
          check_sim.append(description.capitalize() + '.')

      else:
        check_sim.append(description.capitalize() + '.')

    else:
      add = True
      for d in check_sim:
        cap = d.lower()
        cap = cap.strip(" ")
        cap = ''.join(c for c in cap if c not in punctuation)
        final_cap = nlp(cap)

        if stop_word:
          final_cap = nlp(' '.join([str(t) for t in final_cap if not t.is_stop]))

        similarity = round(parsedData.similarity(nlp(final_cap)), 2)
        if similarity >= threshold_similar:
          add = False
          if similarity < 1.0:
            captions_compare = [cap, description]
            classes_ids = []
            probabilities = []
            new = []
            classes_ids, probabilities = sentences_linguistically_acceptable_classify(captions_compare, tokenizer_rob, model_rob)
            add = False

            if classes_ids[0] == 0 and classes_ids[1] == 1:
              check_sim.remove(d)
              check_sim.append(description.capitalize() + '.')

            elif classes_ids[0] == 0 and classes_ids[1] == 0:
              if probabilities[0] > probabilities[1]:
                check_sim.remove(d)
                check_sim.append(description.capitalize() + '.')

            elif classes_ids[0] == 1 and classes_ids[1] == 1:
              if probabilities[0] < probabilities[1]:
                check_sim.remove(d)
                check_sim.append(description.capitalize() + '.')

          break

      if add:
        if len(parsedData) <= 2:
          short = False
          for word in parsedData:
            anwser =  is_body_part(lemmatize_subject(str(word)), nlp)
            if anwser:
              short = True
              break
                
          if short != True:
            check_sim.append(description.capitalize() + '.')

        else:
          check_sim.append(description.capitalize() + '.')

  return check_sim


def check_similarity2(regions, captions_vqa, subject, nlp, threshold_similar, threshold_yes, image, model_vqa, device, dataset_path, file_format, stop_word):

  caps = []
  no_person = []
  check_sim = []

  for region in regions:
    description = region.lower()
    description = description.strip(" ")
    description = ''.join(c for c in description if c not in punctuation)
    description = ' '.join(re.split("\s+", description, flags=re.UNICODE))
    parsedData = nlp(description)

    if stop_word:
      parsedData = nlp(' '.join([str(t) for t in parsedData if not t.is_stop]))

    add = True
      
    for d in captions_vqa:
      cap_vqa = d.lower()
      cap_vqa = cap_vqa.strip(" ")
      cap_vqa = ''.join(c for c in cap_vqa if c not in punctuation)
      final_cap_vqa = nlp(cap_vqa)

      if stop_word:
        final_cap_vqa = nlp(' '.join([str(t) for t in final_cap_vqa if not t.is_stop]))

      similarity = round(parsedData.similarity(nlp(final_cap_vqa)), 2)
      if similarity >= threshold_similar:
        add = False
        break

    if add:
      if not check_sim:
        reg_bool = vie_main.vqa_captions(image, description, model_vqa, threshold_yes, device, dataset_path, file_format)
        if reg_bool:
          check_sim.append(description.capitalize() + '.')
          for i, word in enumerate(parsedData):
            if i < len(parsedData)-1:
              if word.text.lower() == subject:
                caps.append(description.capitalize() + '.')
                break
                  
            else:
              if word.text.lower() == subject:
                caps.append(description.capitalize() + '.')
                break
                  
              else:
                no_person.append(description.capitalize() + '.')
                break

      else:
        add2 = True
        for d in check_sim:
          cap_vqa = d.lower()
          cap_vqa = cap_vqa.strip(" ")
          cap_vqa = ''.join(c for c in cap_vqa if c not in punctuation)
          final_cap_vqa = nlp(cap_vqa)

          if stop_word:
            final_cap_vqa = nlp(' '.join([str(t) for t in final_cap_vqa if not t.is_stop]))

          similarity = round(parsedData.similarity(nlp(final_cap_vqa)), 2)
          if similarity >= threshold_similar:
            add2 = False
            break

        if add2:
          reg_bool = vie_main.vqa_captions(image, description, model_vqa, threshold_yes, device, dataset_path, file_format)
          if reg_bool:
            check_sim.append(description.capitalize() + '.')

            for i, word in enumerate(parsedData):
              if i < len(parsedData)-1:
                if word.text.lower() == subject:
                  caps.append(description.capitalize() + '.')
                  break
                    
              else:
                if word.text.lower() == subject:
                  caps.append(description.capitalize() + '.')
                  break
                    
                else:
                  no_person.append(description.capitalize() + '.')
                  break

  return caps, no_person

def lemmatize_subject(text):
  tag_map = defaultdict(lambda : wordnet.NOUN)
  tag_map['J'] = wordnet.ADJ
  tag_map['V'] = wordnet.VERB
  tag_map['R'] = wordnet.ADV

  tokens = word_tokenize(text)
  lemma_function = WordNetLemmatizer()
  for token, tag in pos_tag(tokens):
    lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
    break

  return lemma
