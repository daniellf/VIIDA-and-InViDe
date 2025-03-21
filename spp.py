import spacy
import sng_parser
from collections import Counter, defaultdict

def pronoun_adjustment(paragraph, nlp):
  new_paragraph = ""

  for i, sentence in enumerate(paragraph.split(".")):

    if i == len(paragraph.split("."))-1: break
    else:

      sentence = sentence.strip()
      doc = nlp(sentence)

      for j, token in enumerate(doc):
        
        if token.pos_ == "NOUN" and (token.dep_ == "nsubj" or token.dep_ == "ROOT") and j != len(doc)-1:
          pronoun = ""
          if token.text.lower() == "man" or token.text.lower() == "boy": pronoun = 'He' 
          elif token.text.lower() == "woman" or token.text.lower() == "girl": pronoun = 'She' 
          else: continue
          
          if i > 0:
            if j-1 == 0: sentence = sentence.replace(("The %s" %  (token.text)), pronoun)
            else: sentence = sentence.replace(("%s" % token.text), pronoun)   
            break

        elif token.pos_ == "NOUN" and token.dep_ == "pobj": 
            pronoun = ""

            if doc[j-1].pos_ != 'DET':
                if token.text == "man" or token.text == "boy": pronoun = 'him' 
                elif token.text == "woman" or token.text == "girl": pronoun = 'her'
                else: pronoun = token.text

                sentence = sentence.replace((token.text), pronoun)
            break

      new_paragraph += sentence + ". "
  return new_paragraph
  
def extraction_relations(caps, parser, nlp):

  relations = []
  freq_rels = []
  adj_head  = []
  freq_head = []

  head = []

  adjs_dict = defaultdict(set)

  for x, phrase in enumerate(caps):
    
    gph = parser.parse(phrase.lower())
    #sng_parser.tprint(gph)

    if len(gph['relations']) > 0:

      doc = nlp(phrase)

      is_key = False
      is_adj = False

      for i, token in enumerate(doc):
        key = ""

        if token.pos_ == "NOUN" and (token.dep_ == 'ROOT' or token.dep_ == 'nsubj'):
          rel_in = False
          if len(gph['relations']) > 1:
            for rel in gph['relations']:
              if rel['relation'] == "in":
                key = token.text.lower() + "__" + rel['relation']
                rel_in = True
                break
                  
            if rel_in == False:
              key = token.text.lower() + "__" + gph['relations'][0]['relation']

          else:
            key = token.text.lower() + "__" + gph['relations'][0]['relation']
            
          relations.append(key)
          freq_rels.append(key)

          is_key = True

        elif token.pos_ == "ADJ":
          if token.dep_ == "ROOT": 
              continue

          elif is_adj != True:
            if token.head.text == token.text:
              previous = i-1
              next = i+1
                
              if doc[previous].pos_ == "NOUN" and doc[previous].dep_ == 'pobj':
                head_text = doc[previous].text

              elif doc[next].pos_ == "NOUN" and doc[next].dep_ == 'pobj':
                head_text = doc[next].text

              adj_head.append(head_text)
              freq_head.append(head_text)
              adjs_dict[head_text].add(token.text)

            else:
              adj_head.append(token.head.text)
              freq_head.append(token.head.text)
              adjs_dict[token.head.text].add(token.text)

            is_adj = True

      if is_key == False:
        for token in doc:
          if  token.pos_ == "NOUN" and token.dep_ == 'npadvmod':
            rel_in = False
            if len(gph['relations']) > 1:
              for rel in gph['relations']:
                if rel['relation'] == "in":
                  key = token.text.lower() + "__" + rel['relation']
                  rel_in = True
                  break
                    
              if rel_in == False:
                key = token.text.lower() + "__" + gph['relations'][0]['relation']

            else:
              key = token.text.lower() + "__" + gph['relations'][0]['relation']

            relations.append(key)
            freq_rels.append(key)

            is_key = True
            break

        if is_key != True:
          freq_rels.append(None)

      if is_adj == False:
        freq_head.append(None)

    else:
      doc = nlp(phrase)
      new_relation = False
      indice_freq_head = False
    
      is_adj = False
    
      for i, token in enumerate(doc):
        key = ""
        if token.pos_ == "NOUN" and (token.dep_ == 'nsubj' or token.dep_ == 'npadvmod'):
          for next in range(i+1, len(doc)):
            if doc[next].text == token.head.text and doc[next].pos_ == "VERB" and doc[next].dep_ == "ROOT":
              key = token.text.lower() + "__" + doc[next].text
              new_relation = True
              
            elif doc[next].pos_ == "NOUN" and doc[next].dep_ == "dobj":
              freq_head.append(doc[next].text)
              indice_freq_head = True

          if new_relation:
            break
        
        elif token.pos_ == "ADJ":
          if token.dep_ == "ROOT":
              continue

          elif is_adj != True:
            previous = i-1
            next = i+1
            
            if token.head.text == token.text: 
                if doc[previous].pos_ == "NOUN" and doc[previous].dep_ == 'pobj':
                    head_text = doc[previous].text
                    adj_noun = token.text  
                    indice_freq_head = True
                    is_adj = True

                elif doc[next].pos_ == "NOUN" and doc[next].dep_ == 'pobj':
                    head_text = doc[next].text
                    adj_noun = token.text  
                    indice_freq_head = True
                    is_adj = True

            else:
                if token.dep_ == "compound":
                    if doc[next].dep_ == "compound":
                        head_text = doc[next].text
                        adj_noun = token.text.lower()
                        indice_freq_head = True
                        is_adj = True
                else:
                    head_text = token.head.text
                    adj_noun = token.text
                    indice_freq_head = True
                    is_adj = True

      if new_relation:
        relations.append(key)
        freq_rels.append(key)
        
        if is_adj:
            adj_head.append(head_text)
            freq_head.append(head_text)
            adjs_dict[head_text].add(adj_noun)
        
        if indice_freq_head == False:
          freq_head.append(None)
                                 
      else:
        freq_rels.append(None)
        freq_head.append(None)

  rel_list = Counter()
  for sub_rel in relations:
    rel_list[sub_rel] += 1

  rel_adj_head = Counter()
  for head in adj_head:
    rel_adj_head[head] += 1

  for head in rel_adj_head:
    if rel_adj_head[head] < 2:
      rel_adj_head[head] = 0

  return rel_list, rel_adj_head, freq_head, freq_rels, adjs_dict

def paragraph_generator(caps, parser, nlp, rel_list, rel_adj_head, freq_head, freq_rels, adjs_dict):

  paragraph = ""  

  for idx, phrase in enumerate(caps):

    phrase = phrase.strip(".")
    phrase = phrase.strip("  ")
    gph = parser.parse(phrase.lower())
    
    jump = False

    if len(gph['relations']) == 0:
      head_current = freq_head[idx]

      if len(adjs_dict[head_current]) < 1 and freq_head[idx] != None:
        continue

      elif freq_head[idx] == None:
        paragraph += phrase + ". "
        continue

      else:
        paragraph += phrase + ". "
        continue

    else:
      rel_current = freq_rels[idx]

      if rel_list[rel_current] < 1:
        if paragraph[-2] != '.': paragraph += ". "
        else:continue

      elif rel_list[rel_current] == 1:

        if adjs_dict[freq_head[idx]] == None:
            paragraph += phrase + ". "
            continue
        
        else:    
            paragraph += phrase + ". "
            rel_list[rel_current] -= 1
            del adjs_dict[freq_head[idx]]
            continue

      else:
        head_current = freq_head[idx]

        if rel_adj_head[head_current] > 0:
          doc = nlp(phrase)
          paragraph += str(doc[:-2])
        
          jump = True
          s = ""
          
          for x in adjs_dict[head_current]:
                s+= x + " "
                rel_adj_head[head_current] -= 1
                rel_list[rel_current] -= 1

          paragraph += " " + s + head_current
      
          del adjs_dict[head_current]

        else:
          paragraph += phrase
          rel_list[rel_current] -= 1

        
        for i in range(idx+1, len(caps)):
        
          if rel_list[rel_current] <= 0:
            break

          if rel_current == freq_rels[i]:
            gphx = parser.parse(caps[i].lower())

            head_current = freq_head[i]

            if rel_list[rel_current] == len(adjs_dict[head_current]):
              if len(adjs_dict[head_current]) > 0:
                s = ""
                
                for x in adjs_dict[head_current]:
                    s+=x + " "
                    rel_list[rel_current] -= 1
                    rel_adj_head[head_current] -= 1

                paragraph += " and " + s + head_current + ". "

            elif len(adjs_dict[head_current]) > 0:
              if len(adjs_dict[head_current]) >= 2:
                jump = True
                s = ""
                
                for x in adjs_dict[head_current]:
                    s+=x + " "
                    rel_list[rel_current] -= 1
                    rel_adj_head[head_current] -= 1

                del adjs_dict[freq_head[i]]

                if rel_list[rel_current] == 0: 
                  paragraph += " and " + s 
                
                elif rel_list[rel_current] < 0:
                    paragraph += " and " + s + head_current  + ". " 
                    break

                else:
                  paragraph += ", " + s + head_current

              else: 
                if rel_list[rel_current]-1 == 0:
                  paragraph += ", " + gphx['entities'][1]['span'] + ". "
                else:
                  paragraph += ", " + gphx['entities'][1]['span']
                rel_list[rel_current] -= 1

              rel_adj_head[head_current] -= 1

            else:
                if jump == True:
                  jump = False
                  continue

                elif rel_list[rel_current]-1 == 0:
                  
                  if len(gphx['relations']) == 0: sentence = freq_head[i]
                  else: sentence = gphx['entities'][1]['span']
                  
                  get_indice = None
                  for ind, word in enumerate(caps[i].split()):
                      if sentence.split()[-1] == word:
                        get_indice = ind
                        break
                        
                  if get_indice != None:
                    for word in range(get_indice+1, len(caps[i].split())):
                        sentence += " " + caps[i].split()[word].strip('.')
                  else:
                    if sentence.split()[-1] != caps[i].split()[-1].strip('.'):
                        sentence += " " + caps[i].split()[-1].strip('.')
            
                  paragraph += " and " + sentence + ". "
                  rel_list[rel_current] -= 1
                  break

                else:
                    if len(gphx['relations']) == 0: sentence = freq_head[i]
                    else: sentence = gphx['entities'][1]['span']

                    paragraph += ", " + sentence
                    rel_list[rel_current] -= 1

          else:
            continue

  paragraph = " ".join(str(paragraph).split())
  paragraph = pronoun_adjustment(paragraph,  nlp)
  paragraph = paragraph.replace("_", " ")

  return paragraph
