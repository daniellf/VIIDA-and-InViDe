import vie_main
import vie_comp
import spp

import json

def create_sub_mask_annotation(idx, cap):

    annotation = {
        'image_id': idx,
        'paragraph': cap
    }
    return annotation

def saved_json(image_id_list, paragraphs_list, annotations_list):

    for ci, i in enumerate(image_id_list):
        for cj, j in enumerate(paragraphs_list):
            if ci == cj:
                annotation = create_sub_mask_annotation(i, j)
                annotations_list.append(annotation)

    with open("./paragraphs.json", 'w') as f:
        json.dump(annotations_list, f)

def main(imgs, questions, threshold_similar, threshold_yes, dataset_path, device, nlp, parser, phoney, inflect, 
         tokenizer_rob, model_rob, model_vqa, annotations_list, dense_model, dense, dense_captioning=True):

    image_id_list, results_approach = [], []

    for count, id_img in enumerate(imgs):
        
        regions = []

        file_format = id_img[-4:]
        image = id_img[:-4]
        
        if dense_captioning:
            if dense_model == 'densecap':
                for j, i in enumerate(dense['results']):
                    id_out = i['img_name'][:-4]

                    if image == id_out:
                        scores = dense['results'][j]['scores']
                        elements = len(list(filter(lambda x: x >=0, scores)))
                        denses = dense['results'][j]['captions']

                        for k, description in enumerate(denses):
                            if k >= elements and len(regions) >= 5: break
                            else: regions.append(description)
                            
                        break
                        
            elif dense_model == 'grit':
                for j, i in enumerate(dense):
                    id_out = i['image_id'][:-4]
                    
                    if image == id_out:
                        scores = i['scores']
                        elements = len(list(filter(lambda x: x >=0.5, scores)))
                        denses = i['captions']

                        for k, description in enumerate(denses):
                            if k >= elements and len(regions) >= 5: break
                            else: regions.append(description)
                        
                        break
                    
            else: #none
                ...

        desc = [] 
        
        answers = vie_main.vqa_basic(image, questions, model_vqa, device, dataset_path, file_format)
        answers_vqa, answers_vqa_not_fisio, subject = vie_main.generate_dense_captions_vqa(answers, phoney, inflect)
        answers_vqa_all = answers_vqa + answers_vqa_not_fisio
        
        if dense_captioning:
            regions = vie_comp.check_densecap_similarity(regions, nlp, threshold_similar, tokenizer_rob, model_rob, stop_word=True)
            dc1, dc2 = vie_comp.check_similarity2(regions, answers_vqa_all, subject, nlp, threshold_similar, threshold_yes, image,  model_vqa, device, dataset_path, file_format, stop_word=True)
            all_caps = answers_vqa_all + dc1 + dc2
            
        else: all_caps = answers_vqa_all
       
        rel_list, rel_adj_head, freq_head, freq_rels, adjs_dict = spp.extraction_relations(all_caps, parser, nlp)  
        results = spp.paragraph_generator(all_caps, parser, nlp, rel_list, rel_adj_head, freq_head, freq_rels, adjs_dict)

        print("\nParagraph:", results)
        print("----------------------------------------------------------------------------------------------------------------------------------\n")

        results_approach.append(results)
        image_id_list.append(id_img)

        if count % 1 == 0:
            saved_json(image_id_list, results_approach, annotations_list)
        break
