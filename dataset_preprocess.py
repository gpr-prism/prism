import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def parse_line(line):
    line = line.strip()
    if not line:
        return None
    
    try:
        return json.loads(line)
    except json.JSONDecodeError as e1:
        try:
            line_fixed = re.sub(r"(?<!\\)'", '"', line)
            line_fixed = re.sub(r'(\w+):', r'"\1":', line_fixed)
            line_fixed = re.sub(r',\s*}', '}', line_fixed)
            line_fixed = re.sub(r',\s*]', ']', line_fixed)
            line_fixed = line_fixed.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            
            return json.loads(line_fixed)
        except json.JSONDecodeError as e2:
            try:
                import ast
                if line.startswith('{') and line.endswith('}'):
                    return ast.literal_eval(line)
                else:
                    print(f"failed line: {line[:100]}...")
                    return None
            except Exception as e3:
                print(f"all method failed: {e3}")
                print(f"failed line: {line[:200]}")
                return None

def process_data(metadata_file, reviews_file, output_prefix):
    """
    
    Args:
        metadata_file: metadata datapath
        reviews_file: review datapath
        output_prefix: prefix
    """
    
    os.makedirs(output_prefix, exist_ok=True)
    
    # 1. read metadata
    items_metadata = {}
    asin_set = set()
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = parse_line(line)
            if item is None:
                continue
            
            asin = item.get('asin', '')
            if asin:
                items_metadata[asin] = {
                    'title': item.get('title', ''),
                    'categories': item.get('categories', []),
                    'description': item.get('description', ''),
                    'salesRank': item.get('salesRank', 'Not available'),
                    'imUrl': item.get('imUrl', '')
                }
                asin_set.add(asin)
    
    print(f"Loaded {len(items_metadata)} items from metadata")
    
    # 2. rank all reviews
    reviews = []
    
    with open(reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = parse_line(line)
            if review is not None:
                reviews.append(review)

    reviews.sort(key=lambda x: x.get('unixReviewTime', 0))
    print(f"Loaded {len(reviews)} reviews")
    
    # 3. collect all users and items
    all_reviewer_ids = set()
    all_reviewed_asins = set()
    
    for review in reviews:
        all_reviewer_ids.add(review.get('reviewerID'))
        all_reviewed_asins.add(review.get('asin'))
    
    # 4. create map
    user_id_map = {}
    for idx, reviewer_id in enumerate(sorted(all_reviewer_ids)):
        user_id_map[reviewer_id] = idx+1
    
    max_user_id = len(user_id_map) - 1
    
    item_id_map = {}
    for idx, asin in enumerate(sorted(all_reviewed_asins)):
        item_id_map[asin] = max_user_id + 1 + idx
    
    # 5. create edge_list
    edges = []
    min_time = None
    
    for review in reviews:
        ts = review.get('unixReviewTime', 0)
        if min_time is None or ts < min_time:
            min_time = ts
    
    # create edges
    for idx, review in enumerate(reviews):
        reviewer_id = review.get('reviewerID')
        asin = review.get('asin')
        unix_time = review.get('unixReviewTime', 0)
        rating = int(review.get('overall', 0))
        
        edge = {
            'u': user_id_map[reviewer_id],
            'r': idx + 1,  # edge ID is 1-based
            'i': item_id_map[asin],
            'ts': (unix_time - min_time)//100,
            'label': rating,
            'idx': idx + 1
        }
        edges.append(edge)
    
    # 8. save edge_list.csv
    edge_df = pd.DataFrame(edges)
    edge_df.insert(0, '', range(len(edge_df))) 
    edge_df.to_csv(f'{output_prefix}/edge_list.csv', index=False, header=True)
    print(f"Saved edge_list to {output_prefix}/edge_list.csv")
    
    # 6. entity_text.csv
    entity_texts = []
    
    entity_texts.append({
        'id': 0,
        'entity_id': 0,
        'text': ''
    })
    
    def clean_text(text):
        if not text:
            return ''
        text = str(text)
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', '')
        text = text.replace("'", '')
        return text.strip()

    user_names = {}
    for review in reviews:
        reviewer_id = review.get('reviewerID')
        reviewer_name = review.get('reviewerName', 'Unknown')
        if reviewer_id not in user_names:
            user_names[reviewer_id] = clean_text(reviewer_name)
    
    for reviewer_id, user_idx in user_id_map.items():
        name = user_names.get(reviewer_id, 'Unknown')
        entity_texts.append({
            'id': user_idx + 1,  
            'entity_id': user_idx + 1, 
            'text': name
        })

    for asin, item_idx in item_id_map.items():
        if asin in items_metadata:
            metadata = items_metadata[asin]
            
            title = metadata.get('title', 'No Title')
            
            categories_str = ''
            if metadata.get('categories'):
                categories = metadata['categories'][0] if metadata['categories'] else []
                categories_str = ', '.join(categories)
            
            rank = metadata.get('salesRank', 'Not available')
            description = metadata.get('description', 'No description')
            
            title = clean_text(title)
            categories_str = clean_text(categories_str)
            description = clean_text(description)
            

            item_text = f"Title: {title}. Category: {categories_str}. Rank: {rank}. Description: {description}"
            
            entity_texts.append({
                'id': item_idx + 1,  
                'entity_id': item_idx + 1, 
                'text': item_text
            })
        else:

            entity_texts.append({
                'id': item_idx + 1, 
                'entity_id': item_idx + 1, 
                'text': f"Title: Unknown. Category: Unknown. Description: No description available for ASIN: {asin}"
            })
    
    # save entity_text.csv
    entity_df = pd.DataFrame(entity_texts)
    entity_df = entity_df.sort_values('id')
    entity_df.to_csv(f'{output_prefix}/entity_text.csv', index=False)
    print(f"Saved entity_text to {output_prefix}/entity_text.csv")
    
    # 7. relation_text.csv
    relation_texts = []
    
    relation_texts.append({
        'i': 0,
        'text': ''
    })
    
    for edge in edges:
        review_idx = edge['idx'] - 1
        if review_idx < len(reviews):
            review = reviews[review_idx]
            review_text = review.get('reviewText', '')
            summary = review.get('summary', '')
            
            def clean_relation_text(text):
                if not text:
                    return ''
                text = str(text)
                text = text.replace('\n', ' ')
                text = text.replace('\r', ' ')
                text = re.sub(r'\s+', ' ', text)
                text = text.replace('"', '')
                text = text.replace("'", '')
                return text.strip()
            
            review_text = clean_relation_text(review_text)
            summary = clean_relation_text(summary)
            
            full_text = f"{summary}. {review_text}"
            
            relation_texts.append({
                'i': edge['r'],  # link ID
                'text': full_text.replace('"', '')
            })
        else:
            relation_texts.append({
                'i': edge['r'],
                'text': ''
            })
    
    # save relation_text.csv
    relation_df = pd.DataFrame(relation_texts)
    relation_df = relation_df.sort_values('i')
    relation_df.to_csv(f'{output_prefix}/relation_text.csv', index=False)
    print(f"Saved relation_text to {output_prefix}/relation_text.csv")
    
    # save statistics
    stats = {
        'num Users': len(user_id_map),
        'num Items': len(item_id_map),
        'num links': len(edges),
        'User IDs': f"1-{len(user_id_map)}",  
        'Item IDs': f"{max_user_id + 2}-{max_user_id + 1 + len(item_id_map)}" 
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(f'{output_prefix}/dataset_stats.csv', index=False)
    print(f"Saved statistics to {output_prefix}/dataset_stats.csv")
    
    # print statistics
    print("\n statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    metadata_file = "./Amazon_books/meta_Books.json"  # meta data
    reviews_file = "./Amazon_books/Books_5.json"    # comment data
    output_prefix = "./Amazon_books"      # prefix

    process_data(metadata_file, reviews_file, output_prefix)
