import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from load_data import (get_dir_and_base_name, get_dataset_info, get_unique_id,
                       get_logger, clean_str)

logger = get_logger('Load_Data_Coldstart', None)

_PROCESSED_DIR = '/home/infres/belguith/PFE/processed'


def read_aspect_data_coldstart(dataset_path, cold_ratio=0.15, cold_seed=42):
    dir_path, basename = get_dir_and_base_name(dataset_path)

    train_path   = f'{dir_path}/{basename}_coldstart_train.json'
    valid_path   = f'{dir_path}/{basename}_coldstart_valid.json'
    test_path    = f'{dir_path}/{basename}_coldstart_test.json'
    info_path    = f'{dir_path}/{basename}_coldstart_dataset_info.json'
    cold_path    = f'{dir_path}/{basename}_cold_items.json'
    inter_path   = os.path.join(_PROCESSED_DIR, f'{os.path.basename(dataset_path).split("_")[0]}_interactions_reviews_coldstart.csv')

    if os.path.exists(info_path):
        train_data = pd.read_json(train_path, lines=True)
        valid_data = pd.read_json(valid_path, lines=True)
        test_data  = pd.read_json(test_path,  lines=True)
        with open(info_path) as f:
            dataset_info = json.load(f)
        return train_data, valid_data, test_data, dataset_info

    logger.info('Building coldstart split …')
    data = pd.read_json(dataset_path, lines=True)
    data = data.rename(columns={'text': 'review_text', 'sentence': 'aspect'})

    tqdm.pandas(desc='clean text')
    data['review_text'] = data['review_text'].progress_map(lambda x: clean_str(x))

    data['review_length'] = data['review_text'].map(lambda x: len(x.split()))
    review_length = int(data['review_length'].quantile(0.8))
    data['review_text'] = data['review_text'].map(
        lambda x: ' '.join(x.split()[:review_length]))
    data['aspect'] = data['aspect'].apply(lambda x: x if isinstance(x, list) else [])
    data = data.loc[:, ['user', 'item', 'rating', 'review_text', 'aspect']]
    data['review_text'] = data['review_text'].map(
        lambda x: '<PAD>' if len(x.strip()) == 0 else x)

    # ── ID mapping (reviews_item2id si dispo) ────────────────────────────────
    _cat = os.path.basename(dataset_path).split('_')[0]
    _item2id_path = os.path.join(dir_path, f'{_cat}_reviews_item2id.json')
    if not os.path.exists(_item2id_path):
        _item2id_path = os.path.join(dir_path, f'{_cat}_item2id.json')
    _user2id_path = os.path.join(dir_path, f'{_cat}_user2id.json')

    with open(_item2id_path) as f:
        _item2id = json.load(f)
    with open(_user2id_path) as f:
        _user2id = json.load(f)

    data['item_id'] = data['item'].map(_item2id)
    data['user_id'] = data['user'].map(_user2id)
    data = data.dropna(subset=['item_id', 'user_id'])
    data['item_id'] = data['item_id'].astype(int)
    data['user_id'] = data['user_id'].astype(int)
    logger.info(f'{data["user_id"].nunique()} users, {data["item_id"].nunique()} items')

    # ── Sélection items cold ─────────────────────────────────────────────────
    rng = np.random.default_rng(cold_seed)
    all_items = np.array(sorted(data['item_id'].unique()))
    n_cold    = int(len(all_items) * cold_ratio)
    cold_set  = set(rng.choice(all_items, n_cold, replace=False).tolist())
    warm_set  = set(all_items.tolist()) - cold_set
    logger.info(f'cold_ratio={cold_ratio}  cold={n_cold}  warm={len(warm_set)}')

    # ── Split warm 80/10/10 ──────────────────────────────────────────────────
    warm_df = data[data['item_id'].isin(warm_set)].sample(
        frac=1.0, random_state=1228).reset_index(drop=True)
    n_w       = len(warm_df)
    n_w_val   = int(0.1 * n_w)
    warm_val  = warm_df[:n_w_val]
    warm_test = warm_df[n_w_val:n_w_val * 2]
    warm_train = warm_df[n_w_val * 2:]

    # ── Split cold 50/50 valid/test ──────────────────────────────────────────
    cold_df   = data[data['item_id'].isin(cold_set)].sample(
        frac=1.0, random_state=1228).reset_index(drop=True)
    n_c       = len(cold_df)
    cold_val  = cold_df[:n_c // 2]
    cold_test = cold_df[n_c // 2:]

    train_data = warm_train.copy()
    valid_data = pd.concat([warm_val,  cold_val ]).reset_index(drop=True)
    test_data  = pd.concat([warm_test, cold_test]).reset_index(drop=True)

    # ── Purge ────────────────────────────────────────────────────────────────
    # Règle : warm items absents du train → remonter dans train (comme avant)
    #         users absents du train → remonter leurs interactions warm dans
    #         train; leurs interactions cold sont simplement retirées de l'eval
    train_item_set = set(train_data['item_id'].tolist())
    train_user_set = set(train_data['user_id'].tolist())

    def _purge(split_df):
        warm_item_missing = (split_df['item_id'].isin(warm_set) &
                             ~split_df['item_id'].isin(train_item_set))
        user_missing      = ~split_df['user_id'].isin(train_user_set)
        # Move to train: warm items missing from train, OR warm interactions of missing users.
        # Cold interactions of missing users are NOT moved to train (never add cold items to train);
        # they will be dropped by the second user filter below.
        to_train_mask = warm_item_missing | (user_missing & split_df['item_id'].isin(warm_set))
        return split_df[to_train_mask], split_df[~to_train_mask]

    to_train_v, valid_data = _purge(valid_data)
    to_train_t, test_data  = _purge(test_data)
    train_data = pd.concat([train_data, to_train_v, to_train_t]).reset_index(drop=True)

    # Après purge, re-check users sans train → retirer cold de l'eval
    train_user_set = set(train_data['user_id'].tolist())
    valid_data = valid_data[valid_data['user_id'].isin(train_user_set)].reset_index(drop=True)
    test_data  = test_data[test_data['user_id'].isin(train_user_set)].reset_index(drop=True)

    logger.info(f'train={len(train_data)}  valid={len(valid_data)}  test={len(test_data)}')

    # ── Cold items dans valid/test (stats) ───────────────────────────────────
    cold_test_items  = set(test_data['item_id'].tolist())  & cold_set
    cold_valid_items = set(valid_data['item_id'].tolist()) & cold_set
    logger.info(f'cold items in valid={len(cold_valid_items)}  test={len(cold_test_items)}')

    # ── Sauvegarde cold_items.json ────────────────────────────────────────────
    with open(cold_path, 'w') as f:
        json.dump(sorted(list(cold_set)), f)
    logger.info(f'cold_items saved → {cold_path}')

    # ── interactions_reviews_coldstart.csv (degrés train uniquement) ─────────
    inter_df = train_data[['user_id', 'item_id']].copy()
    inter_df.columns = ['uid', 'iid']
    inter_df.to_csv(inter_path, index=False)
    logger.info(f'interactions_reviews_coldstart.csv saved → {inter_path}')

    # ── Sauvegarde splits ────────────────────────────────────────────────────
    os.makedirs(dir_path, exist_ok=True)
    train_data.to_json(train_path, orient='records', lines=True, force_ascii=False)
    valid_data.to_json(valid_path, orient='records', lines=True, force_ascii=False)
    test_data.to_json(test_path,   orient='records', lines=True, force_ascii=False)

    dataset_info = get_dataset_info(train_data, valid_data, test_data)
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    return train_data, valid_data, test_data, dataset_info


def load_aspect_data_coldstart(dataset_path):
    train_data, valid_data, test_data, dataset_info = \
        read_aspect_data_coldstart(dataset_path)
    train_data = train_data.loc[:, ['user_id', 'item_id', 'review_text',
                                    'rating', 'user', 'item', 'aspect']]
    valid_data = valid_data.loc[:, ['user_id', 'item_id', 'review_text',
                                    'rating', 'user', 'item', 'aspect']]
    test_data  = test_data.loc[:,  ['user_id', 'item_id', 'review_text',
                                    'rating', 'user', 'item', 'aspect']]
    return train_data, valid_data, test_data, None, None, dataset_info
