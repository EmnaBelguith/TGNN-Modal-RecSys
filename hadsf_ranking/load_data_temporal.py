# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from load_data import (get_dir_and_base_name, get_dataset_info,
                       get_logger, clean_str)

logger = get_logger('Load_Data_Temporal', None)

_PROCESSED_DIR = '/home/infres/belguith/PFE/processed'

# T = p70 : ~15% cold items pour Musical, ~24% pour Baby
_TEMPORAL_PERCENTILE = 70


def read_aspect_data_temporal(dataset_path, temporal_pct=_TEMPORAL_PERCENTILE, seed=42):
    """
    Split temporel :
      - TRAIN  : toutes les interactions AVANT T (p70 des timestamps)
      - POST-T : interactions APRÈS T → shuffle → 50% valid / 50% test
      - cold   : items apparaissant UNIQUEMENT après T  (degree=0 en train)
      - popular: items avec au moins 1 interaction avant T (degree>0 en train)
      - Users sans interaction train → exclus de l'éval (pas d'embedding)
    """
    dir_path, basename = get_dir_and_base_name(dataset_path)
    _cat = os.path.basename(dataset_path).split('_')[0]

    train_path = f'{dir_path}/{basename}_temporal_train.json'
    valid_path = f'{dir_path}/{basename}_temporal_valid.json'
    test_path  = f'{dir_path}/{basename}_temporal_test.json'
    info_path  = f'{dir_path}/{basename}_temporal_dataset_info.json'
    cold_path  = f'{dir_path}/{basename}_temporal_cold_items.json'
    inter_path = os.path.join(_PROCESSED_DIR,
                              f'{_cat}_interactions_reviews_temporal.csv')

    if os.path.exists(info_path):
        train_data = pd.read_json(train_path, lines=True)
        valid_data = pd.read_json(valid_path, lines=True)
        test_data  = pd.read_json(test_path,  lines=True)
        with open(info_path) as f:
            dataset_info = json.load(f)
        return train_data, valid_data, test_data, dataset_info

    logger.info('Building temporal split …')
    data = pd.read_json(dataset_path, lines=True, convert_dates=False)
    data = data.rename(columns={'text': 'review_text', 'sentence': 'aspect'})

    tqdm.pandas(desc='clean text')
    data['review_text'] = data['review_text'].progress_map(lambda x: clean_str(x))
    data['review_length'] = data['review_text'].map(lambda x: len(x.split()))
    review_length = int(data['review_length'].quantile(0.8))
    data['review_text'] = data['review_text'].map(
        lambda x: ' '.join(x.split()[:review_length]))
    data['aspect'] = data['aspect'].apply(
        lambda x: x if isinstance(x, list) else [])
    data = data.loc[:, ['user', 'item', 'rating', 'review_text', 'aspect', 'datetime']]
    data['review_text'] = data['review_text'].map(
        lambda x: '<PAD>' if len(x.strip()) == 0 else x)

    # ── ID mapping ────────────────────────────────────────────────────────────
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

    # ── Coupure temporelle T ──────────────────────────────────────────────────
    # datetime en ms → secondes
    data['ts'] = data['datetime'] / 1000.0
    T = float(np.percentile(data['ts'].values, temporal_pct))
    import datetime as dt
    logger.info(f'T = p{temporal_pct} = {dt.datetime.fromtimestamp(T).date()}')

    # ── Définition cold / popular ─────────────────────────────────────────────
    items_before_T = set(data.loc[data['ts'] < T, 'item_id'].tolist())
    all_items      = set(data['item_id'].tolist())
    cold_set       = all_items - items_before_T   # uniquement après T
    popular_set    = items_before_T

    n_cold    = len(cold_set)
    n_popular = len(popular_set)
    logger.info(f'cold={n_cold} ({100*n_cold/len(all_items):.1f}%)  '
                f'popular={n_popular} ({100*n_popular/len(all_items):.1f}%)')

    # ── Split : train = avant T, post-T → 50/50 val/test ─────────────────────
    train_data = data[data['ts'] < T].copy()
    post_T     = data[data['ts'] >= T].sample(
        frac=1.0, random_state=seed).reset_index(drop=True)

    n_post = len(post_T)
    valid_data = post_T.iloc[:n_post // 2].copy()
    test_data  = post_T.iloc[n_post // 2:].copy()

    logger.info(f'Before purge — train={len(train_data)}  '
                f'valid={len(valid_data)}  test={len(test_data)}')

    # ── Purge ─────────────────────────────────────────────────────────────────
    # cold items → JAMAIS déplacés vers train (leur absence est intentionnelle)
    # popular items absents du train → remonter dans train
    # users absents du train → exclure leurs interactions post-T de l'éval
    train_item_set = set(train_data['item_id'].tolist())
    train_user_set = set(train_data['user_id'].tolist())

    def _purge(split_df):
        popular_missing = (split_df['item_id'].isin(popular_set) &
                           ~split_df['item_id'].isin(train_item_set))
        user_missing    = ~split_df['user_id'].isin(train_user_set)
        # Remonter en train : popular items manquants OU interactions popular de users manquants
        # Cold interactions de users manquants → retirées de l'éval (pas d'embedding user)
        to_train_mask = (popular_missing |
                         (user_missing & split_df['item_id'].isin(popular_set)))
        return split_df[to_train_mask], split_df[~to_train_mask]

    to_train_v, valid_data = _purge(valid_data)
    to_train_t, test_data  = _purge(test_data)
    train_data = pd.concat(
        [train_data, to_train_v, to_train_t]).reset_index(drop=True)

    # Re-check users sans train après purge
    train_user_set = set(train_data['user_id'].tolist())
    valid_data = valid_data[valid_data['user_id'].isin(
        train_user_set)].reset_index(drop=True)
    test_data  = test_data[test_data['user_id'].isin(
        train_user_set)].reset_index(drop=True)

    logger.info(f'After purge  — train={len(train_data)}  '
                f'valid={len(valid_data)}  test={len(test_data)}')

    cold_test_items  = set(test_data['item_id'].tolist())  & cold_set
    cold_valid_items = set(valid_data['item_id'].tolist()) & cold_set
    logger.info(f'cold items in valid={len(cold_valid_items)}  '
                f'test={len(cold_test_items)}')

    # ── Sauvegarde cold_items.json ────────────────────────────────────────────
    with open(cold_path, 'w') as f:
        json.dump(sorted(list(cold_set)), f)
    logger.info(f'cold_items saved → {cold_path}')

    # ── interactions_temporal.csv (degrés train uniquement) ───────────────────
    inter_df = train_data[['user_id', 'item_id']].copy()
    inter_df.columns = ['uid', 'iid']
    inter_df.to_csv(inter_path, index=False)
    logger.info(f'interactions_reviews_temporal.csv saved → {inter_path}')

    # ── Sauvegarde splits (sans colonne ts/datetime) ──────────────────────────
    for df in [train_data, valid_data, test_data]:
        for col in ['ts', 'datetime']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    os.makedirs(dir_path, exist_ok=True)
    train_data.to_json(train_path, orient='records', lines=True, force_ascii=False)
    valid_data.to_json(valid_path, orient='records', lines=True, force_ascii=False)
    test_data.to_json(test_path,   orient='records', lines=True, force_ascii=False)

    dataset_info = get_dataset_info(train_data, valid_data, test_data)
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    return train_data, valid_data, test_data, dataset_info


def load_aspect_data_temporal(dataset_path):
    train_data, valid_data, test_data, dataset_info = \
        read_aspect_data_temporal(dataset_path)
    cols = ['user_id', 'item_id', 'review_text', 'rating', 'user', 'item', 'aspect']
    train_data = train_data.loc[:, cols]
    valid_data = valid_data.loc[:, cols]
    test_data  = test_data.loc[:,  cols]
    return train_data, valid_data, test_data, None, None, dataset_info
