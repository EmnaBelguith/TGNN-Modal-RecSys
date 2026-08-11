"""
Tests de validité du pipeline coldstart Musical.

Vérifie :
  1. cold_items.json — nombre d'items cold cohérent avec cold_ratio
  2. Aucun item cold dans le CSV train (degree=0 garanti)
  3. CSV train : distribution des degrés correcte
  4. Splits train/valid/test — pas de fuite cold→train
  5. Résultats d'évaluation lus depuis les logs SLURM

Usage :
  python test_coldstart_pipeline.py
  python test_coldstart_pipeline.py --logs_dir /path/to/logs
"""

import argparse
import json
import os
import re
import sys
import pandas as pd

# ── Chemins ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = '/home/infres/belguith/PFE/processed'
LOGS_DIR_DEFAULT = '/home/infres/belguith/PFE/logs'

COLD_ITEMS_PATH   = f'{PROCESSED_DIR}/Musical_reviews_with_aspects_cold_items.json'
TRAIN_CSV_PATH    = f'{PROCESSED_DIR}/Musical_interactions_reviews_coldstart.csv'
TRAIN_JSON_PATH   = f'{PROCESSED_DIR}/Musical_reviews_with_aspects_coldstart_train.json'
VALID_JSON_PATH   = f'{PROCESSED_DIR}/Musical_reviews_with_aspects_coldstart_valid.json'
TEST_JSON_PATH    = f'{PROCESSED_DIR}/Musical_reviews_with_aspects_coldstart_test.json'
INFO_JSON_PATH    = f'{PROCESSED_DIR}/Musical_reviews_with_aspects_coldstart_dataset_info.json'
ITEM2ID_PATH      = f'{PROCESSED_DIR}/Musical_reviews_item2id.json'

N_ITEMS_TOTAL = 4886
COLD_RATIO    = 0.15
COLD_RATIO_TOL = 0.02   # ±2% acceptable


# ── Helpers ────────────────────────────────────────────────────────────────────

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
WARN = '\033[93mWARN\033[0m'
INFO = '\033[94mINFO\033[0m'

_results = []

def check(label, cond, msg='', warn=False):
    status = (WARN if warn else FAIL) if not cond else PASS
    tag = 'WARN' if (warn and not cond) else ('PASS' if cond else 'FAIL')
    _results.append((tag, label))
    suffix = f'  → {msg}' if msg else ''
    print(f'  [{status}] {label}{suffix}')
    return cond


def section(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ── Test 1 : cold_items.json ───────────────────────────────────────────────────

def test_cold_items_json():
    section('1. cold_items.json')
    if not os.path.exists(COLD_ITEMS_PATH):
        check('cold_items.json existe', False, 'fichier absent')
        return None

    with open(COLD_ITEMS_PATH) as f:
        cold_items = json.load(f)

    n = len(cold_items)
    expected_min = int(N_ITEMS_TOTAL * (COLD_RATIO - COLD_RATIO_TOL))
    expected_max = int(N_ITEMS_TOTAL * (COLD_RATIO + COLD_RATIO_TOL))
    check(
        f'Nombre cold_items ∈ [{expected_min}, {expected_max}]',
        expected_min <= n <= expected_max,
        f'trouvé {n} (attendu ≈{int(N_ITEMS_TOTAL * COLD_RATIO)})'
    )
    check(
        'IDs cold dans [0, 4885]',
        all(0 <= i < N_ITEMS_TOTAL for i in cold_items),
        f'min={min(cold_items)}, max={max(cold_items)}'
    )
    check(
        'IDs cold uniques',
        len(cold_items) == len(set(cold_items))
    )
    print(f'  [{INFO}] {n} cold items, soit {n/N_ITEMS_TOTAL:.1%} du catalogue')
    return set(cold_items)


# ── Test 2 : CSV train — aucun item cold ──────────────────────────────────────

def test_csv_no_cold(cold_set):
    section('2. CSV train — aucun item cold')
    if not os.path.exists(TRAIN_CSV_PATH):
        check('CSV exists', False, 'fichier absent')
        return None

    df = pd.read_csv(TRAIN_CSV_PATH)
    check('Colonnes uid/iid présentes', {'uid', 'iid'}.issubset(df.columns))

    if cold_set is not None:
        cold_in_train = set(df['iid'].unique()) & cold_set
        check(
            'Aucun item cold dans le CSV train',
            len(cold_in_train) == 0,
            f'{len(cold_in_train)} items cold présents dans train : {sorted(list(cold_in_train))[:10]}'
        )
    else:
        check('cold_set disponible pour la vérification', False, 'sauté (cold_set absent)')

    # Distribution des degrés
    deg = df['iid'].value_counts()
    deg_full = pd.Series(0, index=range(N_ITEMS_TOTAL))
    deg_full.update(deg)
    n_zero = (deg_full == 0).sum()
    n_pos  = (deg_full > 0).sum()
    print(f'  [{INFO}] items degree=0 : {n_zero} | degree>0 : {n_pos}')

    if cold_set is not None:
        expected_cold_deg0 = len(cold_set)
        check(
            f'Nombre d\'items degree=0 == len(cold_items) ({expected_cold_deg0})',
            n_zero == expected_cold_deg0,
            f'trouvé {n_zero}'
        )
    return deg_full


# ── Test 3 : Intégrité des splits ─────────────────────────────────────────────

def test_splits_integrity(cold_set, deg_full):
    section('3. Intégrité des splits (train / valid / test)')

    for path, name in [(TRAIN_JSON_PATH, 'train'), (VALID_JSON_PATH, 'valid'), (TEST_JSON_PATH, 'test')]:
        if not os.path.exists(path):
            check(f'{name} JSON exists', False, 'fichier absent')
            return

    train = pd.read_json(TRAIN_JSON_PATH, lines=True)
    valid = pd.read_json(VALID_JSON_PATH, lines=True)
    test  = pd.read_json(TEST_JSON_PATH,  lines=True)

    print(f'  [{INFO}] train={len(train)} | valid={len(valid)} | test={len(test)}')

    if cold_set is not None:
        cold_in_train_json = set(train['item_id'].unique()) & cold_set
        check(
            'Aucun item cold dans train.json',
            len(cold_in_train_json) == 0,
            f'{len(cold_in_train_json)} cold items trouvés'
        )
        cold_in_valid = set(valid['item_id'].unique()) & cold_set
        cold_in_test  = set(test['item_id'].unique())  & cold_set
        check(
            'Cold items présents dans valid',
            len(cold_in_valid) > 0,
            f'{len(cold_in_valid)} cold items dans valid'
        )
        check(
            'Cold items présents dans test',
            len(cold_in_test) > 0,
            f'{len(cold_in_test)} cold items dans test'
        )
        print(f'  [{INFO}] cold dans valid={len(cold_in_valid)} | test={len(cold_in_test)}')

    # Overlap item train/valid (warm items dans valid → must be in train)
    train_items = set(train['item_id'].unique())
    warm_in_valid = set(valid['item_id'].unique()) - (cold_set or set())
    check(
        'Tous les warm items de valid sont dans train',
        warm_in_valid.issubset(train_items),
        f'{len(warm_in_valid - train_items)} warm items absents du train'
    )

    # Users: tous les users de valid/test sont dans train
    train_users = set(train['user_id'].unique())
    valid_users = set(valid['user_id'].unique())
    test_users  = set(test['user_id'].unique())
    miss_v = valid_users - train_users
    miss_t = test_users  - train_users
    check('Tous les users de valid sont dans train', len(miss_v) == 0,
          f'{len(miss_v)} users absents')
    check('Tous les users de test  sont dans train', len(miss_t) == 0,
          f'{len(miss_t)} users absents')


# ── Test 4 : DIAG_SETUP dans les logs ─────────────────────────────────────────

def test_diag_setup(logs_dir, cold_set):
    section(f'4. DIAG_SETUP dans les logs ({logs_dir})')

    if not os.path.isdir(logs_dir):
        print(f'  [{WARN}] logs_dir introuvable : {logs_dir}')
        return

    coldstart_logs = [f for f in os.listdir(logs_dir)
                      if 'coldstart' in f and f.endswith('.out')]

    if not coldstart_logs:
        print(f'  [{WARN}] Aucun log *coldstart*.out dans {logs_dir}')
        return

    expected_cold = len(cold_set) if cold_set else None
    for fname in sorted(coldstart_logs):
        fpath = os.path.join(logs_dir, fname)
        diag_lines = []
        with open(fpath) as fh:
            for line in fh:
                if 'DIAG_SETUP' in line:
                    diag_lines.append(line.strip())
        if not diag_lines:
            continue

        diag = diag_lines[0]
        m = re.search(r'cold_items=(\d+).*popular_items=(\d+)', diag)
        if not m:
            print(f'  [{WARN}] {fname}: DIAG_SETUP format inattendu: {diag}')
            continue
        n_cold_log = int(m.group(1))
        n_pop_log  = int(m.group(2))

        ok = (expected_cold is None) or (n_cold_log == expected_cold)
        status = PASS if ok else FAIL
        msg = (f'cold={n_cold_log}, popular={n_pop_log}'
               + (f'  ← attendu cold={expected_cold}' if not ok else ''))
        print(f'  [{status}] {fname}: {msg}')
        if not ok:
            _results.append(('FAIL', f'DIAG_SETUP {fname}'))
        else:
            _results.append(('PASS', f'DIAG_SETUP {fname}'))


# ── Test 5 : Résultats d'évaluation depuis les logs ───────────────────────────

def test_eval_results(logs_dir):
    section(f'5. Résultats ITEM RANKING dans les logs')

    if not os.path.isdir(logs_dir):
        print(f'  [{WARN}] logs_dir introuvable')
        return

    coldstart_logs = [f for f in os.listdir(logs_dir)
                      if 'coldstart' in f and f.endswith('.out')]

    THRESHOLD_GLOBAL  = 0.05   # nDCG@10 global minimum acceptable (sanity check)
    THRESHOLD_COLD    = 0.01   # cold doit être > 0 pour être utile

    for fname in sorted(coldstart_logs):
        fpath = os.path.join(logs_dir, fname)
        with open(fpath) as fh:
            content = fh.read()

        # Parse ITEM RANKING block
        m_global  = re.findall(r'@10\s+([\d.]+)', content)
        m_cold    = re.search(r'\[cold\]\s*\n\s*@5[^\n]*\n\s*@10\s+([\d.]+)', content)
        m_popular = re.search(r'\[popular\]\s*\n\s*@5[^\n]*\n\s*@10\s+([\d.]+)', content)

        if not m_global:
            # Job still running or no ranking block yet
            # Check last epoch
            epoch_m = re.findall(r'Epoch=(\d+)', content)
            last_ep = epoch_m[-1] if epoch_m else '?'
            print(f'  [{INFO}] {fname}: pas encore de ITEM RANKING (dernier epoch: {last_ep})')
            continue

        # First @10 occurrence is usually the global one before the per-group breakdown
        # Find the block precisely
        rank_block = re.search(
            r'──.*ITEM RANKING.*\n(.*\n){0,5}.*@10\s+([\d.]+)',
            content
        )

        # Extract all @10 values near ITEM RANKING
        block_m = re.search(r'ITEM RANKING.*?={20,}', content, re.DOTALL)
        if not block_m:
            print(f'  [{INFO}] {fname}: bloc ITEM RANKING incomplet')
            continue

        block = block_m.group(0)
        global_10_m  = re.search(r'@10\s+([\d.]+)', block)
        cold_10_m    = re.search(r'\[cold\][^\n]*\n[^\n]*\n\s*@10\s+([\d.]+)', block)
        popular_10_m = re.search(r'\[popular\][^\n]*\n[^\n]*\n\s*@10\s+([\d.]+)', block)

        g10 = float(global_10_m.group(1))  if global_10_m  else None
        c10 = float(cold_10_m.group(1))    if cold_10_m    else None
        p10 = float(popular_10_m.group(1)) if popular_10_m else None

        print(f'\n  {fname}')
        if g10 is not None:
            ok = g10 >= THRESHOLD_GLOBAL
            print(f'    [{"PASS" if ok else FAIL}] Global @10 = {g10:.4f}  (seuil ≥ {THRESHOLD_GLOBAL})')
            _results.append(('PASS' if ok else 'FAIL', f'{fname} global@10'))
        if c10 is not None:
            ok = c10 >= THRESHOLD_COLD
            print(f'    [{"PASS" if ok else FAIL}] Cold   @10 = {c10:.4f}  (seuil ≥ {THRESHOLD_COLD})')
            _results.append(('PASS' if ok else 'FAIL', f'{fname} cold@10'))
            if c10 == 0.0:
                print(f'    [{WARN}] Cold @10 = 0.000 — les cold items ne sont jamais bien rankés.'
                      '  Cause probable : cold items absents des batches train (degree=0).')
        if p10 is not None:
            print(f'    [{INFO}] Popular@10 = {p10:.4f}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Vérifie le pipeline coldstart Musical')
    parser.add_argument('--logs_dir', default=LOGS_DIR_DEFAULT,
                        help='Répertoire des logs SLURM')
    args = parser.parse_args()

    print(f'\n{"="*60}')
    print('  Pipeline coldstart — tests de validité')
    print(f'  Dataset : Musical_HADSF_coldstart  ({N_ITEMS_TOTAL} items)')
    print(f'{"="*60}')

    cold_set = test_cold_items_json()
    deg_full = test_csv_no_cold(cold_set)
    test_splits_integrity(cold_set, deg_full)
    test_diag_setup(args.logs_dir, cold_set)
    test_eval_results(args.logs_dir)

    # Résumé
    n_pass = sum(1 for t, _ in _results if t == 'PASS')
    n_fail = sum(1 for t, _ in _results if t == 'FAIL')
    n_warn = sum(1 for t, _ in _results if t == 'WARN')
    print(f'\n{"="*60}')
    print(f'  Résumé : {n_pass} PASS  |  {n_fail} FAIL  |  {n_warn} WARN')
    if n_fail:
        print('\n  Échecs :')
        for t, label in _results:
            if t == 'FAIL':
                print(f'    ✗ {label}')
    print(f'{"="*60}\n')

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
