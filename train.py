"""Train on synthetic S-02 listings and export predictions for the static demo."""
import csv
import io
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DEMO = ['PR00080', 'PR03061', 'PR02140']
OUT = Path(__file__).resolve().parent / 'dist'
CLUES = [
    'Çmimi në raport me tregun', 'Çmimi nën 60% të tregut',
    'Pretendim për origjinalitet', 'Gjuhë e ngjashmërisë ose imitimit',
    'Fjalë që mund të lidhen me replika', 'Variant i emrit të markës',
    'Përshkrim që kërkon kontakt privat', 'Paketim ose etiketë e diskutueshme',
    'Llogari e krijuar pas 5 majit 2026',
]
PATTERNS = [
    r'origjinal|autentik|dyqan zyrtar',
    r'i ngjashëm|e ngjashme|në frymën|frymëzuar|në stilin|si modeli',
    r'\breplik\w*|\bkopj\w*|1:1|\baaa\b|first copy|\bmirror\b',
    r'inbox|dm për|nuk e di|më shkruani',
    r'pa kuti|pa etiket|pa vul|qese plastik|kuti e bardhë',
]


def read_rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig'))))


def photo_features(archive, path):
    if not path.startswith('foto/PR'):
        return [0] * 7
    with Image.open(io.BytesIO(archive.read(path))) as image:
        pixels = np.asarray(image.convert('RGB').resize((32, 32))) / 255
    return [1, *pixels.mean(axis=(0, 1)), *pixels.std(axis=(0, 1))]


def features(row, sellers, brands, archive):
    text = (row['titulli'] + ' ' + row['pershkrimi']).casefold()
    ratio = int(row['cmimi_lek']) / int(row['cmimi_tregut_lek'])
    variants = brands[row['marka']]['variante_te_falsifikuara'].split(';')
    changed = any(re.search(r'(?<!\w)' + re.escape(v.casefold()) + r'(?!\w)', text)
                  for v in variants if v)
    recent = sellers[row['shitesi_id']]['data_krijimit'] > '2026-05-05'
    return [ratio, ratio < .60, *(bool(re.search(p, text)) for p in PATTERNS[:3]),
            changed, *(bool(re.search(p, text)) for p in PATTERNS[3:]),
            recent, *photo_features(archive, row['foto'])]


with ZipFile(sys.argv[1]) as archive:
    rows = read_rows(archive, 'produktet.csv')
    sellers = {r['shitesi_id']: r for r in read_rows(archive, 'shitesit.csv')}
    brands = {r['marka']: r for r in read_rows(archive, 'markat.csv')}
    X = np.array([features(r, sellers, brands, archive) for r in rows])
    y = np.array([r['etiketa'] for r in rows])
    groups = np.array([r['shitesi_id'] for r in rows])
    demo_sellers = {r['shitesi_id'] for r in rows if r['produkt_id'] in DEMO}
    train, test = next(GroupShuffleSplit(test_size=.22, random_state=42).split(X, y, groups))
    move = np.isin(groups[train], list(demo_sellers))
    test = np.concatenate([test, train[move]])
    train = train[~move]
    assert not set(groups[train]) & set(groups[test])

    scaler = StandardScaler().fit(X[train])
    scaled = scaler.transform(X)
    model = LogisticRegression(max_iter=500, class_weight='balanced').fit(scaled[train], y[train])
    predictions = model.predict(scaled)
    listings = []
    for index in test:
        row = rows[index]
        label = predictions[index]
        coefficients = model.coef_[list(model.classes_).index(label)]
        effects = (X[index] / scaler.scale_) * coefficients
        ranked = sorted(range(1, len(CLUES)), key=lambda j: effects[j], reverse=True)
        reasons = [CLUES[j] for j in ranked if effects[j] > 0][:3]
        reasons.insert(0, f"Çmimi është {X[index, 0]:.0%} e çmimit referencë.")
        image_used = row['foto'].startswith('foto/PR')
        if image_used:
            reasons.append('Modeli përdori statistika të ngjyrave të fotos.')
        entry = {k: row[k] for k in ('produkt_id','marka','kategoria','titulli',
                                   'pershkrimi','cmimi_lek','cmimi_tregut_lek',
                                   'kanali','qyteti','shitesi_id')}
        entry.update(prediction=str(label), reasons=reasons, image_used=image_used,
                     seller=sellers[row['shitesi_id']]['emri_profilit'])
        if row['foto']:
            image = 'photos/' + Path(row['foto']).name
            (OUT / image).write_bytes(archive.read(row['foto']))
            entry['image'] = image
        if row['produkt_id'] in DEMO:
            entry['reference'] = row['etiketa']
        listings.append(entry)

    metrics = dict(train=len(train), test=len(test), accuracy=accuracy_score(y[test], predictions[test]),
                   macro_f1=f1_score(y[test], predictions[test], average='macro'))
    (OUT / 'data.json').write_text(json.dumps(dict(demo=DEMO, metrics=metrics, listings=listings),
                                              ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(metrics)
    print([(r['produkt_id'], r['prediction'], r['reference']) for r in listings if 'reference' in r])
