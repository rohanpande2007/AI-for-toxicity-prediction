import pandas as pd
import numpy as np
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import shap
import matplotlib.pyplot as plt

def create_dataset():
    print("Creating fresh dataset...")

    smiles = [
        # NON-TOXIC
        "CCO","CCC","COC","CCOCC","CCCO","CO","CCOCCC","COCC",
        "CC(C)O","CCOC(=O)C","CCOCCO",

        # TOXIC
        "CCCl","CCBr","C#N","CCN","CCCN","CCClC","CCBrC","CCS",
        "CN","CCNCC","CCClCC","CCBrCC","CCF","CCFCCC",

        # AZO (toxic)
        "CCN=NC","N=N",
        "CC1=CC=C(C=C1)N=NC2=CC=CC=C2"
    ]

    toxicity = [0]*11 + [1]*14 + [1]*3

    df = pd.DataFrame({"SMILES": smiles, "Toxicity": toxicity})
    df.to_csv("toxicity_data.csv", index=False)

def features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        fp = np.array(fp)

        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        return np.concatenate([fp, [num_atoms, num_bonds]])

    return np.zeros(258)

def azo_break(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rxn = AllChem.ReactionFromSmarts('[N:1]=[N:2]>>[N:1].[N:2]')

    try:
        products = rxn.RunReactants((mol,))
        if products:
            return [Chem.MolToSmiles(p) for p in products[0]]
    except:
        pass

    return []

def train(X, y):
    counter = Counter(y)
    print("Class distribution:", counter)

    min_samples = min(counter.values())

    if min_samples > 1:
        k = min(5, min_samples - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        X, y = smote.fit_resample(X, y)

    model = XGBClassifier(
        eval_metric='logloss',
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1
    )

    model.fit(X, y)
    return model

def main():
    print("Starting AI Toxicity System")

    # FORCE NEW DATASET
    create_dataset()

    data = pd.read_csv("toxicity_data.csv")

    print("\nDataset size:", len(data))
    print(data['Toxicity'].value_counts())

    X = np.array([features(s) for s in data['SMILES']])
    y = data['Toxicity'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = train(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)

    print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]

    plt.barh(range(10), importances[indices])
    plt.yticks(range(10), indices)
    plt.title("Top Feature Importance")
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:100])
    shap.summary_plot(shap_values, X_train[:100])

    test = "CC1=CC=C(C=C1)N=NC2=CC=CC=C2"
    print("\nTesting:", test)

    pred = model.predict(features(test).reshape(1, -1))
    print("Original:", "Toxic" if pred[0] else "Non-Toxic")

    products = azo_break(test)
    print("\nProducts:", products)

    for p in products:
        pred = model.predict(features(p).reshape(1, -1))
        print(p, "→", "Toxic" if pred[0] else "Non-Toxic")

if __name__ == "__main__":
    main()
