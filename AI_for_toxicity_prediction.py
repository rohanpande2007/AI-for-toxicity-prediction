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

def load_dataset():
    try:
        df = pd.read_csv("tox21_sample.csv")
        print("Loaded real dataset")
    except:
        print("Using demo dataset")
        smiles = [
            "CCO","CCC","COC","CCOCC","CCCl","CCBr","C#N",
            "CCN","CCCN","CCClC","CCS","CN","CCNCC",
            "CC1=CC=C(C=C1)N=NC2=CC=CC=C2"
        ]
        toxicity = [0,0,0,0,1,1,1,1,1,1,1,1,1,1]
        df = pd.DataFrame({"SMILES": smiles, "Toxicity": toxicity})

    return df

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        fp = np.array(fp)

        descriptors = [
            mol.GetNumAtoms(),
            mol.GetNumBonds(),
            Chem.rdMolDescriptors.CalcTPSA(mol),
            Chem.rdMolDescriptors.CalcExactMolWt(mol)
        ]

        return np.concatenate([fp, descriptors])

    return np.zeros(260)

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

def mixture_features(sm1, sm2):
    f1 = featurize(sm1)
    f2 = featurize(sm2)

    concat = np.concatenate([f1, f2])
    diff = np.abs(f1 - f2)
    mult = f1 * f2

    return np.concatenate([concat, diff, mult])

def train_model(X, y):
    counter = Counter(y)
    print("Class distribution:", counter)

    min_samples = min(counter.values())

    if min_samples > 1:
        k = min(5, min_samples - 1)  # FIXED
        print(f"Using SMOTE with k_neighbors={k}")

        smote = SMOTE(k_neighbors=k, random_state=42)
        X, y = smote.fit_resample(X, y)
    else:
        print("Skipping SMOTE (not enough samples)")

    model = XGBClassifier(
        eval_metric='logloss',
        max_depth=5,
        n_estimators=200,
        learning_rate=0.05
    )

    model.fit(X, y)
    return model

def train_mixture_model(df):
    mixtures = []
    labels = []

    for i in range(len(df)):
        for j in range(i+1, len(df)):
            sm1 = df.iloc[i]["SMILES"]
            sm2 = df.iloc[j]["SMILES"]

            mixtures.append(mixture_features(sm1, sm2))

            label = max(df.iloc[i]["Toxicity"], df.iloc[j]["Toxicity"])
            labels.append(label)

    X = np.array(mixtures)
    y = np.array(labels)

    return train_model(X, y)

def main():
    print("AI Toxicity Research System")

    df = load_dataset()
    X = np.array([featurize(s) for s in df["SMILES"]])
    y = df["Toxicity"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:50])
    shap.summary_plot(shap_values, X_train[:50])
    test = "CC1=CC=C(C=C1)N=NC2=CC=CC=C2"

    print("\nTest molecule:", test)

    pred = model.predict(featurize(test).reshape(1, -1))
    print("Original:", "Toxic" if pred[0] else "Non-Toxic")

    products = azo_break(test)
    print("Products:", products)

    for p in products:
        pred = model.predict(featurize(p).reshape(1, -1))
        print(p, "→", "Toxic" if pred[0] else "Non-Toxic")
    print("\n--- Mixture Toxicity ---")

    mix_model = train_mixture_model(df)

    mol1 = "CCCl"
    mol2 = "CCO"

    mix_feat = mixture_features(mol1, mol2).reshape(1, -1)
    mix_pred = mix_model.predict(mix_feat)

    print(f"{mol1} + {mol2} →", "Toxic" if mix_pred[0] else "Non-Toxic")

if __name__ == "__main__":
    main()
