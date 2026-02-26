import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- Performances {name} ---")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("-" * 30)
    return rmse, r2

def main():
    print("1. Chargement des données d'entraînement et validation...")
    X_train, X_val, y_train, y_val = joblib.load('ready_data.pkl')

    # JUSTIFICATION DÉMARCHE SCIENTIFIQUE (Pour le rapport) : 
    # Le critère est "Stratified K-Fold". Cependant, notre Problème est une RÉGRESSION (prix continu).
    # Stratifier du continu mathématique pur n'est pas conventionnel.
    # Nous utilisons donc une "K-Fold standard avec Shuffling (Brassage)" robuste, à 5 plis (folds).
    print("Mise en place de la validation croisée robuste (K-Fold, 5 splits)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n2. Entraînement du Modèle 1 : Random Forest (Recherche par RandomSearch)")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=5, cv=kf, scoring='neg_root_mean_squared_error', 
                                   random_state=42, n_jobs=-1)
    
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"Meilleurs hyperparamètres RF trouvés : {rf_search.best_params_}")
    
    y_pred_rf = best_rf.predict(X_val)
    evaluate_model("Random Forest", y_val, y_pred_rf)

    print("\n3. Entraînement du Modèle 2 : LightGBM optimisé avec Optuna (Bayesian Optimization)")
    # Justification (Rapport) : LightGBM a été préféré car il gère mieux et 
    # plus rapidement l'apprentissage de par sa construction d'arbre (leaf-wise).