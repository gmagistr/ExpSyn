import io
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="Auto-ML Light Platform", layout="wide")
st.title("Piattaforma Auto-ML leggera per CSV")
st.write("Carica un CSV, seleziona la/e variabile/i dipendente/i e lascia che il sistema provi diverse strategie di ML per scegliere la migliore.")


def auto_detect_problem_type(y: pd.Series) -> str:
    # Heuristic: categorical/object -> classification; if few unique levels -> classification; else regression
    nunique = y.nunique(dropna=True)
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classificazione"
    # small number of unique values relative to size => likely classification
    if nunique <= 20 and nunique / max(len(y), 1) < 0.2:
        return "classificazione"
    return "regressione"


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_models(problem_type: str) -> Dict[str, Any]:
    if problem_type == "classificazione":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        }


def get_scoring(problem_type: str, metric_choice: str) -> str:
    if problem_type == "classificazione":
        mapping = {
            "accuracy": "accuracy",
            "f1 (weighted)": "f1_weighted",
            "roc_auc (ovr weighted)": "roc_auc_ovr_weighted",
        }
        return mapping[metric_choice]
    else:
        mapping = {
            "r2": "r2",
            "MAE (negativo)": "neg_mean_absolute_error",
            "RMSE (negativo)": "neg_root_mean_squared_error",
        }
        return mapping[metric_choice]


def fit_and_select_best(X: pd.DataFrame, y: pd.Series, problem_type: str, cv_options: List[int], metric_choice: str, random_state: int) -> Dict[str, Any]:
    preprocessor_base, num_cols, cat_cols = build_preprocessor(X)

    # Two preprocessing variants: with and without feature selection (50% percentile)
    if problem_type == "classificazione":
        selector = SelectPercentile(score_func=f_classif, percentile=50)
    else:
        selector = SelectPercentile(score_func=f_regression, percentile=50)

    preprocess_variants = {
        "base": Pipeline(steps=[("preprocessor", preprocessor_base)]),
        "selezione_feature_50%": Pipeline(steps=[("preprocessor", preprocessor_base), ("select", selector)])
    }

    models = get_models(problem_type)
    scoring = get_scoring(problem_type, metric_choice)

    # Choose CV splitter
    candidates = []

    for cv_folds in cv_options:
        if problem_type == "classificazione":
            splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for prep_name, prep_pipe in preprocess_variants.items():
            for model_name, model in models.items():
                pipe = Pipeline(steps=[
                    ("prep", prep_pipe),
                    ("model", model)
                ])

                try:
                    cv_res = cross_validate(pipe, X, y, cv=splitter, scoring=scoring, return_train_score=False, n_jobs=-1)
                    mean_score = float(np.mean(cv_res["test_score"]))
                    std_score = float(np.std(cv_res["test_score"]))
                    candidates.append({
                        "cv_folds": cv_folds,
                        "preprocessing": prep_name,
                        "model": model_name,
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "pipeline": pipe,
                    })
                except Exception as e:
                    # Skip incompatible combos silently but record info
                    candidates.append({
                        "cv_folds": cv_folds,
                        "preprocessing": prep_name,
                        "model": model_name,
                        "mean_score": -np.inf,
                        "std_score": np.inf,
                        "pipeline": None,
                        "error": str(e),
                    })

    # Select best by mean_score
    candidates_valid = [c for c in candidates if c["pipeline"] is not None and np.isfinite(c["mean_score"])]
    if not candidates_valid:
        # fallback: raise last error
        raise RuntimeError("Nessuna combinazione valida trovata. Controlla i dati o riduci le opzioni.")

    best = max(candidates_valid, key=lambda c: c["mean_score"])

    return {
        "best": best,
        "all_candidates": sorted(candidates_valid, key=lambda c: c["mean_score"], reverse=True)
    }


def train_test_evaluate(best_info: Dict[str, Any], X: pd.DataFrame, y: pd.Series, problem_type: str, test_size: float, random_state: int) -> Dict[str, Any]:
    stratify = y if problem_type == "classificazione" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    pipeline = best_info["pipeline"]
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    results: Dict[str, Any] = {
        "pipeline": pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    if problem_type == "classificazione":
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        # Try AUC if proba/decision_function available and multiclass-capable
        auc = None
        try:
            if hasattr(pipeline, "predict_proba") or hasattr(pipeline, "decision_function"):
                if hasattr(pipeline, "predict_proba"):
                    y_score = pipeline.predict_proba(X_test)
                else:
                    y_score = pipeline.decision_function(X_test)
                # Handle binary/multiclass
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_score[:, 1] if y_score.ndim == 2 else y_score)
                else:
                    auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        results.update({
            "accuracy": acc,
            "f1_weighted": f1w,
            "roc_auc": auc,
            "report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        })
    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        results.update({
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })

    return results


def extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    # Attempt to extract feature names and importances/coefficients
    try:
        # Expand nested pipelines: we used steps: ("prep", <Pipeline>), ("model", <Estimator>)
        prep_pipe: Pipeline = pipeline.named_steps["prep"]
        preprocessor: ColumnTransformer = prep_pipe.named_steps["preprocessor"]

        feature_names = []
        # numeric
        num_transformer: Pipeline = preprocessor.named_transformers_["num"]
        num_cols = preprocessor.transformers_[0][2]
        if hasattr(num_transformer, "get_feature_names_out"):
            num_names = [str(n) for n in num_cols]
        else:
            num_names = [str(n) for n in num_cols]
        # categorical
        cat_transformer: Pipeline = preprocessor.named_transformers_["cat"]
        cat_encoder: OneHotEncoder = cat_transformer.named_steps["onehot"]
        cat_cols = preprocessor.transformers_[1][2]
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))

        feature_names = num_names + cat_names

        # If selector exists
        if "select" in prep_pipe.named_steps:
            selector = prep_pipe.named_steps["select"]
            support_mask = selector.get_support()
            feature_names = [name for keep, name in zip(support_mask, feature_names) if keep]

        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False)
            return fi
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            fi = pd.DataFrame({"feature": feature_names, "importance": coef})
            fi = fi.sort_values("importance", ascending=False)
            return fi
    except Exception:
        pass

    return pd.DataFrame(columns=["feature", "importance"])


uploaded = st.file_uploader("Carica un file CSV", type=["csv"]) 

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        # Try semicolon
        df = pd.read_csv(uploaded, sep=";")

    st.subheader("Anteprima dati")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} righe x {df.shape[1]} colonne")

    target_cols = st.multiselect("Seleziona la/e variabile/i dipendente/i (target)", options=df.columns.tolist())

    # General options
    with st.expander("Impostazioni esperimento"):
        test_size = st.slider("Dimensione test set", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        cv_choices = st.multiselect("Numero di fold per CV da provare", options=[5, 10], default=[5, 10])
        random_state = st.number_input("Random seed", value=42, step=1)

    if target_cols:
        run_btn = st.button("Esegui Auto-ML")
        if run_btn:
            for target in target_cols:
                st.markdown(f"### Target: `{target}`")
                y = df[target]
                X = df.drop(columns=target_cols) if len(target_cols) > 1 else df.drop(columns=[target])

                auto_type = auto_detect_problem_type(y)
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Tipo problema (auto): **{auto_type}**")
                with col2:
                    if auto_type == "classificazione":
                        metric_choice = st.selectbox(
                            f"Metrica di valutazione per `{target}`",
                            options=["accuracy", "f1 (weighted)", "roc_auc (ovr weighted)"],
                            index=0,
                            key=f"metric_{target}"
                        )
                    else:
                        metric_choice = st.selectbox(
                            f"Metrica di valutazione per `{target}`",
                            options=["r2", "MAE (negativo)", "RMSE (negativo)"],
                            index=0,
                            key=f"metric_{target}"
                        )

                with st.spinner("Ricerca del modello migliore in corso..."):
                    try:
                        selection = fit_and_select_best(X, y, auto_type, cv_choices, metric_choice, int(random_state))
                        best = selection["best"]

                        st.success("Miglior combinazione trovata")
                        st.write({
                            "cv_folds": best["cv_folds"],
                            "preprocessing": best["preprocessing"],
                            "model": best["model"],
                            "cv_mean_score": round(best["mean_score"], 4),
                            "cv_std": round(best["std_score"], 4),
                        })

                        eval_res = train_test_evaluate(best, X, y, auto_type, test_size, int(random_state))

                        if auto_type == "classificazione":
                            st.write(f"Accuracy (test): **{eval_res['accuracy']:.4f}**")
                            st.write(f"F1 weighted (test): **{eval_res['f1_weighted']:.4f}**")
                            if eval_res.get("roc_auc") is not None:
                                st.write(f"ROC AUC (test): **{eval_res['roc_auc']:.4f}**")
                            with st.expander("Report di classificazione"):
                                st.text(eval_res["report"]) 
                            with st.expander("Matrice di confusione"):
                                st.dataframe(pd.DataFrame(eval_res["confusion_matrix"]))
                        else:
                            st.write(f"R2 (test): **{eval_res['r2']:.4f}**")
                            st.write(f"MAE (test): **{eval_res['mae']:.4f}**")
                            st.write(f"RMSE (test): **{eval_res['rmse']:.4f}**")

                        # Show top candidates table
                        st.markdown("#### Top combinazioni (CV)")
                        top_df = pd.DataFrame([
                            {
                                "model": c["model"],
                                "preprocessing": c["preprocessing"],
                                "cv_folds": c["cv_folds"],
                                "cv_mean_score": c["mean_score"],
                                "cv_std": c["std_score"],
                            }
                            for c in selection["all_candidates"][:20]
                        ])
                        st.dataframe(top_df, use_container_width=True)

                        # Feature importances
                        st.markdown("#### Importanza delle feature")
                        fi = extract_feature_importances(eval_res["pipeline"]) 
                        if not fi.empty:
                            st.dataframe(fi.head(30), use_container_width=True)
                        else:
                            st.caption("Il modello selezionato non fornisce importanze/coefficiente interpretabili.")

                        # Download best model
                        st.markdown("#### Scarica il modello addestrato")
                        buffer = io.BytesIO()
                        pickle.dump(eval_res["pipeline"], buffer)
                        buffer.seek(0)
                        st.download_button(
                            label="Scarica modello (.pkl)",
                            data=buffer,
                            file_name=f"best_model_{target}.pkl",
                            mime="application/octet-stream",
                        )

                    except Exception as e:
                        st.error(f"Errore durante la ricerca: {e}")