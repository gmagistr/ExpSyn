import io
import pickle
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate,
    GroupKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
)

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
st.caption("Carica un CSV, seleziona la/e variabile/i dipendente/i e lascia che il sistema provi diverse strategie di ML per scegliere la migliore.")


def auto_detect_problem_type(y: pd.Series) -> str:
    nunique = y.nunique(dropna=True)
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classificazione"
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


def leakage_report(df_X: pd.DataFrame, y: pd.Series, problem_type: str) -> Tuple[List[str], pd.DataFrame]:
    warnings: List[str] = []
    suspicious_rows: List[Dict[str, Any]] = []

    # Numeric extreme correlations
    numeric_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        try:
            if problem_type == "regressione":
                corr = df_X[numeric_cols].corrwith(y, method="pearson").abs()
                for feat, val in corr.dropna().items():
                    if val >= 0.995:
                        suspicious_rows.append({"feature": feat, "reason": "correlazione Pearson ~1 con target", "score": float(val)})
            else:
                # Encode target to integers for a rough correlation check
                y_encoded = pd.factorize(y)[0]
                corr = df_X[numeric_cols].corrwith(pd.Series(y_encoded, index=y.index), method="pearson").abs()
                for feat, val in corr.dropna().items():
                    if val >= 0.95:
                        suspicious_rows.append({"feature": feat, "reason": "correlazione alta con target (proxy)", "score": float(val)})
        except Exception:
            pass

    # Deterministic mapping category -> target (classification)
    if problem_type == "classificazione":
        cat_cols = [c for c in df_X.columns if c not in numeric_cols]
        for c in cat_cols:
            try:
                grp = df_X.groupby(c)[y.name].nunique()
                if (grp == 1).all() and grp.size > 1:
                    suspicious_rows.append({"feature": c, "reason": "mappatura deterministica categoria->target", "score": 1.0})
            except Exception:
                continue

    suspicious_df = pd.DataFrame(suspicious_rows)
    if not suspicious_df.empty:
        warnings.append("Possibile data leakage: alcune feature hanno correlazione estrema con il target o mappatura deterministica.")

    # Duplicates
    dup_count = df_X.join(y).duplicated().sum()
    if dup_count > 0:
        warnings.append(f"Rilevate {dup_count} righe duplicate. Valuta la rimozione prima dello split.")

    return warnings, suspicious_df


def fit_and_select_best(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    cv_options: List[int],
    metric_choice: str,
    random_state: int,
    split_strategy: str,
    groups: Optional[pd.Series] = None,
    time_col: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    preprocessor_base, _, _ = build_preprocessor(X)

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

    candidates = []

    # If time-based, ensure sorted by time
    if split_strategy == "time" and time_col is not None:
        order = np.argsort(pd.to_datetime(time_col, errors="coerce").values)
        X = X.iloc[order]
        y = y.iloc[order]
        if groups is not None:
            groups = groups.iloc[order]

    for cv_folds in cv_options:
        if split_strategy == "group" and groups is not None:
            if problem_type == "classificazione":
                splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                splitter = GroupKFold(n_splits=cv_folds)
            split_kwargs = {"groups": groups}
        elif split_strategy == "time":
            splitter = TimeSeriesSplit(n_splits=cv_folds)
            split_kwargs = {}
        else:
            if problem_type == "classificazione":
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            split_kwargs = {}

        for prep_name, prep_pipe in preprocess_variants.items():
            for model_name, model in models.items():
                pipe = Pipeline(steps=[
                    ("prep", prep_pipe),
                    ("model", model)
                ])

                try:
                    cv_res = cross_validate(
                        pipe, X, y,
                        cv=splitter,
                        scoring=scoring,
                        return_train_score=False,
                        n_jobs=-1,
                        **split_kwargs,
                    )
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
                    candidates.append({
                        "cv_folds": cv_folds,
                        "preprocessing": prep_name,
                        "model": model_name,
                        "mean_score": -np.inf,
                        "std_score": np.inf,
                        "pipeline": None,
                        "error": str(e),
                    })

    candidates_valid = [c for c in candidates if c["pipeline"] is not None and np.isfinite(c["mean_score"])]
    if not candidates_valid:
        raise RuntimeError("Nessuna combinazione valida trovata. Controlla i dati o riduci le opzioni.")

    best = max(candidates_valid, key=lambda c: c["mean_score"])
    return {"best": best, "all_candidates": sorted(candidates_valid, key=lambda c: c["mean_score"], reverse=True)}


def train_test_evaluate(
    best_info: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    test_size: float,
    random_state: int,
    split_strategy: str,
    groups: Optional[pd.Series] = None,
    time_col: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if split_strategy == "group" and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    elif split_strategy == "time" and time_col is not None:
        order = np.argsort(pd.to_datetime(time_col, errors="coerce").values)
        X_sorted, y_sorted = X.iloc[order], y.iloc[order]
        split_point = int(np.floor((1 - test_size) * len(X_sorted)))
        X_train, X_test = X_sorted.iloc[:split_point], X_sorted.iloc[split_point:]
        y_train, y_test = y_sorted.iloc[:split_point], y_sorted.iloc[split_point:]
    else:
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
        auc = None
        try:
            if hasattr(pipeline, "predict_proba") or hasattr(pipeline, "decision_function"):
                if hasattr(pipeline, "predict_proba"):
                    y_score = pipeline.predict_proba(X_test)
                else:
                    y_score = pipeline.decision_function(X_test)
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
        results.update({"r2": r2, "mae": mae, "rmse": rmse})

    return results


def extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    try:
        prep_pipe: Pipeline = pipeline.named_steps["prep"]
        preprocessor: ColumnTransformer = prep_pipe.named_steps["preprocessor"]

        num_cols = preprocessor.transformers_[0][2]
        num_names = [str(n) for n in num_cols]
        cat_transformer: Pipeline = preprocessor.named_transformers_["cat"]
        cat_encoder: OneHotEncoder = cat_transformer.named_steps["onehot"]
        cat_cols = preprocessor.transformers_[1][2]
        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        feature_names = num_names + cat_names

        if "select" in prep_pipe.named_steps:
            selector = prep_pipe.named_steps["select"]
            support_mask = selector.get_support()
            feature_names = [name for keep, name in zip(support_mask, feature_names) if keep]

        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
            return fi
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            fi = pd.DataFrame({"feature": feature_names, "importance": coef}).sort_values("importance", ascending=False)
            return fi
    except Exception:
        pass

    return pd.DataFrame(columns=["feature", "importance"])


# Sidebar controls
with st.sidebar:
    st.header("Configurazione")
    uploaded = st.file_uploader("Carica un file CSV", type=["csv"]) 
    test_size = st.slider("Dimensione test set", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    cv_choices = st.multiselect("Fold per CV", options=[5, 10], default=[5, 10])
    random_state = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.subheader("Controlli data leakage")
    enable_leak_checks = st.checkbox("Abilita controlli", value=True)
    split_strategy = st.radio("Strategia di split", options=["random", "group", "time"], index=0)
    group_col_name = None
    time_col_name = None
    drop_duplicates = st.checkbox("Rimuovi righe duplicate prima dello split", value=True)


if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")

    st.subheader("Anteprima dati")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} righe x {df.shape[1]} colonne")

    # Targets and special columns
    target_cols = st.multiselect("Seleziona target", options=df.columns.tolist())

    if split_strategy == "group":
        group_col_name = st.selectbox("Colonna gruppi (stesso gruppo non deve apparire in train e test)", options=[None] + df.columns.tolist(), index=0)
    elif split_strategy == "time":
        time_col_name = st.selectbox("Colonna temporale (usata per ordinare e fare split futuro)", options=[None] + df.columns.tolist(), index=0)

    if target_cols:
        run_btn = st.button("Esegui Auto-ML")
        if run_btn:
            # Prepare dataframe (optional dedup)
            work_df = df.copy()
            if drop_duplicates:
                work_df = work_df.drop_duplicates()

            for target in target_cols:
                st.markdown(f"### Target: `{target}`")
                y = work_df[target]
                other_targets = [t for t in target_cols if t != target]
                X = work_df.drop(columns=[target] + other_targets)

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

                # Prepare special series
                groups_series = work_df[group_col_name] if (split_strategy == "group" and group_col_name is not None) else None
                time_series = work_df[time_col_name] if (split_strategy == "time" and time_col_name is not None) else None

                # Leakage checks
                if enable_leak_checks:
                    with st.expander("Report data leakage potenziale"):
                        try:
                            wns, sus = leakage_report(X, y, auto_type)
                            if wns:
                                for w in wns:
                                    st.warning(w)
                            if not sus.empty:
                                st.dataframe(sus.sort_values("score", ascending=False), use_container_width=True)
                            else:
                                st.caption("Nessuna evidenza forte di leakage dalle euristiche applicate.")
                            if split_strategy == "time" and time_series is None:
                                st.info("Hai scelto split temporale ma non hai selezionato alcuna colonna temporale.")
                            if split_strategy == "group" and groups_series is None:
                                st.info("Hai scelto split a gruppi ma non hai selezionato alcuna colonna gruppi.")
                        except Exception as e:
                            st.info(f"Impossibile calcolare il report di leakage: {e}")

                with st.spinner("Ricerca del modello migliore in corso..."):
                    try:
                        selection = fit_and_select_best(
                            X, y,
                            auto_type,
                            cv_choices,
                            metric_choice,
                            int(random_state),
                            split_strategy=split_strategy,
                            groups=groups_series,
                            time_col=time_series,
                        )
                        best = selection["best"]

                        st.success("Miglior combinazione trovata")
                        st.write({
                            "cv_folds": best["cv_folds"],
                            "preprocessing": best["preprocessing"],
                            "model": best["model"],
                            "cv_mean_score": round(best["mean_score"], 4),
                            "cv_std": round(best["std_score"], 4),
                        })

                        eval_res = train_test_evaluate(
                            best, X, y, auto_type, test_size, int(random_state),
                            split_strategy=split_strategy, groups=groups_series, time_col=time_series
                        )

                        # UI tabs for results
                        if auto_type == "classificazione":
                            tabs = st.tabs(["Metriche", "Report", "Confusione", "Top CV", "Importanze", "Scarica modello"])
                            with tabs[0]:
                                st.metric("Accuracy (test)", f"{eval_res['accuracy']:.4f}")
                                st.write(f"F1 weighted: **{eval_res['f1_weighted']:.4f}**")
                                if eval_res.get("roc_auc") is not None:
                                    st.write(f"ROC AUC: **{eval_res['roc_auc']:.4f}**")
                            with tabs[1]:
                                st.text(eval_res["report"]) 
                            with tabs[2]:
                                st.dataframe(pd.DataFrame(eval_res["confusion_matrix"]))
                            with tabs[3]:
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
                            with tabs[4]:
                                fi = extract_feature_importances(eval_res["pipeline"]) 
                                if not fi.empty:
                                    st.dataframe(fi.head(30), use_container_width=True)
                                else:
                                    st.caption("Il modello selezionato non fornisce importanze/coefficiente interpretabili.")
                            with tabs[5]:
                                buffer = io.BytesIO()
                                pickle.dump(eval_res["pipeline"], buffer)
                                buffer.seek(0)
                                st.download_button(
                                    label="Scarica modello (.pkl)",
                                    data=buffer,
                                    file_name=f"best_model_{target}.pkl",
                                    mime="application/octet-stream",
                                )
                        else:
                            tabs = st.tabs(["Metriche", "Top CV", "Importanze", "Scarica modello"])
                            with tabs[0]:
                                st.metric("R2 (test)", f"{eval_res['r2']:.4f}")
                                st.write(f"MAE: **{eval_res['mae']:.4f}**")
                                st.write(f"RMSE: **{eval_res['rmse']:.4f}**")
                            with tabs[1]:
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
                            with tabs[2]:
                                fi = extract_feature_importances(eval_res["pipeline"]) 
                                if not fi.empty:
                                    st.dataframe(fi.head(30), use_container_width=True)
                                else:
                                    st.caption("Il modello selezionato non fornisce importanze/coefficiente interpretabili.")
                            with tabs[3]:
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