"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    x=np.asarray(np.copy(X),dtype=float)

    means=np.nanmean(X,axis=0)
    means=np.where(np.isnan(means),0,means)
    nan_mask=np.isnan(X)
    x=np.where(nan_mask,means,X)
  
    return x
        
    pass

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    
    N,F=X.shape

    q1,q3=np.percentile(X, 25, axis=0),np.percentile(X, 75, axis=0)
    IQR=q3-q1

    lower,upper=q1-k*IQR,q3+k*IQR

    return (lower,upper)





    
    pass

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    
    x=np.copy(X)
    (N,F)=X.shape

    for c in range(F):
        low,high=lower[c],upper[c]
        for l in range(N):
            x[l][c]=min(high,max(X[l][c],low))

    return x
    
    pass

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    
    return numerator/(denominator+eps)

    pass

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.

    x=np.copy(X)
    col=col.reshape(-1, 1)

    return np.column_stack([x,col])
    
    pass

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.

    uniques=np.unique(np.asarray(labels).reshape(-1,1))

    result=labels[:, None] == uniques[None, :]

    (N,F)=result.shape
    result=result.astype(float)

    return result

    pass

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...

    mean,std=np.mean(X, axis=0),np.std(X,axis=0)
    
    std = np.where(std==0, 1.0, std)

    return(mean,std)
    pass

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.

    return (X-mean)/std
    pass

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...

    N=X.shape[0]

    bias=np.ones((N,1))

    return np.hstack([bias,X])
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.

    indices=np.arange(n_samples)
    rng=np.random.RandomState(seed)

    return rng.permutation(indices)
    pass

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.

    n=indices.shape[0]
    N_train=int(train_ratio*n)
    N_val=int(val_ratio*n)
    N_test=n-(N_train+N_val)

    train_idx,val_idx,test_idx=indices[0:N_train],indices[N_train:N_train+N_val],indices[N_train+N_val:]
    
    return (train_idx,val_idx,test_idx)
    pass

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.

    return(X[indices],y[indices])
    pass

# Step 13 - ols_fit
# ── Step 013  ols_fit ──
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    
    # On utilise np.linalg.lstsq au lieu de résoudre manuellement (X.T @ X)
    # rcond=None est la recommandation standard pour éviter les warnings futurs
    theta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.

    return X@theta

    pass

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions

    return np.mean(abs(y_true-y_pred))
    pass

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float

    return np.sqrt(np.mean((y_true-y_pred)**2))
    pass

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    
    mean=np.mean(y_true)
    SS_res,SS_tot=np.sum((y_true-y_pred)**2),np.sum((y_true-mean)**2)

    if SS_tot==0:
        return 0

    r=1-SS_res/SS_tot
    return r

    pass

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    
    D={}

    r=y_true-y_pred
    mean=np.mean(r)
    
    D['mean']=mean
    D['std']=np.sqrt(np.mean((r-mean)**2))
    D['median_abs']=np.median(abs(r))

    return D
    pass

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip

    Copy=np.copy(X)
    X_cleaned=impute_nan_with_mean(Copy)
    (low,up)=compute_iqr_bounds(X_cleaned,iqr_k)
    X_cleaned=clip_columns(X_cleaned,low,up)

    return X_cleaned
    pass

# Step 20 - assemble_feature_matrix
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...

    num,den=X_num[:,ratio_num_idx], X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num,den)
    X=append_column(X_num,ratio)

    if cat_labels is not None:
        labels=one_hot_encode(cat_labels)
        X=np.hstack([X,labels])

    return X


    pass

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...

    N=y.shape[0]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    D={}
    
    rng=np.random.default_rng(seed)
    indices = make_shuffled_indices(N, seed)
    D['X_train'],D['y_train']=subset_xy(X, y, indices[:n_train])
    D['X_val'],D['y_val']= subset_xy(X,y,indices[n_train:n_train+n_val])
    D['X_test'],D['y_test']= subset_xy(X,y,indices[n_train+n_val:])

    return D



    pass

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...

    mean,std=fit_standardizer(splits['X_train'])
    D = splits.copy()

    for key in ['X_train', 'X_val', 'X_test']:
        D[key] = add_bias_column(apply_standardizer(splits[key], mean, std))

    return (D,mean,std)
    

    pass

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    D={}
    #E=residual_summary(y_true,y_pred)
    #mean,std,median_abs=E['mean'],E['std'],E['median_abs']


    D['mae']=mean_absolute_error(y_true, y_pred)
    D['rmse']=root_mean_squared_error(y_true, y_pred)
    D['r2']=r_squared(y_true, y_pred)
    D['residual_summary']=residual_summary(y_true,y_pred)
    
    return D
    pass

# Step 24 - house_price_pipeline
# ── Step 024  house_price_pipeline ──
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    """Run full clean->featurize->split->standardize->OLS->evaluate pipeline."""
    
    # 1. Nettoyage (Imputation des NaNs et clipping IQR)
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    
    # 2. Ajout des features (Ratio + One-Hot encoding)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels)
    
    # 3. Séparation reproductible en train/val/test
    splits = make_train_val_test(X_feat, y, train_ratio, val_ratio, seed)
    
    # 4. Standardisation basée sur le train et ajout du biais
    std_splits, _, _ = standardize_and_add_bias(splits)
    
    # 5. Entraînement du modèle OLS sur les données train
    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])
    
    # 6. Prédictions sur la validation et le test
    y_val_pred = ols_predict(std_splits['X_val'], theta)
    y_test_pred = ols_predict(std_splits['X_test'], theta)

    # 7. Évaluation des métriques
    val_metrics = evaluate_predictions(splits['y_val'], y_val_pred)
    test_metrics = evaluate_predictions(splits['y_test'], y_test_pred)

    # 8. Retour du dictionnaire attendu
    return {
        'theta': theta,
        'y_test': splits['y_test'],
        'y_test_pred': y_test_pred,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

