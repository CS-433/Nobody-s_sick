"""Some data preprocessing functions for project 1."""

import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """De-standardize to the original data set."""
    return x * std_x + mean_x

def one_hot_encode(categories):
    """
    Creates a binary variable for a specific category or one-hot encodings for all categories.

    Parameters:
    - categories (np.array): Array of categorical data.
    Returns:
    - np.array: A binary variable array or one-hot encoded array.
    """

    unique_categories = np.unique(categories)
    one_hot_encoded = np.zeros((categories.size, unique_categories.size), dtype=int)

    for i, category in enumerate(unique_categories):
        one_hot_encoded[:, i] = np.where(categories == category, 1, 0)

    # avoid the dummy variable trap 
    one_hot_encoded = one_hot_encoded[:,1:]
    unique_categories = unique_categories[1:]
    
    return one_hot_encoded, unique_categories

def clean_continuous_data(X, feature_names, continuous_features_idx):
    '''
        Clean continuous (2 categories) data, given knoledge of context-specific information 
    '''
    idx_to_clean = continuous_features_idx
    data = np.copy(X[:, idx_to_clean])
    data[(data == 88)] = 0 # 88 -> 0
    data[(data == 888)] = 0 # 888 -> 0
    data[(data == 77) | (data == 99)] = -1 # not answered
    data[(data == 777) | (data == 999)] = -1 # not answered

    for col in range(data.shape[1]):
        # Calculate Q1 and Q3 for the column
        Q1 = np.percentile(data[:, col], 25)
        Q3 = np.percentile(data[:, col], 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers for this column
        data[(data[:, col] < lower_bound) | (data[:, col] > upper_bound)] = np.nan

    X[:, idx_to_clean] = data
    return X

def clean_binary_data(X, feature_names, binary_features_idx):
    '''
        Clean binary (2 categories) data, given knoledge of context-specific information 
    '''
    idx_to_clean = binary_features_idx
    data = X[:, idx_to_clean]
    
    data[(data == 9) | (data == 7)] = -1 # not answered

    X[:, idx_to_clean] = data  
    
    return X
    

def clean_categorical_data(X, feature_names, categorical_features_idx):
    '''
        Clean categorical (more than two categories) data, given knoledge of context-specific information 
    '''
    idx_to_clean = categorical_features_idx
    
    data = np.copy(X[:, idx_to_clean])

    data[(data == 9) | (data == 7)] = -1 # not answered
    data[(data == 99) | (data == 77)] = -1 # not answered
    data[data == 8] = 0 # 8 -> 0
    data[data == 88] = 0 # 8 -> 0
    
    X[:, idx_to_clean] = data

    return X

def cat_vs_cont(X):
    
    unique_data_per_feature = np.array([np.unique(X[~np.isnan(X)[:,col], col]).size for col in range(X.shape[1])])

    continuous_features_idx = np.where(unique_data_per_feature > 20)[0] 
    categorical_features_idx = np.where((unique_data_per_feature > 4) & (unique_data_per_feature <= 20))[0]
    binary_features_idx = np.where(unique_data_per_feature <= 4)[0]
    
    return continuous_features_idx, categorical_features_idx, binary_features_idx
    
def clean_data(X, feature_names, features_to_keep):

    # Limit the predictors to a given subset of features to simplify the interpretation of results
    features_to_keep_logical = np.isin(feature_names, features_to_keep)
    if features_to_keep_logical.any():
        X = X[:, features_to_keep_logical]
        feature_names = feature_names[features_to_keep_logical]

    features_to_discard = ['_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE','SEQNO', '_PSU', # irrelevant questionnaire data
                           '_RAWRAKE', '_WT2RAKE', '_DUALUSE', '_LLCPWT', '_RFHLTH','_HCVU651', # irrelevant questionnaire data
                           'BPHIGH4', 'BLOODCHO', 'ASTHMA3', 'HAVARTH3','EDUCA', 'CHILDREN', 'INCOME2', 'STRENGTH', 'WEIGHT2', 'HEIGHT3', 'USENOW3','ALCDAY5', 'FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG','VEGETAB1', 'SEATBELT', 'QSTVER', 'QSTLANG', '_STSTR', '_STRWT', '_CASTHM1', '_ASTHMS1', '_PRACE1', '_RACE', '_RACEG21','_RACE_G1', '_AGEG5YR', '_AGE65YR', '_AGE_G', 'HTIN4', '_BMI5','_RFSMOK3', 'DRNKANY5', '_DRNKWEK', '_MISFRTN', '_MISVEGN','_FRTRESP', '_VEGRESP', '_FRT16', '_VEG23', '_FRUITEX', '_VEGETEX','PAMISS1_', '_PA150R2', '_PA300R2', '_PA30021'] # redundant information

    features_to_keep = [feature for feature in feature_names if feature not in features_to_discard]
    features_to_keep_logical = np.isin(feature_names, features_to_keep)

    X = X[:, features_to_keep_logical]
    feature_names = feature_names[features_to_keep_logical]

    # Distinguish between continuous and categorical features and clean accordingly 
    continuous_features_idx, categorical_features_idx, binary_features_idx = cat_vs_cont(X)

    # Clean data
    if continuous_features_idx.size > 0:
        X = clean_continuous_data(X, feature_names, continuous_features_idx)
    if categorical_features_idx.size > 0:
        X = clean_categorical_data(X, feature_names, categorical_features_idx)
    if binary_features_idx.size > 0:
        X = clean_binary_data(X, feature_names, binary_features_idx)

    # 1. Fill missing data with median and mode respectively for continuous and categorical data
    if (categorical_features_idx.size > 0) or (binary_features_idx.size > 0):
        categorical_features_idx = np.concatenate([categorical_features_idx, binary_features_idx]).ravel()
        data = X[:,categorical_features_idx]

        # treat "not answered as nan" 
        data[data == -1] = np.nan

        for col in range(data.shape[1]):
            
            column_data = data[:, col]
        
            # fill with mode
            values, counts = np.unique(column_data[~np.isnan(column_data)], return_counts=True)
            mode_value = values[np.argmax(counts)]
                
            data[np.isnan(data[:, col]), col] = mode_value
    
        X[:,categorical_features_idx] = data
    
    if continuous_features_idx.size > 0:
        data = X[:,continuous_features_idx]

        # treat "not answered as nan"
        data[data == -1] = np.nan
        
        for col in range(data.shape[1]):
            
            column_data = data[:, col]
            
            # compute median
            median_value = np.median(column_data[~np.isnan(column_data)])
            data[np.isnan(column_data), col] = median_value

        X[:,continuous_features_idx] = data
    return X, feature_names, continuous_features_idx, categorical_features_idx

def pca_with_n_components(data, n_components):
    """
    Perform PCA and return the indices of the top n_components.
    
    Parameters:
        data (np.ndarray): The input data (n_samples x n_features).
        n_components (int): The number of principal components to keep.
    
    Returns:
        np.ndarray: Indices of the selected principal components.
    """
    
    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    # Step 3: Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Retrieve the indices of the top n_components
    significant_components_indices = sorted_indices[:n_components]
    
    return significant_components_indices
    
def pca_with_variance_threshold(data, variance_threshold=0.80):
    
    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    # Step 3: Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    
    # Step 5: Calculate explained variance ratio
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Step 6: Find the number of components needed to meet the variance threshold
    num_components = np.argmax(cumulative_explained_variance >= variance_threshold) + 1
    
    # Step 7: Retrieve the indices of the significant components
    significant_components_indices = sorted_indices[:num_components]
    
    return significant_components_indices

def mca_with_kaiser(data):
    # Step 2: Compute the Correspondence Matrix
    # Create the contingency table
    contingency_table = np.dot(data.T, data)

    # Step 3: Perform Eigen Decomposition
    # Normalize the contingency table
    n = data.shape[0]
    norm_table = contingency_table / n

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(norm_table)

    # Step 4: Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Apply the Kaiser criterion to select significant components
    significant_components_indices = np.where(eigenvalues > 1)[0]
    return significant_components_indices

def mca_with_n_components(data, n_components):
    """
    Perform Multiple Correspondence Analysis (MCA) and return the indices of the top n_components.
    
    Parameters:
        data (np.ndarray): The input data (n_samples x n_features) in the form of dummy variables.
        n_components (int): The number of components to keep.
    
    Returns:
        np.ndarray: Indices of the selected components.
    """
    
    # Step 2: Compute the Correspondence Matrix
    # Create the contingency table
    contingency_table = np.dot(data.T, data)

    # Step 3: Perform Eigen Decomposition
    # Normalize the contingency table
    n = data.shape[0]
    norm_table = contingency_table / n

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(norm_table)

    # Step 4: Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Retrieve the indices of the top n_components
    significant_components_indices = sorted_indices[:n_components]
    
    return significant_components_indices

def feature_selection(X_cat, X_cont, n_comp_cat, n_comp_cont):

    # Feature selection based on significance of predictors 
    selected_cat_features_idx = mca_with_n_components(X_cat, n_comp_cat)
    selected_cont_features_idx = pca_with_n_components(X_cont, n_comp_cont)
    
    return selected_cat_features_idx, selected_cont_features_idx

def feature_selection_no_given_n_components(X_cat, X_cont):

    # Feature selection based on significance of predictors 
    selected_cat_features_idx = mca_with_kaiser(X_cat)
    selected_cont_features_idx = pca_with_variance_threshold(X_cont)
    
    return selected_cat_features_idx, selected_cont_features_idx

def preprocess_data(X, feature_names, categorical_features_idx, continuous_features_idx):
        
    # Standardize and encode respectively continuous and categorical data
    if categorical_features_idx.size > 0:
        data = X[:,categorical_features_idx]

        X_encoded = []
        feature_cat_map = []
        feature_cat_encoded_map = []
        for col in range(data.shape[1]):
            
            column_data = data[:, col]
        
            # encode
            one_hot_encoded, unique_categories = one_hot_encode(column_data)
            X_encoded.append(one_hot_encoded)
            feature_cat_map.append((col * np.ones(len(unique_categories))).astype(int))
            feature_cat_encoded_map.append((np.arange(len(unique_categories))).astype(int))
        X_encoded = np.hstack(X_encoded)
        feature_cat_map = np.hstack(feature_cat_map)
        feature_cat_encoded_map = np.hstack(feature_cat_encoded_map)
    else:
        X_encoded = []
        unique_categories_list = []
    
    if continuous_features_idx.size > 0:
        data = X[:,continuous_features_idx]
        
        X_standardized = np.zeros(np.shape(data))
        mean_X = np.zeros(np.shape(data)[1])
        std_X = np.zeros(np.shape(data)[1])
        feature_cont_map = np.zeros(np.shape(data)[1])
        for col in range(data.shape[1]):
            
            column_data = data[:, col]
    
            # standardize
            X_standardized[:,col], mean_X[col], std_X[col] = standardize(column_data)
        feature_cont_map = (np.arange(np.shape(data)[1])).astype(int)
    else:
        X_standardized = []
        mean_X = []
        std_X = []
        feature_cont_map = []
 
    return X_encoded, feature_cat_map, feature_cat_encoded_map, X_standardized, feature_cont_map, mean_X, std_X

def build_model_data(y,x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx