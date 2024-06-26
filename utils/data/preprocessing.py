from _utils import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import skew


class Preprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for data records.

    Args:
        meta: Metadata.

    Attributes:
        meta: Metadata.
        params: Parameters for preprocessing.
        fit: Whether to fit or not.
    """
    def __init__(self, meta: pd.DataFrame) -> None:
        self.meta   = copy(meta)
        self.params = {}
        self.fit    = None

    @T
    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform data records.

        Args:
            X: Data records.
            y: Target values.

        Returns:
            Transformed data records.
        """
        self.fit = True
        return self._process(X)

    @T
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data records.

        Args:
            X: Data records.

        Returns:
            Transformed data records.
        """
        self.fit = False
        return self._process(X)

    def _process(self, data_: pd.DataFrame) -> pd.DataFrame:
        """Process data records.

        Args:
            data_: Data records.

        Returns:
            Processed data records.
        """
        meta = self.meta
        data = copy(data_)

        # Replace options and impute
        self._replace_ambiguous_options(data, meta)

        # Impute
        self._impute_data(data, meta)

        # Change dtypes
        self._set_dtypes(data, meta)

        # Impute numerical features
        self._impute_numerical_features(data, meta)

        # Drop constant / redundant / imbalance columns
        self._drop_columns(data)

        # Manual handling
        self._manual_handling(data, meta)

        # Except diabetes-relevant columns
        self._drop_diabetes_columns(data, meta)

        # Standardize numerical features
        self._standardize(data)

        # Label Encoding
        self._label_encoding(data)

        return data

    @T
    def _replace_ambiguous_options(self, data, meta):
        """Replace ambiguous options into 'unknown`
        - don't know
        - not ascertained
        - refused
        - etc
        """
        replacement_map = {src: "unknown" for src in ("don't know", "don’t know", "not ascertained", "refused", "not available", "time period format", "undefined", "undefinable")}

        for col in data:
            # 1. Replace options with words ('1' -> 'yes')
            options   = get_meta_values(meta, col)['options']  # {'1': 'yes', '2': 'no', '3': 'don\'t know', '4': 'not ascertained', '5': 'refused', '-1': 'unknown'}
            data[col] = data[col].replace(options)

            # 2. Replace ambiguous options with 'unknown'
            data[col] = data[col].replace(replacement_map)

            # 3. Replace words with options (restore)
            options_inversed = inverse_dict(options)           # {'yes': '1', 'no': '2', 'don\'t know': '3', 'not ascertained': '4', 'refused': '5', 'unknown': '-1'}
            data[col] = data[col].replace(options_inversed)

        if self.fit:
            # 4. Update metadata
            for col in data:
                options = get_meta_values(meta, col)['options']
                updated_options = {key: val for key, val in options.items() if val not in replacement_map}
                meta.loc[meta['final_id'] == col, 'options'] = json.dumps(updated_options)
            meta['options'] = meta['options'].map(lambda v: eval(v) if isinstance(v, str) else v)

    @T
    def _impute_data(self, data, meta):
        """Impute nan values with 'unknown' value"""
        for col in data:
            data[col] = data[col].fillna(get_unknown_value(meta, col))

    @T
    def _set_dtypes(self, data, meta):
        # Numerical features: only have '-1'('unknown') or '-' in option or YEAR
        if self.fit:
            num_features = []
            for col in data:
                options = get_meta_values(meta, col)['options']
                if list(options) == ['-1']:
                    num_features.append(col)
                else:
                    for key in options:
                        if re.search("^[0-9]+-[0-9]+$", key):
                            num_features.append(col)
                            break

            # Manual filtering
            # Append features
            num_features_app  = [PARAMS.target]  # drop unknown options -> binary label (yes or no)
            num_features_app += [col for col in ('ASISAD', 'ASIMUCH') if col in data]
            for words in ("how satisfied", "total number", "get sick or have accident", "days 5+/4+ drinks", "alcohol drinking", "strength activity", "freq ", "work status", "confidence", "time since", "received calls", "cost of", "education of", "total combined", "ratio of", "how difficult", "duration of", "agree/disagree", "length", "how worried", "how often", "how long", "time ago", "most recent", "year of", "time period", "degree of difficulty", "diff ", "frequency"):
                num_features_app += [col for col in data if get_meta_values(meta, col)['description'].startswith(words)]
            for final_id in ("RPAP", "HEAR_", "COG_"):
                num_features_app += [col for col in data if get_meta_values(meta, col)['final_id'].startswith(final_id)]

            num_features = list(set(num_features + num_features_app))

            # Except features
            num_features = [col for col in num_features if col not in ('HHX', 'FMX', 'FPX', 'SRVY_YR', 'OCCUPN1', 'OCCUPN2', 'INDSTRN1', 'INDSTRN2')]
            
            # Categorical features: otherwise
            cat_features = data.columns.drop(num_features)

            self.params['num_features'] = num_features
            self.params['cat_features'] = cat_features

        # Apply
        for col in self.params['num_features']:  # error at once
            data[col] = data[col].astype('float')
        data[self.params['cat_features']] = data[self.params['cat_features']].astype('str')

    @T
    def _impute_numerical_features(self, data, meta):
        """Fill numeric unknown values with mode (zero-centered or long tailed distribution)"""
        for col in data.select_dtypes('number').columns:
            data[col] = data[col].replace(float(get_unknown_value(meta, col)), None)
            if self.fit:
                if 'mode' not in self.params:
                    self.params['mode'] = {}
                self.params['mode'][col] = data[col].mode()[0]
                mode = self.params['mode'][col]
            else:
                if col in self.params['mode']:
                    mode = self.params['mode'][col]
                else:
                    mode = data[col].mode()[0]  # use mode from test data
            data[col] = data[col].fillna(mode)  # fill with mode

    @T
    def _drop_columns(self, data):
        # Drop constant columns
        cnts = data.nunique().sort_values()
        if self.fit:
            self.params['constant_cols'] = cnts[cnts == 1].index
        constant_cols = self.params['constant_cols']
        data.drop(columns=constant_cols, inplace=True)

        # Drop redundant columns
        redundant_cols = ['HHX', 'FMX', 'INTV_QRT']
        cols = [col for col in redundant_cols if col in data]
        data.drop(columns=cols, inplace=True)

        # *CAUTION* Drop imbalanced distributed columns
        # imbalance_cols = []
        # for col in data:
        #     if col == 'label':  # except label
        #         continue
        #     max_cnt = data[col].value_counts(normalize=True).values[0]
        #     if max_cnt >= 0.85:
        #         imbalance_cols.append(col)
        # data.drop(columns=imbalance_cols, inplace=True)

    @T
    def _manual_handling(self, data, meta):
        """Handle 'do nothing' options"""
        for col in data:
            if col in (PARAMS.target, 'family_id'):
                continue
            options = get_meta_values(meta, col)['options']
            options_inversed = inverse_dict(options)
            for option in options_inversed:
                if re.search("^unable to do ", option) or (option == 'never'):
                    value = float(options_inversed[option])
                    data[col] = data[col].replace(value, 0)  # never do that

    @T
    def _drop_diabetes_columns(self, data, meta):
        """Drop columns which have diabetes keywords except target"""
        idxs = meta['keywords'].astype(str).str.contains('diabetes')
        diabetes_cols = meta.loc[idxs[idxs].index, 'final_id']
        diabetes_cols = [col for col in diabetes_cols if col in data]
        diabetes_cols.remove(PARAMS.target)
        data.drop(columns=diabetes_cols, inplace=True)

    @T
    def _standardize(self, data):
        """Standardize numerical features"""
        num_features = data.select_dtypes(exclude='object').columns.drop(PARAMS.target)  # except target
        if self.fit:
            self.params['std_scaler'] = StandardScaler()
            self.params['std_scaler'].fit(data[num_features])
        data[num_features] = self.params['std_scaler'].transform(data[num_features])

    @T
    def _label_encoding(self, data):
        """Encode label into [0, 1]"""
        if self.fit:
            self.params['label_encoder'] = LabelEncoder()
            self.params['label_encoder'].fit(data[PARAMS.target])
        data[PARAMS.target] = self.params['label_encoder'].transform(data[PARAMS.target])

    def get_metadata(self):
        """Get metadata"""
        return self.meta
    
    
    
    
    
    
    ## data transformation for tabs ##
    
    @T
    def set_dtypes(self, data, meta):
        # Numerical features: only have '-1'('unknown') or '-' in option or YEAR
        if self.fit:
            num_features = []
            for col in data:
                options = get_meta_values(meta, col)['options']
                if list(options) == ['-1']:
                    num_features.append(col)
                else:
                    for key in options:
                        if re.search("^[0-9]+-[0-9]+$", key):
                            num_features.append(col)
                            break

            # Manual filtering
            # Append features
            if PARAMS.target in data.columns:
                num_features_app  = [PARAMS.target]  # drop unknown options -> binary label (yes or no)
            elif PARAMS.target2 in data.columns:
                num_features_app  = [PARAMS.target2]
            else:
                num_features_app  = []
            num_features_app += [col for col in ('ASISAD', 'ASIMUCH') if col in data]
            for words in ("how satisfied", "total number", "get sick or have accident", "days 5+/4+ drinks", "alcohol drinking", "strength activity", "freq ", "work status", "confidence", "time since", "received calls", "cost of", "education of", "total combined", "ratio of", "how difficult", "duration of", "agree/disagree", "length", "how worried", "how often", "how long", "time ago", "most recent", "year of", "time period", "degree of difficulty", "diff ", "frequency"):
                num_features_app += [col for col in data if get_meta_values(meta, col)['description'].startswith(words)]
            for final_id in ("RPAP", "HEAR_", "COG_"):
                num_features_app += [col for col in data if get_meta_values(meta, col)['final_id'].startswith(final_id)]

            num_features = list(set(num_features + num_features_app))

            # Except features
            num_features = [col for col in num_features if col not in ('HHX', 'FMX', 'FPX', 'SRVY_YR', 'OCCUPN1', 'OCCUPN2', 'INDSTRN1', 'INDSTRN2')]
            
            # Categorical features: otherwise
            cat_features = data.columns.drop(num_features)

            self.params['num_features'] = num_features
            self.params['cat_features'] = cat_features

        # Apply
        for col in self.params['num_features']:  # error at once
            data[col] = data[col].astype('float')
        data[self.params['cat_features']] = data[self.params['cat_features']].astype('str')

        return data
    
    @T
    def replace_ambiguous_options(self, data, meta):
        """Replace ambiguous options into 'unknown`
        - don't know
        - not ascertained
        - refused
        - etc
        """
        replacement_map = {src: "unknown" for src in ("don't know", "don’t know", "not ascertained", "refused", "not available", "time period format", "undefined", "undefinable")}

        for col in data:
            # 1. Replace options with words ('1' -> 'yes')
            options   = get_meta_values(meta, col)['options']  # {'1': 'yes', '2': 'no', '3': 'don\'t know', '4': 'not ascertained', '5': 'refused', '-1': 'unknown'}
            data[col] = data[col].replace(options)

            # 2. Replace ambiguous options with 'unknown'
            data[col] = data[col].replace(replacement_map)

            # 3. Replace words with options (restore)
            options_inversed = inverse_dict(options)           # {'yes': '1', 'no': '2', 'don\'t know': '3', 'not ascertained': '4', 'refused': '5', 'unknown': '-1'}
            data[col] = data[col].replace(options_inversed)

        if self.fit:
            # 4. Update metadata
            for col in data:
                options = get_meta_values(meta, col)['options']
                updated_options = {key: val for key, val in options.items() if val not in replacement_map}
                meta.loc[meta['final_id'] == col, 'options'] = json.dumps(updated_options)
            meta['options'] = meta['options'].map(lambda v: eval(v) if isinstance(v, str) else v)
            
        return data, meta
    
    @T
    def impute_data(self, data, meta):
        """Impute nan values with 'unknown' value"""
        for col in data:
            data[col] = data[col].fillna(get_unknown_value(meta, col))
        
        return data
    
    @T        
    def impute_numerical_features(self, data, meta):
        """Fill numeric unknown values with mode (zero-centered or long tailed distribution)"""
        for col in data.select_dtypes('number').columns:
            data[col] = data[col].replace(float(get_unknown_value(meta, col)), None)
            if self.fit:
                if 'mode' not in self.params:
                    self.params['mode'] = {}
                self.params['mode'][col] = data[col].mode()[0]
                mode = self.params['mode'][col]
            else:
                if col in self.params['mode']:
                    mode = self.params['mode'][col]
                else:
                    mode = data[col].mode()[0]  # use mode from test data
            data[col] = data[col].fillna(mode)  # fill with mode
        
        return data
    
    @T
    def drop_columns(self, data):
        # Drop constant columns
        cnts = data.nunique().sort_values()
        if self.fit:
            self.params['constant_cols'] = cnts[cnts == 1].index
        constant_cols = self.params['constant_cols']
        data.drop(columns=constant_cols, inplace=True)

        # Drop redundant columns
        redundant_cols = ['HHX', 'FMX', 'INTV_QRT', 'FPX', 'SRVY_YR']
        cols = [col for col in redundant_cols if col in data]
        data.drop(columns=cols, inplace=True)
        
        return data
    
    @T
    def manual_handling(self, data, meta):
        """Handle 'do nothing' options"""
        for col in data:
            if col in (PARAMS.target, PARAMS.target2, 'family_id'):
                continue
            options = get_meta_values(meta, col)['options']
            options_inversed = inverse_dict(options)
            for option in options_inversed:
                if re.search("^unable to do ", option) or (option == 'never'):
                    value = float(options_inversed[option])
                    data[col] = data[col].replace(value, 0)  # never do that
                    
        return data
    
    @T
    def drop_diabetes_columns(self, data, meta):
        """Drop columns which have diabetes keywords except target"""
        idxs = meta['keywords'].astype(str).str.contains('diabetes')
        diabetes_cols = meta.loc[idxs[idxs].index, 'final_id']
        diabetes_cols = [col for col in diabetes_cols if col in data]
        if PARAMS.target in diabetes_cols:
            diabetes_cols.remove(PARAMS.target)
        if PARAMS.target2 in diabetes_cols:
            diabetes_cols.remove(PARAMS.target2)
        data.drop(columns=diabetes_cols, inplace=True)
        
        return data
    
    @T
    def standardize(self, data, target=None):
        """Standardize numerical features"""
        if target != None:
            num_features = data.select_dtypes(exclude='object').columns.drop(target)  # except target
        elif target == None:
            num_features = data.select_dtypes(exclude='object').columns
            
        if self.fit:
            self.params['std_scaler'] = StandardScaler()
            self.params['std_scaler'].fit(data[num_features])
        data[num_features] = self.params['std_scaler'].transform(data[num_features])
        
        return data
    
