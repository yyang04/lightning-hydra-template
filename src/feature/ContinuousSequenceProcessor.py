class ContinuousSequenceProcessor:
    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.scaler = None
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)
        all_values = []
        sequence_lengths = []

        for sequence in data.dropna():
            if isinstance(sequence, (list, np.ndarray)):
                all_values.extend(sequence)
                sequence_lengths.append(len(sequence))

        self.stats['avg_sequence_length'] = np.mean(sequence_lengths) if sequence_lengths else 0
        self.stats['max_sequence_length'] = max(sequence_lengths) if sequence_lengths else 0

        if all_values:
            self.stats['mean'] = np.mean(all_values)
            self.stats['std'] = np.std(all_values)
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(all_values).reshape(-1, 1))

    def transform(self, data: pd.Series) -> List[List[float]]:
        max_len = self.config.get('max_sequence_length', 50)
        padding_value = self.config.get('padding_value', 0.0)

        processed_sequences = []
        for sequence in data:
            if pd.isna(sequence) or sequence is None:
                processed_sequences.append([padding_value] * max_len)
            else:
                if self.scaler is not None:
                    sequence_array = np.array(sequence).reshape(-1, 1)
                    standardized = self.scaler.transform(sequence_array).flatten().tolist()
                else:
                    standardized = sequence

                if len(standardized) < max_len:
                    standardized = standardized + [padding_value] * (max_len - len(standardized))
                else:
                    standardized = standardized[:max_len]
                processed_sequences.append(standardized)

        return processed_sequences

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'continuous_sequence',
            'stats': self.stats
        }