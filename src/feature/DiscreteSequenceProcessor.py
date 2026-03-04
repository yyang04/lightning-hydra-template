class DiscreteSequenceProcessor:
    def __init__(self, feature_name: str, config: Dict):
        self.feature_name = feature_name
        self.config = config
        self.sequence_type = True
        self.value_to_idx = {}
        self.stats = {}

    def fit(self, data: pd.Series):
        self.stats['null_count'] = data.isnull().sum()
        self.stats['null_ratio'] = self.stats['null_count'] / len(data)

        all_values = set()
        sequence_lengths = []

        for sequence in data.dropna():
            if isinstance(sequence, (list, np.ndarray)):
                all_values.update(sequence)
                sequence_lengths.append(len(sequence))

        self.stats['unique_count'] = len(all_values)
        self.stats['vocab_size'] = len(all_values) + 1
        self.stats['avg_sequence_length'] = np.mean(sequence_lengths) if sequence_lengths else 0
        self.stats['max_sequence_length'] = max(sequence_lengths) if sequence_lengths else 0
        self.value_to_idx = {value: idx + 1 for idx, value in enumerate(all_values)}

    def transform(self, data: pd.Series) -> List[List[int]]:
        max_len = self.config.get('max_sequence_length', 50)
        padding_value = self.config.get('padding_value', 0)

        processed_sequences = []
        for sequence in data:
            if pd.isna(sequence) or sequence is None:
                processed_sequences.append([padding_value] * max_len)
            else:
                encoded = [self.value_to_idx.get(x, 0) for x in sequence]
                if len(encoded) < max_len:  # 需要在后面填充 padding value
                    encoded = encoded + [padding_value] * (max_len - len(encoded))
                else:
                    encoded = encoded[:max_len]
                processed_sequences.append(encoded)
        return processed_sequences

    def get_meta_data(self) -> Dict[str, Any]:
        return {
            'processor_type': 'discrete_sequence',
            'stats': self.stats,
            'value_mapping': self.value_to_idx
        }