class Vocabulary(object):
    """
    Класс для обработки текста и извлечения словарного запаса для маппинга
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        :param token_to_idx (dict): существующий маппинг токенов в индексы
        :param add_unk (bool): флаг, указывающий добавлять ли токен UNK
        :param unk_token (str): UNK-токен для добавления в словарь
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """
        :return: словарь, который может быть сериализован
        """
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }

    @classmethod
    def from_serializable(cls, contents):
        """
        :param contents: сериализованыый словарь
        :return: словарь и сериализованного словаря (десериализация)
        """
        return cls(**contents)

    def add_token(self, token):
        """
        Обновляет маппинги
        :param token: (str) значение для добавления в словарь
        :return: index: (int) число, соответствующее токену
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """
        Добавляет в словарь список токенов
        :param tokens: (list<str>) список строковых токенов
        :return: indices: (list<int>) список индексов, соответствующих токенам
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Возвращает индекс токена или индекс UNK если токена нет
        `unk_index` должен быть >=0 (добавлен в словарь)
        :param token: (str) токен, который нужно найти
        :return: index (int) индекс токена
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """
        Возвращает токен по индексу
        :param index: (int) индекс, который нкжно найти
        :return: token (str)
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)