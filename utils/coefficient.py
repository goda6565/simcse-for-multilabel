def jaccard_index(vec1, vec2):
    """Jaccard係数の計算"""
    # 交差部分: 両方が1のインデックスの数
    intersection = sum(1 for a, b in zip(vec1, vec2) if a == 1 and b == 1)

    # 和集合部分: 少なくとも1つが1のインデックスの数
    union = sum(1 for a, b in zip(vec1, vec2) if a == 1 or b == 1)

    # 和集合が空でない場合に計算
    return intersection / union if union != 0 else 0


def simpson_coefficient(vec1, vec2):
    """Simpson係数の計算"""
    # 交差部分: 両方が1のインデックスの数
    intersection = sum(1 for a, b in zip(vec1, vec2) if a == 1 and b == 1)

    # 小さい方の集合のサイズ
    min_size = min(sum(vec1), sum(vec2))

    # シンプソン係数の計算
    return intersection / min_size if min_size != 0 else 0


def dice_coefficient(vec1, vec2):
    """Dice係数の計算"""
    # 交差部分: 両方が1のインデックスの数
    intersection = sum(1 for a, b in zip(vec1, vec2) if a == 1 and b == 1)

    # ダイス係数の計算
    return (
        2 * intersection / (len(vec1) + len(vec2))
        if (len(vec1) + len(vec2)) != 0
        else 0
    )
