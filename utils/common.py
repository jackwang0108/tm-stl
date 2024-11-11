# Standard Library
from pathlib import Path
from collections import defaultdict
from typing import Any

# Third-Party Library

# Torch Library

# My Library


def make_tupleList(hierarchy_dict: dict[Any, list[Any]]) -> list[tuple[Any, Any]]:
    """
    将值为列表的字典转换为元组的列表。

    该函数接受一个字典，其中每个键映射到一个值的列表，并将其转换为元组的列表。每个元组包含列表中的一个项目及其对应的字典键。

    Args:
        hierarchy_dict (dict[Any, list[Any]]): 一个字典，键与值的列表相关联。

    Returns:
        list[tuple[Any, Any]]: 一个元组的列表，每个元组包含列表中的一个项目及其对应的键。
    """
    return [
        (item, key) for key, value_list in hierarchy_dict.items() for item in value_list
    ]


def make_hierarchyDict(tuple_list: list[tuple[Any, Any]]) -> dict[Any, list[Any]]:
    """
    将元组列表转换为值为列表的字典。

    该函数接受一个元组列表，每个元组包含一个项目及其对应的键，并将它们组织成一个字典。字典中的每个键映射到与该键相关联的项目列表。

    Args:
        tuple_list (list[tuple[Any, Any]]): 一个元组列表，每个元组包含一个项目及其对应的键。

    Returns:
        dict[Any, list[Any]]: 一个字典，其中每个键映射到与该键相关联的项目列表。
    """
    hierarchy_dict = defaultdict(list)

    for item, key in tuple_list:
        hierarchy_dict[key].append(item)

    return dict(hierarchy_dict)
