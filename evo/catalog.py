"""The fixed, initially unimplemented vocabulary of robot atomic skills.

The catalog is a vocabulary and a contract, not a set of demonstration policies.
Argument values are scene bindings supplied by a planner, never baked into code.
"""

from __future__ import annotations

import copy


_ROWS = [
    ("P01", "pick", "抓取拿起", "pick_place", ("object",)),
    ("P02", "place", "放置释放", "pick_place", ("object", "target")),
    ("P03", "hold", "持续固定支撑", "pick_place", ("object",)),
    ("P04", "reorient", "物体调姿", "reorient", ("object", "orientation")),
    ("P05", "slide", "推移物体", "reorient", ("object", "target")),
    ("P06", "handover", "双臂交接", "reorient", ("object", "target")),
    ("P07", "articulate", "沿铰链/滑轨/盖机构开合或收折", "articulate", ("object", "target")),
    ("P08", "turn_knob", "旋转控制旋钮", "controls", ("object", "target")),
    ("P09", "turn_lever", "拨动杆件", "controls", ("object", "target")),
    ("P10", "press_button", "按压按钮", "controls", ("object",)),
    ("P11", "press_flatten", "压平或压合", "controls", ("object",)),
    ("P12", "screw", "旋拧螺纹件", "fasten", ("object", "target")),
    ("P13", "latch", "扣合或解开卡扣", "fasten", ("object", "target")),
    ("P14", "insert", "约束插入或就位", "insertion", ("object", "target")),
    ("P15", "extract", "从约束中取出或拆离", "insertion", ("object",)),
    ("P16", "pour", "倾倒转移内容物", "material", ("object", "target")),
    ("P17", "scoop", "舀取、铲取或捞取", "material", ("object", "target")),
    ("P18", "dip", "蘸取或润湿工具", "material", ("object", "target")),
    ("P19", "dispense", "挤出或喷洒物料", "material", ("object", "target")),
    ("P20", "drain", "沥水过滤", "material", ("object", "target")),
    ("P21", "stir", "容器内搅拌", "material", ("object", "target")),
    ("P22", "shake", "抖动或晃动", "material", ("object",)),
    ("P23", "wipe", "布类接触擦拭", "surface", ("object", "target")),
    ("P24", "brush", "刷涂或刷洗", "surface", ("object", "target")),
    ("P25", "squeegee", "刮拭集水", "surface", ("object", "target")),
    ("P26", "sweep", "扫拢收集", "surface", ("object", "target")),
    ("P27", "vacuum", "吸取表面碎屑", "surface", ("object", "target")),
    ("P28", "coat", "铺涂或摊开材料", "surface", ("object", "target")),
    ("P29", "cut", "工具切割", "surface", ("object", "target")),
    ("P30", "flip_with_tool", "借工具翻面", "reorient", ("object", "target")),
    ("P31", "scan", "持工具扫码", "surface", ("object", "target")),
    ("P32", "unfold", "展开铺平", "deformable", ("object",)),
    ("P33", "fold", "沿折线折叠", "deformable", ("object", "target")),
    ("P34", "roll", "卷起柔性物", "deformable", ("object",)),
    ("P35", "stretch", "撑开柔性开口", "deformable", ("object",)),
    ("P36", "invert", "内外翻转", "deformable", ("object",)),
    ("P37", "fit_cover", "袋口套装", "deformable", ("object", "target")),
    ("P38", "tear", "沿边界撕离", "deformable", ("object", "target")),
    ("P39", "attach", "贴附标签", "deformable", ("object", "target")),
    ("P40", "navigate", "移动底盘到达目标位", "navigation", ("target",)),
]


CATALOG = {
    atom: {
        "id": atom,
        "name": name,
        "description": description,
        "family": family,
        "parameters": {key: {"required": False, "recommended": True, "description": "Suggested runtime scene binding"} for key in parameters},
        "entrypoint": "skill(api, params) -> dict",
        "implemented": False,
    }
    for atom, name, description, family, parameters in _ROWS
}


def get_spec(atom: str) -> dict:
    """Return an isolated copy; unknown atomic IDs are always rejected."""
    if atom not in CATALOG:
        raise ValueError(f"Unknown atomic skill: {atom!r}; expected P01 through P40")
    return copy.deepcopy(CATALOG[atom])


def stub(atom: str) -> str:
    """Return an explicitly empty function, never a hand-authored policy."""
    spec = get_spec(atom)
    return (
        "def skill(api, params):\n"
        f'    """{atom}: {spec["name"]} — {spec["description"]}.\n\n'
        f'    Bindings: {", ".join(spec["parameters"])}.\n'
        '    Implement using only the primitive API and validated dependencies.\n'
        '    """\n'
        f'    raise NotImplementedError("{atom} has not been learned")\n'
    )


def validate_params(atom: str, params: dict, strict: bool = False) -> None:
    """Validate a binding dictionary; optional strict mode checks suggested names.

    A learned implementation may use more specific bindings. The adapter and
    replay postcondition validate actual scene semantics rather than key names.
    """
    spec = get_spec(atom)
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    missing = [key for key, value in spec["parameters"].items()
               if (value["required"] or strict) and key not in params]
    if missing:
        raise ValueError(f"{atom} missing required parameter(s): {', '.join(missing)}")
