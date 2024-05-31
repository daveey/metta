# from env.griddly.builder.object import GriddlyObject, Init, Reset


# class Charger(GriddlyObject):
#     def __init__(self, **cfg):
#         super().__init__(
#             name = "charger",
#             symbol = "c",
#             images = [
#                 "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_pda_A.png",
#                 "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_pda_B.png",
#                 "oryx/oryx_tiny_galaxy/tg_sliced/tg_items/tg_items_pda_C.png",
#             ],
#             features = [
#                 Init([
#                     SetVar("charger:input:1", "meta.i1"),
#                     SetVar("charger:input:2", "meta.i2"),
#                     SetVar("charger:input:3", "meta.i3"),
#                     SetVar("charger:output", cfg.energy),
#                     If(["meta.bonus", 1], [
#                         AddVar("charger:output", 2*cfg.energy),
#                         SetVar("charger:bonus", cfg.energy)
#                     ]),
#                 ], choices = [
#                     {"i1": 1, "i2": 0, "i3": 0, "bonus": 0},
#                     {"i1": 0, "i2": 1, "i3": 0, "bonus": 0},
#                     {"i1": 0, "i2": 0, "i3": 1, "bonus": 0},
#                     {"i1": 0, "i2": 0, "i3": 0, "bonus": 1}
#                 ]),
#                 Reset([
#                     SetVar("charger:ready", 1),
#                     SetTile(0)
#                 ]),
#             ],
#             properties = {
#                 "energy": 0,
#                 "input:1": 0,
#                 "input:2": 0,
#                 "input:3": 0,
#                 "output": 0,
#                 "bonus": 0
#             },
#             **cfg
#         )

