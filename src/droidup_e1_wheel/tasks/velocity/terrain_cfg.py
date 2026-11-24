"""MuJoCo counterparts to the six RobotLab rough terrain types."""

from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)


def e1_wheel_rough_terrain_cfg(*, play: bool) -> TerrainEntityCfg:
    size = (8.0, 8.0)
    stairs = {
        "proportion": 0.2,
        "size": size,
        "step_height_range": (0.05, 0.23),
        "step_width": 0.3,
        "platform_width": 3.0,
        "border_width": 1.0,
    }
    slopes = {
        "proportion": 0.1,
        "size": size,
        "slope_range": (0.0, 0.4),
        "platform_width": 2.0,
        "border_width": 0.25,
    }
    return TerrainEntityCfg(
        terrain_type="generator",
        max_init_terrain_level=5,
        terrain_generator=TerrainGeneratorCfg(
            curriculum=not play,
            size=size,
            border_width=10.0 if play else 20.0,
            num_rows=5 if play else 10,
            num_cols=5 if play else 20,
            sub_terrains={
                "pyramid_stairs": BoxPyramidStairsTerrainCfg(**stairs),
                "pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(**stairs),
                "boxes": BoxRandomGridTerrainCfg(
                    size=size,
                    proportion=0.2,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.2),
                    platform_width=2.0,
                    border_width=0.0,
                ),
                "random_rough": HfRandomUniformTerrainCfg(
                    size=size,
                    proportion=0.2,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(**slopes),
                "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
                    **slopes, inverted=True
                ),
            },
        ),
    )
