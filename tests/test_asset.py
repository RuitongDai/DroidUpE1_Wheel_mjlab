import numpy as np
import pytest
from mjlab.entity import Entity

from droidup_e1_wheel.assets.e1_wheel.e1_wheel_constants import (
    LEG_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    get_e1_wheel_robot_cfg,
    get_spec,
)


def test_e1_wheel_mjcf_preserves_mass_joints_and_collision_filters():
    spec = get_spec()
    model = spec.compile()
    assert model.body_mass.sum() == pytest.approx(24.52)
    assert model.nq == 27 and model.nv == 26  # Floating base and 20 hinges.
    for name in LEG_JOINT_NAMES:
        assert model.joint(name).limited.item()
    for name in WHEEL_JOINT_NAMES:
        joint = model.joint(name)
        assert not joint.limited.item()
        np.testing.assert_allclose(joint.axis, (0.0, 1.0, 0.0))
    assert model.geom("base_link_collision").conaffinity.item() == 1
    assert model.geom("base_link_collision").contype.item() == 0
    assert model.geom("FL_hip_roll_link_visual").conaffinity.item() == 0
    assert not any(geom.name.endswith("hip_roll_link_collision") for geom in spec.geoms)


def test_e1_wheel_has_independent_position_and_velocity_actuators():
    entity = Entity(get_e1_wheel_robot_cfg())
    model = entity.compile()
    assert model.nu == 20
    for name in LEG_JOINT_NAMES:
        actuator = model.actuator(name)
        assert actuator.gainprm[0] > 0
        assert actuator.biasprm[1] == -actuator.gainprm[0]
    for name in WHEEL_JOINT_NAMES:
        actuator = model.actuator(name)
        assert actuator.gainprm[0] == pytest.approx(0.5)
        assert actuator.biasprm[1] == 0.0
        assert actuator.biasprm[2] == pytest.approx(-0.5)
        np.testing.assert_allclose(actuator.forcerange, (-60.0, 60.0))
